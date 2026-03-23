
import os
import sys
sys.path.append(os.getcwd())
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline import StableDiffusionControlNetPipeline
from utils.misc import load_dreambooth_lora
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from dataloaders.paired_dataset_indoor import PairedCaptionDataset3
logger = get_logger(__name__, log_level="INFO")


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 512)), 
            ])

def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)


def load_vttac_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel
    
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.vttac_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.vttac_model_path, subfolder="controlnet")
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def apply_motion_blur(image, kernel_size=10, angle=45):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = 1.0
    kernel = ndimage.rotate(kernel, angle, reshape=False, order=1)
    kernel = kernel / np.sum(kernel)
    image_np = np.array(image).astype(np.float32)
    if len(image_np.shape) == 3:
        blurred_channels = []
        for i in range(image_np.shape[2]):
            chan = ndimage.convolve(image_np[:, :, i], kernel, mode='nearest')
            blurred_channels.append(chan)
        blurred_image = np.stack(blurred_channels, axis=2)
    else:
        blurred_image = ndimage.convolve(image_np, kernel, mode='nearest')
    
    return Image.fromarray(np.clip(blurred_image, 0, 255).astype(np.uint8))

from torch.nn.functional import interpolate
def main(args, enable_xformers_memory_efficient_attention=True):
    txt_path = os.path.join(args.output_dir, 'txt')
    os.makedirs(txt_path, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("vttac")

    pipeline = load_vttac_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)

    from transformers import CLIPModel

    clip_model = CLIPModel.from_pretrained('/clip_encoder/clip_vit_L_14')
    clip_model = clip_model.eval()
    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained('/clip_encoder/clip_vit_L_14')
    clip_model = clip_model.to(accelerator.device)  

    def load_and_preprocess_image(image):
        inputs = processor(images=image, return_tensors="pt")
        return inputs["pixel_values"]

    def get_image_embeddings(model, pixel_values, output_attentions=True, output_hidden_states=True):
        model.eval()
        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=False
            )
            
            last_hidden_state = vision_outputs[0]  
            pooler_output = vision_outputs[1]     
            #hidden_states = vision_outputs[2]     
            #attentions = vision_outputs[3]        
            
            image_embeds = model.visual_projection(pooler_output)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            
        return {
            "vision_outputs": vision_outputs,
            "image_embeds": image_embeds,
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooler_output
            #"hidden_states": hidden_states,
            #"attentions": attentions
        }
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)
        test_dataset = PairedCaptionDataset3()
        for image_idx, example in enumerate(test_dataset):
            print(f'================== process {image_idx} imgs... ===================')
            #xy 测试图片数据2
            image_name = example["vision"] 
            if args.datasets == 'tac':
                image_item = example["item_name"]
                output_dir="/output_tacquad/output"
                save_dir = os.path.join(output_dir, image_item)
                os.makedirs(save_dir, exist_ok=True)
            elif args.datasets == 'ssvtp': 
                image_item = example["item_name"]
                output_dir="/output_ssvtp/output"
                save_path = os.path.join(output_dir, image_item)
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
            elif args.datasets == 'hct': 
                image_item = example["item_name"]
                output_dir="/output_hct/output/"
                save_path = os.path.join(output_dir, image_item)
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
            validation_image = Image.open(image_name).convert("RGB")
            if args.move_blur:
                validation_image = apply_motion_blur(validation_image, kernel_size=30, angle=45)
            clip_image = load_and_preprocess_image(validation_image)
            results = get_image_embeddings(
                    model=clip_model,
                    pixel_values=clip_image.to(accelerator.device),
                    output_attentions=False,  
                    output_hidden_states=False  
                )
            clip_encoder_hidden_states = results['last_hidden_state']
            projection = torch.nn.Linear(1024, 512).to(accelerator.device)
            clip_encoder_hidden_states = projection(clip_encoder_hidden_states) 
            ram_encoder_hidden_states = clip_encoder_hidden_states
            validation_prompt= example["input_ids"] 
            
            negative_prompt="" 
            ori_width, ori_height = validation_image.size
            resize_flag = False
            rscale = args.upscale

            bg_image_name = example["bg_img"] 
            validation_image = Image.open(bg_image_name).convert("RGB")

            validation_image = validation_image.resize((512, 512))
            resize_flag = True
            width, height=512,512

            for sample_idx in range(args.sample_times):  
                with torch.autocast("cuda"):
                    image = pipeline(
                            validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator, height=height, width=width,
                            guidance_scale=args.guidance_scale, negative_prompt=negative_prompt, conditioning_scale=args.conditioning_scale,
                            start_point=args.start_point,ram_encoder_hidden_states=ram_encoder_hidden_states,
                            latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
                            args=args,
                        ).images[0] 
                if args.align_method == 'nofix':
                    image = image
                else:
                    if args.align_method == 'wavelet':
                        image = wavelet_color_fix(image, validation_image)
                    elif args.align_method == 'adain':
                        image = adain_color_fix(image, validation_image)
                if args.datasets == 'tac':                       
                    name, ext = os.path.splitext(os.path.basename(image_name))
                    image.save(f'{output_dir}/{image_item}/{name}.png')
                elif args.datasets == 'ssvtp': 
                    image.save(save_path)
                elif args.datasets == 'hct': 
                    image.save(save_path)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="") 
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k")
    parser.add_argument("--negative_prompt", type=str, default="dotted, noise, blur, lowres, smooth")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="bf16") 
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) 
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='nofix')
    parser.add_argument("--start_steps", type=int, default=999) 
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr') 
    parser.add_argument("--save_prompts", action='store_true')
    parser.add_argument("--datasets", type=str, default='tac')
    parser.add_argument("--move_blur", action='store_true')
    input_args = [
        "--pretrained_model_path", "/SD-2-base",
        "--prompt", "",
        "--model_path","/output/checkpoint-10000",
        "--image_path", "/datasets/test_datasets",
        "--output_dir", "/output_test",
        "--start_point", "noise",
        "--num_inference_steps", "50",
        "--guidance_scale", "5.5",
        "--process_size", "512",
        "--seed","11",
        "--datasets", "tac",
    ]
    args = parser.parse_args(input_args)

    main(args)



