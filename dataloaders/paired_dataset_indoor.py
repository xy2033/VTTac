import glob
import os
from PIL import Image
import random
import numpy as np

from torch import nn
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F
import csv
from natsort import natsorted
from .realesrgan import RealESRGAN_degradation
from transformers import CLIPProcessor
#xy
tacquad_indoor_dir = ''

tacquad_indoor_file = ''

tacquad_outdoor_dir = ''

tacquad_outdoor_file = ''
processor = CLIPProcessor.from_pretrained('/clip_encoder/clip_vit_L_14')

# 图像处理方法
def load_and_preprocess_image(image):
    """加载并预处理图像"""
    #image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"]


class PairedCaptionDataset(data.Dataset):

    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=1,#0.5
            # use_ram_encoder=False,
            # use_gt_caption=False,
            # caption_type = 'gt_caption',
    ):
        super(PairedCaptionDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        self.tag_path_list = []
        self.sen_path_list = []
        self.bg_path_list = []

        with open(tacquad_indoor_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_name = row[0]
                phrase = str(row[7])
                sentence = row[8]
                train = row[9]

                lr_path = tacquad_indoor_dir + item_name +'/img_gelsight/'
                for i in eval(train):
                    lr_list_indoor=lr_path + str(i)+'.png'
                    self.lr_list.append(lr_list_indoor)

                gt_path = tacquad_indoor_dir + item_name +'/gelsight/'
                for i in eval(train):
                    gt_list_indoor=gt_path + str(i)+'.png'
                    self.gt_list.append(gt_list_indoor)

                self.tag_path_list+=len(eval(train))*[phrase]
                self.sen_path_list+=len(eval(train))*[sentence]
                bg_path=lr_path+"0.png"
                self.bg_path_list+=len(eval(train))*[bg_path]

        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
            transforms.Resize((512, 512)), 
        ])

        self.img_preproc_clip = transforms.Compose([       
            transforms.ToTensor(),
            transforms.Resize((224, 224)), 
        ])

        ram_mean = [0.485, 0.456, 0.406]
        ram_std = [0.229, 0.224, 0.225]
        self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)

        self.tokenizer = tokenizer
    #xy
    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids


    def __getitem__(self, index):

       
        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)
        
        lq_path = self.lr_list[index]
        #clip
        clip_img = Image.open(lq_path).convert('RGB')


        #tvl
        example = dict()
        example["clip_values"]=load_and_preprocess_image(clip_img).squeeze(0)
        #tvl
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)

        if random.random() < self.null_text_ratio:
            tag = ''
            sen = ''
        else:
            tag = self.tag_path_list[index]
            sen = self.sen_path_list[index]


        bg_path = self.bg_path_list[index]
        bg_img = Image.open(bg_path).convert('RGB')
        bg_img = self.img_preproc(bg_img)
        example["conditioning_pixel_values"] = bg_img.squeeze(0)
        example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
        example["input_ids"] = self.tokenize_caption(caption=tag).squeeze(0)
        example["sentence_ids"] = self.tokenize_caption(caption=sen).squeeze(0)



        return example

    def __len__(self):
        return len(self.gt_list)


class PairedCaptionDataset2(data.Dataset):
   
    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=1,
            # use_ram_encoder=False,
            # use_gt_caption=False,
            # caption_type = 'gt_caption',
    ):
        super(PairedCaptionDataset2, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        self.tag_path_list = []
        self.sen_path_list = []
        self.bg_path_list = []

        with open(tacquad_indoor_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_name = row[0]
                phrase = str(row[7])
                sentence = row[8]
                test = row[10]

                lr_path = tacquad_indoor_dir + item_name +'/img_gelsight/'
                for i in eval(test):
                    lr_list_indoor=lr_path + str(i)+'.png'
                    self.lr_list.append(lr_list_indoor)

                gt_path = tacquad_indoor_dir + item_name +'/gelsight/'
                for i in eval(test):
                    gt_list_indoor=gt_path + str(i)+'.png'
                    self.gt_list.append(gt_list_indoor)

                self.tag_path_list+=len(eval(test))*[phrase]
                self.sen_path_list+=len(eval(test))*[sentence]

                bg_path=lr_path+"0.png"
                self.bg_path_list+=len(eval(test))*[bg_path]

        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
            transforms.Resize((512, 512)),  
        ])

        self.img_preproc_clip = transforms.Compose([       
            transforms.ToTensor(),
            transforms.Resize((512, 512)), 
        ])

        ram_mean = [0.485, 0.456, 0.406]
        ram_std = [0.229, 0.224, 0.225]
        self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)

        self.tokenizer = tokenizer
    #xy
    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

   
    def __getitem__(self, index):

       
        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)
        
        lq_path = self.lr_list[index]
        #clip
        clip_img = Image.open(lq_path).convert('RGB')
        # clip_img = self.img_preproc_clip(clip_img)
        # clip_img = self.ram_normalize(clip_img.squeeze(0))

        #tvl
        example = dict()
        example["clip_values"]=load_and_preprocess_image(clip_img).squeeze(0)
        #tvl
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)

        if random.random() < self.null_text_ratio:
            tag = ''
            sen = ''
        else:
            tag = self.tag_path_list[index]
            sen = self.sen_path_list[index]


        bg_path = self.bg_path_list[index]
        bg_img = Image.open(bg_path).convert('RGB')
        bg_img = self.img_preproc(bg_img)
        example["conditioning_pixel_values"] = bg_img.squeeze(0)
        example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
        example["input_ids"] = self.tokenize_caption(caption=tag).squeeze(0)
        example["sentence_ids"] = self.tokenize_caption(caption=sen).squeeze(0)



        return example
    #xy
    def __len__(self):
        return len(self.gt_list)
    
class PairedCaptionDataset3(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=0.5,

    ):
        super(PairedCaptionDataset3, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        self.tag_path_list = []
        self.sen_path_list = []
        self.bg_path_list = []
        self.item_name_list = []

        with open(tacquad_indoor_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_name = row[0]
                phrase = str(row[7])
                sentence = row[8]
                test = row[10]#10

                lr_path = tacquad_indoor_dir + item_name +'/img_gelsight/'
                for i in eval(test):
                    lr_list_indoor=lr_path + str(i)+'.png'
                    self.lr_list.append(lr_list_indoor)

                gt_path = tacquad_indoor_dir + item_name +'/gelsight/'
                for i in eval(test):
                    gt_list_indoor=gt_path + str(i)+'.png'
                    self.gt_list.append(gt_list_indoor)

                self.tag_path_list+=len(eval(test))*[phrase]
                self.sen_path_list+=len(eval(test))*[sentence]
                self.item_name_list+=len(eval(test))*[item_name]

                bg_path=lr_path+"0.png"
                self.bg_path_list+=len(eval(test))*[bg_path]

        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)

    
    def __getitem__(self, index):

       
        gt_path = self.gt_list[index]
        # gt_img = Image.open(gt_path).convert('RGB')
        # gt_img = self.img_preproc(gt_img)
        
        lq_path = self.lr_list[index]
        # lq_img = Image.open(lq_path).convert('RGB')
        # lq_img = self.img_preproc(lq_img)

        # if random.random() < self.null_text_ratio:
        #     tag = ''
        # else:
        tag = "" #self.tag_path_list[index]
        item_name = self.item_name_list[index]
        sen = "" #self.sen_path_list[index]


        example = dict()
        example["vision"] = lq_path
        example["touch"] = gt_path
        example["input_ids"] = tag
        example["sentence_ids"] = sen
        example["bg_img"] = self.bg_path_list[index] 

        example["item_name"] = item_name

        return example
    #xy
    def __len__(self):
        return len(self.gt_list)

class PairedCaptionDataset4(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=0.5,
            # use_ram_encoder=False,
            # use_gt_caption=False,
            # caption_type = 'gt_caption',
    ):
        super(PairedCaptionDataset4, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        self.tag_path_list = []
        self.sen_path_list = []
        self.bg_path_list = []
        self.item_name_list = []

        with open(tacquad_outdoor_file,'r',encoding='utf-8-sig') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_name = row[0]
                phrase = str(row[7])
                sentence = row[8]
                test = row[10]

                lr_path = tacquad_outdoor_dir + item_name +'/img_gelsight/'
                for i in eval(test):
                    lr_list_outdoor=lr_path + str(i)+'.png'
                    self.lr_list.append(lr_list_outdoor)

                gt_path = tacquad_outdoor_dir + item_name +'/gelsight/'
                for i in eval(test):
                    gt_list_outdoor=gt_path + str(i)+'.png'
                    self.gt_list.append(gt_list_outdoor)

                self.tag_path_list+=len(eval(test))*[phrase]
                self.sen_path_list+=len(eval(test))*[sentence]
                self.item_name_list+=len(eval(test))*[item_name]

                bg_path=lr_path+"0.png"
                self.bg_path_list+=len(eval(test))*[bg_path]

        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)
      
    
    def __getitem__(self, index):

       
        gt_path = self.gt_list[index]
        # gt_img = Image.open(gt_path).convert('RGB')
        # gt_img = self.img_preproc(gt_img)
        
        lq_path = self.lr_list[index]
        # lq_img = Image.open(lq_path).convert('RGB')
        # lq_img = self.img_preproc(lq_img)

        # if random.random() < self.null_text_ratio:
        #     tag = ''
        # else:
        tag = self.tag_path_list[index]
        item_name = self.item_name_list[index]
        sen = self.sen_path_list[index]


        example = dict()
        example["vision"] = lq_path
        example["touch"] = gt_path
        example["input_ids"] = ""
        example["sentence_ids"] = ""
        example["bg_img"] = self.bg_path_list[index]  # 背景图像路径

        # lq_img = lq_img.squeeze()

        # ram_values = F.interpolate(lq_img.unsqueeze(0), size=(384, 384), mode='bicubic')
        # ram_values = ram_values.clamp(0.0, 1.0)
        example["item_name"] = item_name

        return example
    #xy
    def __len__(self):
        return len(self.gt_list)