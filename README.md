
# VTTac: Visual-Text Information-Driven Tactile Data Generation 
目前代码包含针对不同数据集（如 TacQuad, SSVTP, HCT）的推理流程，并支持添加视觉退化（如运动模糊）以验证模型的鲁棒性。

## 🛠️ 1. 环境配置

本项目需要 Python 运行环境。建议使用 Anaconda 创建虚拟环境后安装所需依赖。

```bash
# 创建虚拟环境 (推荐 Python 3.10)
conda create -n vttac python=3.10
conda activate vttac

# 安装核心依赖
pip install torch==2.0.1 torchvision==0.15.2 
pip install diffusers==0.21.0 transformers==4.25.0 accelerate xformers
pip install opencv-python huggingface-hub==0.21.0 pytorch_lightning loralib fairscale pydantic==1.10.11 gradio==3.24.0
```

*注意：为了优化显存占用和加速推理，代码默认启用了 `xformers` 的内存高效注意力机制。*

## 📂 2. 预训练权重与模型准备

在运行推理代码之前，请确保您已经下载并配置了以下预训练模型权重：

1.  **Stable Diffusion V2 Base**: 需要通过 `--pretrained_model_path` 指定，例如 `/SD-2-base`。
2.  **CLIP 视觉编码器**: 默认从 `/clip_encoder/clip_vit_L_14` 加载（`CLIP_ViT-L/14`），用于提取参考图像的视觉特征。
3.  **VTTac Checkpoint**: 我们微调后的模型权重路径，需要通过 `--model_path` 指定，例如 `/output/checkpoint-10000`。

## 🚀 3. 运行推理

推理脚本 `vttac_inference.py` 内部已经配置了默认参数。您可以直接运行脚本，或者通过命令行参数覆盖默认配置。

### 基础推理指令
```bash
python vttac_inference.py \
    --pretrained_model_path "/SD-2-base" \
    --model_path "/output/checkpoint-10000" \
    --image_path "/datasets/test_datasets" \
    --output_dir "/output_test" \
    --datasets "tac" \
    --num_inference_steps 50 \
    --guidance_scale 5.5 \
    --seed 11
```

### 视觉退化（鲁棒性测试）推理
如果您需要复现针对 SSVTP 数据集的视觉退化（运动模糊）测试，请添加 `--move_blur` 参数。代码会自动应用核大小为 30、角度为 45 度的运动模糊处理：
```bash
python vttac_inference.py \
    --datasets "ssvtp" \
    --move_blur \
    --align_method "nofix"
```

## ⚙️ 4. 核心参数说明

以下是 `vttac_inference.py` 中关键参数的解释：

* `--pretrained_model_path`: 基础 Stable Diffusion 模型的路径（包含 unet, vae, scheduler, text_encoder 等）。
* `--model_path`: VTTac 训练后保存的权重检查点路径。
* `--datasets`: 指定评测数据集的名称，代码会根据该参数决定输出目录结构。可选值：
    * `tac`: 输出至 `/output_tacquad/output`
    * `ssvtp`: 输出至 `/output_ssvtp/output`
    * `hct`: 输出至 `/output_hct/output/`
* `--move_blur`: 布尔标志位。激活后会对输入的视觉图像应用运动模糊，用于测试模型在视觉信息受损时的鲁棒性。
* `--align_method`: 颜色对齐/修复方法。可选值为 `['wavelet', 'adain', 'nofix']`，默认值为 `nofix`。
* `--start_point`: 推理起点，可选 `lr` 或 `noise`。
* `--num_inference_steps`: 扩散模型的采样步数，默认 `50`。
* `--guidance_scale`: 无分类器引导 (CFG) 权重，默认 `5.5`。
* `--process_size`: 图像处理和生成的分辨率，默认 `512`。

## 📁 5. 输出目录结构

生成的触觉图片将根据所选的 `--datasets` 参数保存在不同的目录中：
* **TacQuad 数据集 (`--datasets tac`)**: 保存至 `/output_tac/output/
* **SSVTP 数据集 (`--datasets ssvtp`)**: 保存至 `/output_ssvtp/output/
* **HCT 数据集 (`--datasets hct`)**: 保存至 `/output_hct/output/
