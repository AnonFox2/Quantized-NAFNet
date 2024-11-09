# Quantized NAFNet for Efficient Motion Deblurring on Edge Devices

This repository contains an optimized version of the NAFNet model for real-time motion deblurring on resource-constrained edge devices. The model has been quantized and pruned to improve inference speed and reduce memory usage, while maintaining high-quality results, making it ideal for applications like robotics.

## Project Overview

NAFNet is a neural network architecture designed for image restoration tasks, including motion deblurring. However, the original model's size and computational requirements make it challenging to deploy on edge devices with limited resources. This project includes:
- A smaller, pruned version of NAFNet with quantization-aware training
- Scripts for training, validating, and testing the model
- Instructions to set up the environment and run the model on sample images

## Features

- **Pruned Model**: A reduced-size model with approximately 75% fewer parameters and 16x faster inference speed
- **8-bit Quantization**: Maintains ~95% of the original accuracy while enabling efficient deployment on hardware with 8-bit computation

## Environment Setup

To set up the environment, clone this repository and create a CONDA virtual environment:

```bash
git clone https://github.com/AnonFox2/Quantized-NAFNet.git
cd Quantized-NAFNet

conda create -n quant python=3.9.19
conda activate quant

pip install -r requirements.txt
```

## Model Validation

### Validation Dataset Preparation (optional, committed in existing repo)
1. Download validation data from [Google Drive](https://drive.google.com/file/d/1_WPxX6mDSzdyigvie_OlpI-Dknz7RHKh/view)
2. Extract and move the `val` folder into `./datasets/REDS/`

Required folder structure:
```
Quantized-NAFNet-0607/
└── datasets/
    └── REDS/
        └── val/
            ├── blur_300.lmdb/
            └── sharp_300.lmdb/
```

### Running Validation

```bash
# Original Model
python main.py --dataset_root datasets/ --mode val --weight fpw64_original_model.pth

# Pruned Model
python main.py --dataset_root datasets/ --mode val --weight fpw32_small_full_precision.pth

# Pruned + Quantized Model
python main.py --dataset_root datasets/ --mode val --weight intw32_quantized_model.pth
```

## Testing on Custom Images

```bash
# Test different models on various images
python test.py --input_img blurry.jpg --output_img deblur.png --weights fpw64_original_model.pth
python test.py --input_img blurry.jpg --output_img deblur.png --weights fpw32_small_full_precision.pth
python test.py --input_img blurry.jpg --output_img deblur.png --weights intw32_quantized_model.pth

python test.py  --input_img 1.jpg --output_img deblur.png --weights fpw64_original_model.pth
python test.py  --input_img 2.jpg --output_img deblur.png --weights fpw32_small_full_precision.pth
python test.py  --input_img 3.jpg --output_img deblur.png --weights intw32_quantized_model.pth
```

## Model Training

### Training Dataset Preparation

1. Download REDS dataset (requires ~150GB free space):
   - [Blurry images](https://drive.google.com/file/d/1VTXyhwrTgcaUWklG-6Dh4MyCmYvX39mW/view)
   - [Sharp images](https://drive.google.com/file/d/1YLksKtMhd2mWyVSkvhDaDLWSc1qYNCz-/view)
   - Source: [REDS Dataset](https://seungjunnah.github.io/Datasets/reds)

2. Extract the downloaded datasets into `./datasets/REDS/`

Required folder structure:
```
Quantized-NAFNet-0607/
└── datasets/
    └── REDS/
        ├── val/
        │   ├── blur_300.lmdb/
        │   └── sharp_300.lmdb/
        └── train/
            ├── train_blur_jpeg/
            └── train_sharp/
```

3. Prepare the dataset:
```bash
python prep_dataset.py
```

### Training Models

```bash
# Original Model (requires >24GB VRAM)
python main.py --dataset_root datasets/ --mode train --weight fpw64_original_model.pth

# Pruned Model (requires ~24GB VRAM)
python main.py --dataset_root datasets/ --mode train --weight fpw32_small_full_precision.pth

# Pruned + Quantized Model (requires ~24GB VRAM)
python main.py --dataset_root datasets/ --mode train --weight intw32_quantized_model.pth
```
