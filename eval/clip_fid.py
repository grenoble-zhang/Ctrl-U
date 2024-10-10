import torch
import argparse

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.multimodal.clip_score import CLIPScore
from cleanfid import fid
import os
import time
import random
import torch
torch.cuda.set_device(0)
import numpy as np
import torchvision.transforms.functional as F
import cv2
import pdb
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from diffusers import (
    T2IAdapter, StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline, ControlNetModel,
    UniPCMultistepScheduler, DDIMScheduler,
    StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetPipeline, AutoencoderKL
)
from datasets import load_dataset, load_from_disk
from accelerate import PartialState
from PIL import Image
from kornia.filters import canny
from transformers import DPTImageProcessor, DPTForDepthEstimation

from torchvision.transforms import Compose, Normalize, ToTensor
transforms = Compose([
    ToTensor(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import rectified_flow

from PIL import PngImagePlugin
MaximumDecompressedsize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedsize * MegaByte
Image.MAX_IMAGE_PIXELS = None
import pdb


scores = {}

def fid_score(real_image_path, generated_image_path, only_image):
    score = 0.0
    # We have 4 groups of generated images
    if not only_image:
        for i in range(4):
            score += fid.compute_fid(
                real_image_path,
                f'{generated_image_path}/group_{i}',
                dataset_res=512,
                batch_size=128
            )
    else:
        for i in range(4):
            score += fid.compute_fid(
                real_image_path,
                f'{generated_image_path}/only_image_group_{i}',
                dataset_res=512,
                batch_size=128
            )
    # Report the average FID score
    average_fid = score / 4
    scores['FID'] = average_fid

def clip(real_image_path, generated_image_path):
    dataset = load_dataset("parquet", data_files=f"{real_image_path}/validation*.parquet")["train"]
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").cuda()
    bar = tqdm(range(len(dataset)), desc=f"Evaluating {real_image_path}")
    rewards = []
    for idx in range(len(dataset)):
        data = dataset[idx]
        prompt = data["prompt"]

        image_paths = [f'{generated_image_path}/group_{i}/{idx}.png' for i in range(4)]
        images = [Image.open(x).convert('RGB') for x in image_paths]
        images = [pil_to_tensor(x).cuda() for x in images]
        metric.update(torch.stack(images), [prompt]*4)
        bar.update(1)
    clip_score = metric.score / metric.n_samples
    scores['CLIP'] = clip_score
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLIP-FID")
    parser.add_argument('--generated', type=str, default="work_dirs/eval_dirs/data/validation/Ade20k/controlnet-ade20k_reward-model-FCN-R101-d8-origin_3.0-20")
    parser.add_argument('--clip_dataset', type=str, default="Captioned_ADE20K/data")
    parser.add_argument('--fid_dataset', type=str, default='work_dirs/Captioned_ADE20K_val')
    parser.add_argument('--only_image', action="store_true")
    args = parser.parse_args()

    generated_dir = os.path.join(args.generated, 'images')
    fid_score(args.fid_dataset, generated_dir, args.only_image)
    
    
    clip(args.clip_dataset, generated_dir)
    formatted_scores = ', '.join([f"{key}:{value}" for key, value in scores.items()])
    print(formatted_scores)