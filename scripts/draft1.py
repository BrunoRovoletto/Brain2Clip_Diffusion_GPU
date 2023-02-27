
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms
import requests
import pandas as pd
import matplotlib.pyplot as plt

im_path="data/example_conditioning/superresolution/sample_0.jpg",
ckpt="/workspace/Datasets/Models/sd-clip-vit-l14-img-embed_ema_only.ckpt",
config="configs/stable-diffusion/sd-image-condition-finetune.yaml",
outpath="im_variations",
scale=1.0,
h=200,
w=200,
n_samples=1,
precision="fp32",
plms=True,
ddim_steps=10,
ddim_eta=0.0,
device_idx=0,
save=True,
eval=False,

plt.imshow("data/example_conditioning/superresolution/sample_0.jpg")