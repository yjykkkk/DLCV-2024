from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.transforms as trns
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
from UNet import UNet
from PIL import Image, ImageDraw, ImageFont
import random
import time

start_time = time.time()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(18) #87


def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    return betas

class DDIM:
    def __init__(self, checkpoint_path, unet=UNet, beta_scheduler=beta_scheduler(), ddim_steps=50, eta=0.0):
        self.unet = unet()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unet = self.unet.to(device)
        checkpoint = torch.load(checkpoint_path)
        self.unet.load_state_dict(checkpoint)
        self.betas = beta_scheduler
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.ddim_steps = ddim_steps
        self.eta = eta  # Controlling the randomness in DDIM steps

    @torch.no_grad()
    def sample(self, x_t, time_steps):
        device = x_t.device  
        #time_steps = torch.tensor(time_steps, device=device)
        
        # Loop over timesteps, starting from t_T to t_0
        time_steps = time_steps.to(device)
        for i, step in enumerate(time_steps[1:], start=1):
            #step = time_steps[i]
            if i < 49:
                t_prev = time_steps[i+1]
            else:
                t_prev = 0
            
            t = time_steps[i]
            
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t_prev]
            
            beta_t = self.betas[t]
            
            #self.unet.eval()

            # Predict the noise at this step
            noise_pred = self.unet(x_t, t)
            
            # Calculate x_0 based on the model prediction
            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, min=-1.0, max=1.0) #clip???

            # compute sigma: σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigma_t = self.eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t/alpha_prev)

            # Calculate direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_prev - sigma_t**2) * noise_pred 

            # compute x_{t-1}
            x_t = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma_t * torch.randn_like(x_t)
        return x_t


def compare_mse():
    img_dir = "p2_outputs3"
    GT_dir = "hw2_data/face/GT/"
    img = [os.path.join(img_dir, f"{i:02d}.png") for i in range(10)]
    GT = [os.path.join(GT_dir, f"{i:02d}.png") for i in range(10)]
    #transform = transforms.Compose([transforms.ToTensor(),])
    mse_total = 0.0
    for i, (generated_path, ground_truth_path) in enumerate(zip(img, GT)):
        img = Image.open(generated_path)
        GT = Image.open(ground_truth_path)
        img =  np.array(img)
        GT = np.array(GT)
        mse = np.mean((img - GT)**2)
        print(f"MSE for image pair {i}: {mse:5f}")
        mse_total += mse
    print(f"MSE total: {mse_total:5f}")


def compare_mse_tensor():
    img_dir = "p2_outputs3"
    GT_dir = "hw2_data/face/GT/"
    img = [os.path.join(img_dir, f"{i:02d}.png") for i in range(10)]
    GT = [os.path.join(GT_dir, f"{i:02d}.png") for i in range(10)]
    transform = transforms.Compose([transforms.ToTensor(),])
    mse_total = 0.0
    #compare_img_list = torch.empty(0, dtype=torch.float32)
    for i, (generated_path, ground_truth_path) in enumerate(zip(img, GT)):
        img = transform(Image.open(generated_path))
        GT = transform(Image.open(ground_truth_path))

        img =  np.array(img)
        GT = np.array(GT)
        img = img * 255
        GT = GT * 255

        img_tensor = torch.from_numpy(img).float()  # Convert to float32
        GT_tensor = torch.from_numpy(GT).float()
        mse = torch.nn.functional.mse_loss(img_tensor, GT_tensor)
        print(f"tensor MSE for image pair {i}: {mse.item():5f}")
        mse_total += mse
    print(f"tensor MSE total: {mse_total}")


if __name__ == "__main__":
    ddim_steps = 50
    eta = 0.0
    num_samples = 10
    betas = beta_scheduler()
    total_timesteps = len(betas)
    time_steps = torch.linspace(1, total_timesteps + 1, ddim_steps + 1).long().flip(0)
    save_dir = sys.argv[2]
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_path = sys.argv[1]  #"hw2_data/face/noise"
    filenames = [f"{i:02d}.pt" for i in range(0, num_samples)]
    tensors = [torch.load(os.path.join(noise_path, filename)) for filename in filenames]
    
    checkpoint_path = sys.argv[3]  #"hw2_data/face/UNet.pt"
    ddim = DDIM(checkpoint_path=checkpoint_path)

    for i in range(10):
        x_t = tensors[i].to(device)
        with torch.no_grad():
            x_gen = ddim.sample(x_t, time_steps)
            
            img = x_gen
            min_val = torch.min(img)
            max_val = torch.max(img)
            normalized_x_gen = (img - min_val) / (max_val - min_val)
            save_image(normalized_x_gen, os.path.join(save_dir, f"{i:02d}.png"))
            
            #save_image(x_gen, os.path.join(save_dir, f"{i:02d}.png"), normalize=True)
    
    #compare_mse()
    #compare_mse_tensor()
    end_time = time.time()
    print("time:", end_time-start_time)