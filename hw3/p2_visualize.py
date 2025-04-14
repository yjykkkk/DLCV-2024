import torch
import torch.nn as nn
import timm
from tokenizer import BPETokenizer 
from decoder_v import Decoder, Config
import os
import json
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import loralib as lora 
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
import time
import clip
import numpy as np
import random
import math
batch_cnt = 1
batch_size = 1
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(87)

amp_dtype = torch.float16
amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

vit_name = "vit_gigantic_patch14_clip_224.laion2b"
class CustomModel(nn.Module):
    def __init__(self,  encoder_name=vit_name, decoder_path="./hw3_data/p2_data/decoder_model.bin"):
        super(CustomModel, self).__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=True)
        cfg = Config(checkpoint=decoder_path)
        self.decoder = Decoder(cfg)
        self.encoder_proj = nn.Sequential(
            nn.Linear(1664, 1200),
            nn.LayerNorm(1200),
            nn.GELU(),
            nn.Dropout(0.1), 
            nn.Linear(1200, cfg.n_embd),
            nn.LayerNorm(cfg.n_embd)
        )

    def forward(self, image, text_id):
        image_feat = self.encoder.forward_features(image) #([batch_size, 257, 1024])
        image_emb = self.encoder_proj(image_feat) #([batch_size, 257, 768])
        text_pred = self.decoder(text_id, image_emb) #text_id: #([batch_size, 128]) 
        return text_pred
    def generate2(self, images, max_length=70):
        image_features = self.encoder.forward_features(images)
        image_emb = self.encoder_proj(image_features) 
        batch_size = images.shape[0]
        text_ids = torch.full((batch_size, max_length), 50256, dtype=torch.long, device=images.device)
        text_ids[:, 0] = 50256  # 初始 token
        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=True):
            for step in range(1, max_length):
                yhat = self.decoder(text_ids, image_emb)
                next_tokens = torch.argmax(yhat[:, step - 1, :], dim=-1)
                text_ids[:, step] = next_tokens  
                if all(next_tokens == 50256):  
                    break
        return text_ids.tolist()
    def generate_with_attention(self, images, max_length=70, layer_idx=11): #11
        image_features = self.encoder.forward_features(images)
        image_emb = self.encoder_proj(image_features)  # [B, 257, D]
        batch_size = images.shape[0]
        text_ids = torch.full((batch_size, max_length), 50256, dtype=torch.long, device=images.device)
        text_ids[:, 0] = 50256  # Initialize with [START] token

        all_text_to_image_attentions = []  # 保存文本到图像 patch 的注意力

        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=True):
            for step in range(1, max_length):
                yhat, attentions = self.decoder(text_ids[:, :step], image_emb, return_attention=True, layer_idx=layer_idx)
                #print("attention shape", attentions.shape)
                text_to_image = attentions[:, :,  -1, 1:257]  #257+step-1
                all_text_to_image_attentions.append(text_to_image.cpu().detach())
                next_tokens = torch.argmax(yhat[:, step - 1, :], dim=-1)
                text_ids[:, step] = next_tokens
                if all(next_tokens == 50256):  # Stop if [EOS] token is generated
                    break
        return text_ids.tolist(), all_text_to_image_attentions

def project_attention_to_image(attention_map, image_size, patch_size=14):
    height, width = image_size #224, 224
    
    #print("attention map min", attention_map.min(), "max", attention_map.max())
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    patches_per_row = 16  # Assuming the grid is 16x16
    attention_map = attention_map.reshape(patches_per_row, patches_per_row)  # 16x16 
    attention_map_tensor = torch.tensor(attention_map).float().unsqueeze(0).unsqueeze(0)  
    attention_map_tensor = torch.nn.functional.interpolate(
        attention_map_tensor, size=(height, width), mode='bilinear', align_corners=False 
    )
    attention_map_resized = attention_map_tensor.squeeze().numpy()  
    return attention_map_resized

def visualize_attention_map_in_row(image, attention_weights, text_id, token_len, save_path, top_k=10, patch_size=14):
    image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert torch tensor to numpy array
    print("attention_weights len", len(attention_weights))  # Print the length of attention_weights    

    cols = 5 
    rows = (token_len+1 + cols - 1) // cols 
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() 

    ax = axes[0]
    ax.imshow(image_np)  
    ax.set_title("<start>")
    ax.axis("off")
    # Loop through each token's attention map
    for idx in range(0, token_len):  #here 1
        text = tokenizer.decode([text_id[idx]])
        attention_map = attention_weights[idx]  # Get attention map for the current token
        attn = attention_map.squeeze(0)  # [head_num, patch_num]
        avg_attention = attn.mean(dim=0).cpu().detach().numpy()  # 256
        attention_map_resized = project_attention_to_image(avg_attention, (image_np.shape[0], image_np.shape[1]), patch_size)

        ax = axes[idx+1]
        ax.imshow(image_np)  
        ax.imshow(attention_map_resized, cmap='jet', alpha=0.6)  
        ax.set_title(f"{text}")
        ax.axis("off")

    for j in range(token_len+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_mean.png", bbox_inches='tight')
    plt.close(fig) 

def visualize_attention_map_by_head_in_row(image, attention_weights, text_id, token_len, save_path, patch_size=14):
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert torch tensor to numpy array
    else:
        image_np = image
    n_heads = attention_weights[1].squeeze(0).shape[0]  # 假设所有 tokens 的 head 数一致
    
    for head_idx in range(n_heads):
        cols = 5  
        rows = (token_len+1 + cols - 1) // cols 
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() 


        ax = axes[0]  
        ax.imshow(image_np) 
        ax.set_title(f"<start>")
        ax.axis("off")

        for idx in range(0, token_len):
            text = tokenizer.decode([text_id[idx]])
            attention_map = attention_weights[idx]  # Get attention map for the current token
            attn = attention_map.squeeze(0)  # [n_heads, patch_num]
            single_head_attention = attn[head_idx].cpu().detach().numpy()  
            
            attention_map_resized = project_attention_to_image(single_head_attention, (image_np.shape[0], image_np.shape[1]), patch_size)
            
            ax = axes[idx+1]  
            ax.imshow(image_np)
            ax.imshow(attention_map_resized, cmap='jet', alpha=0.6) 
            ax.set_title(f"{text}")
            ax.axis("off")
    
        for j in range(token_len+1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(f"{save_path}_head_{head_idx}.png", bbox_inches='tight')
        plt.close(fig) 


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CustomModel()
    model.encoder.eval()
    model = model.to(device)
    checkpoint = torch.load("./p2_best.pt")
    model.load_state_dict(checkpoint, strict=False) 
    model.eval()

    tokenizer = BPETokenizer('encoder.json', 'vocab.bpe')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  #here normalize
    ])
    image_folder = "./hw3_data/p3_data/images"
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    for image_file in image_files:
        image_name = image_file.split('.')[0]
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        save_path = image_name
        image_tensor = transform(image).unsqueeze(0).to(device)
        generated_text_ids, attention_map = model.generate_with_attention(image_tensor)
        text_sub = generated_text_ids[0][1:]
        try:
            out_endidx = text_sub.index(50256)+1
        except:
            out_endidx = len(text_sub)
        token_len = out_endidx
        out_sub = text_sub[:out_endidx]
        predict = tokenizer.decode(out_sub)
        transform2 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image_tensor2 = transform2(image).to(device)
        visualize_attention_map_in_row(image_tensor2, attention_map, out_sub, token_len, save_path)
        visualize_attention_map_by_head_in_row(image_tensor2, attention_map, out_sub, token_len, save_path)
