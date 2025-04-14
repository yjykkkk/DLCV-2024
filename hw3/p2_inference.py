import torch
import torch.nn as nn
import timm
from tokenizer import BPETokenizer
from decoder import Decoder, Config
import sys
import os
import json
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import loralib as lora 
from tqdm import tqdm
import torch.nn.functional as F
import time
import clip
import numpy as np
import random

batch_cnt = 1
batch_size = 1
amp_dtype = torch.float16
amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

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

vit_name = "vit_gigantic_patch14_clip_224.laion2b"
class CustomModel(nn.Module):
    def __init__(self, decoder_path, encoder_name=vit_name):
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
        image_feat = self.encoder.forward_features(image) #([bs, 257, 1024])
        image_emb = self.encoder_proj(image_feat) #([bs, 257, 768])
        text_pred = self.decoder(text_id, image_emb) #text_id: #([bs, 128]) 
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

class TestDataset(Dataset):
    def __init__(self, data_path, transform=image_transforms):
        super().__init__()
        self.data_path = data_path
        self.filename = os.listdir(self.data_path)
        self.transform = transform

    def __len__(self):
        return len(self.filename)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_path, self.filename[idx])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def create_test_dataloader(data_path, batch_size=1, num_workers=4):
    dataset = TestDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    return loader, dataset.filename

if __name__ == "__main__":
    start_time = time.time()
    decoder_path = sys.argv[3] # "./hw3_data/p2_data/decoder_model.bin"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CustomModel(decoder_path=decoder_path)
    model.encoder.eval()
    model = model.to(device)
    checkpoint = torch.load("./p2_best.pt")
    model.load_state_dict(checkpoint, strict=False)  #9
    model.eval()

    data_path = sys.argv[1]  # "hw3_data/p2_data/images/val"
    loader, filename = create_test_dataloader(data_path)

    filecnt = 0
    result = {}
    
    tokenizer = BPETokenizer('encoder.json', 'vocab.bpe')
    with torch.no_grad():
        for img in tqdm(loader): #img: [bs, 3, 224, 224]
            batch_cnt = img.size(0)
            img = img.to(device)
            out = model.generate2(img)

            for i in range(batch_cnt):
                name = filename[filecnt].split('.')[0]
                #print(name)
                out_sub = out[i]
                out_sub = out_sub[1:]
                try:
                    out_endidx = out_sub.index(50256)
                except:
                    out_endidx = len(out_sub)
                out_sub = out_sub[:out_endidx]
                predict = tokenizer.decode(out_sub)
                #print(predict)
                
                filecnt += 1
                result[name] = predict
    print(len(result))
    output_path = sys.argv[2] 
    output_dir = os.path.dirname(output_path)  
    if output_dir: 
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir) 
    with open(output_path, 'w') as f:
        json.dump(result, f, indent = 4)
    end_time = time.time()
    print("exec time: ", end_time-start_time)
    