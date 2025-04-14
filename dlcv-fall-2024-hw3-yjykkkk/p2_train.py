import torch
import torch.nn as nn
import timm
from tokenizer import BPETokenizer
import os
import json
import torch 
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import loralib as lora 
from tqdm import tqdm
from decoder import Decoder, Config
from torch.nn.utils.rnn import pad_sequence
import clip
from torch.cuda.amp import autocast, GradScaler

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
        text_pred = self.decoder(text_id, image_emb) #text_id: #([batch_size, 128]) # 
        return text_pred

class CustomDataset(Dataset):
    def __init__(self, data_dir, json_file, transform=image_transforms):
        self.data_dir = data_dir
        self.transform = transform
        with open(json_file) as f:
            json_data = json.load(f)
        id_path = {str(item['id']): item['file_name'] for item in json_data['images']}
        data = [
            [item['caption'] for item in json_data['annotations']],
            [id_path[str(item['image_id'])] for item in json_data['annotations']],
        ]
        self.df = pd.DataFrame({'caption': data[0], 'image_path': data[1]})
        self.tokenizer = BPETokenizer('encoder.json', 'vocab.bpe')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.df['image_path'][idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = self.df['caption'][idx]
        caption_id = self.tokenizer.encode(caption)
        caption_input = [50256] + caption_id 
        caption_gt = caption_id + [50256]
        return image, torch.tensor(caption_input), torch.tensor(caption_gt)

def custom_collate_fn(batch):
    images, captions_input, captions_gt = zip(*batch)
    images = torch.stack(images, dim=0)
    captions_input = pad_sequence(captions_input, batch_first=True, padding_value=50256)
    captions_gt = pad_sequence(captions_gt, batch_first=True, padding_value=-100)
    return images, captions_input, captions_gt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = CustomModel()
model = model.to(device)
data_dir = 'hw3_data/p2_data/images/train'
json_file = os.path.join("hw3_data", "p2_data", 'train.json')
dataset = CustomDataset(data_dir=data_dir, json_file=json_file, transform=image_transforms)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)

val_data_dir = 'hw3_data/p2_data/images/val'
val_json_file = os.path.join("hw3_data", "p2_data", 'val.json')
val_dataset = CustomDataset(data_dir=val_data_dir, json_file=val_json_file, transform=image_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3) 

lora.mark_only_lora_as_trainable(model)
for name, param in model.named_parameters():
    if 'encoder_proj' in name:
        param.requires_grad = True
num_para_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total Paramters:', num_para_grad)

model.train()
#model.clip_model.eval()
model.encoder.eval()
total_loss = 0
num_train_epochs = 100
save_dir = "p2_model"
os.makedirs(save_dir, exist_ok=True)
scaler = GradScaler(enabled=True)

for epoch in range(num_train_epochs):
    model.train() 
    epoch_loss = 0.0
    num_batches = 0
    for idx, (img, txt, gt) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad() 
        img, txt, gt = img.to(device), txt.to(device), gt.to(device)
        
        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=True):
            yhat = model(img, txt)
            loss = criterion(yhat.view(-1, 50257), gt.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        num_batches += 1
        

    average_loss = epoch_loss / num_batches
    print(f"Epoch [{epoch+1}/{num_train_epochs}], Loss: {average_loss:.4f}")
    
    
    model.eval() 
    val_loss = 0.0
    num_val_batches = 0
    with torch.no_grad():
        for idx, (img, txt, gt) in enumerate(tqdm(val_dataloader)):
            img, txt, gt = img.to(device), txt.to(device), gt.to(device)
            
            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=True):
                yhat = model(img, txt)
                loss = criterion(yhat.view(-1, 50257), gt.view(-1))
            
            val_loss += loss.item()
            num_val_batches += 1
    average_val_loss = val_loss / num_val_batches
    print(f"Validation Loss: {average_val_loss:.4f}")
    
    trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad]
    save_weights = {k: v for k, v in model.state_dict().items() if k in trainable_weights}
    torch.save(save_weights, os.path.join(save_dir, f"ep_{epoch}.pt"))