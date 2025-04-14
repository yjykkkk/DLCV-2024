import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os
from tqdm import tqdm 
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataset import Dataset
import torchvision.models as models
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision
import torch.nn.functional as F
import imageio
import scipy.misc
import argparse
#import skimage.io
from mean_iou_evaluate import read_masks, mean_iou_score
import random
from argparse import ArgumentParser

batch_size = 8
n_epochs = 200

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

def basic_augmentation(image, mask):
    if random.random() > 0.5: #翻轉
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)
    if random.random() > 0.5:
        image = transforms.functional.vflip(image)
        mask = transforms.functional.vflip(mask)
    if random.random() > 0.5: #旋轉
        angle = random.uniform(-30, 30)  # -30 到 30 度之間的角度
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
    return image, mask

def load_checkpoint(model, optimizer, checkpoint_path):
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 下一個epoch開始
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    else:
        print("Checkpoint not found. Starting from scratch.")
        start_epoch = 0
    return model, optimizer, start_epoch
    
class satDataset(Dataset):
    def __init__(self, transformer, path, augmentation=None):
        self.transformer = transformer
        self.path = path
        self.augmentation = augmentation
        self.train_id_list = sorted(list(set([i.split("_")[0] for i in os.listdir(self.path)])))
        self.fname = []
        self.labels = read_masks(self.path)
        for file_prefix in self.train_id_list:
            self.fname.append(file_prefix)

    def __getitem__(self, index):
        image = Image.open(self.path+self.fname[index]+"_sat.jpg")
        mask = self.labels[index]
        if self.augmentation:
            mask = Image.fromarray(mask) # 把 numpy 轉成 PIL.Image
            image, mask = self.augmentation(image, mask)
            #image = np.array(image)
            mask = np.array(mask)
        image = self.transformer(image)
        return image, mask, self.fname[index]

    def __len__(self):
        return len(self.train_id_list)

class VGG16FCN32s(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG16FCN32s, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = self.vgg16.features  # output size [batch_size, 512, 16, 16]
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=3),  # [batch_size, 4096, 16, 16]
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),  # [batch_size, 4096, 16, 16]
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, kernel_size=1, stride=1),  # [batch_size, num_classes, 16, 16]
        )
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)

    def forward(self, x):
        x = self.features(x)  # VGG16 feature extraction
        x = self.classifier(x)  #  [batch_size, num_classes, 16, 16]
        x = self.upsample(x)  #  [batch_size, num_classes, 512, 512]
        return x


if __name__=='__main__':
    parser = ArgumentParser('HW1 p2')
    parser.add_argument('--data_path', 
                        default='./hw1_data/p2_data', type=str, help='Please enter the path of the pretrained checkpoiny.')
    args = parser.parse_args()
    data_path = args.data_path

    img_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.408507245, 0.37851192, 0.280892825], std=[0.14260272161078436, 0.10846701523901407, 0.09821331597487479])
    ])
    train_dataset = satDataset(transformer=img_transformer, path=os.path.join(data_path, 'train/'), augmentation=basic_augmentation)
    val_dataset = satDataset(transformer=img_transformer, path=os.path.join(data_path, 'validation/'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    criterion = nn.CrossEntropyLoss()
    model = VGG16FCN32s()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    # ----train-----
    best_miou = 0
    best_val_miou = 0

    checkpoint_path = None
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        train_loss = 0.0
        
        for i, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad() 
            images = data[0]
            labels = data[1]
            labels = labels.long()
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
          
        avg_train_loss = train_loss / len(train_loader)
        print("epoch: ", epoch, f"Training Loss: {avg_train_loss:.4f}")
        
        # -----validation-----
        model.eval() 
        val_loss = 0.0
        num_batches = 0
        labels_list = []
        pred_list = []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                labels = labels.long()
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                labels_list.append(labels.detach().cpu().numpy().astype(np.int64))
                pred_list.append(predicted.detach().cpu().numpy().astype(np.int64))
       
        avg_val_loss = val_loss / len(val_loader)
        val_miou = mean_iou_score(np.concatenate(pred_list, axis=0), np.concatenate(labels_list, axis=0))
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation mIOU: {val_miou:.4f}")

        if best_val_miou < val_miou:
            best_val_miou = val_miou
            print("best miou: ", val_miou)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            checkpoint_path = 'p2_A_best.pth'
            torch.save(checkpoint, checkpoint_path)
    print(f"Best Validation mIOU: {best_val_miou:.4f}")