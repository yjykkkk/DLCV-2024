import numpy as np
#import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm.auto import tqdm
import random
import torchvision.models as models
from argparse import ArgumentParser
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    #transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomGrayscale(0.3),   
    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=(-0.5, 0.5)),
    #transforms.GaussianBlur((3, 3), (1.0, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class customDataset(Dataset):
    def __init__(self,path,tfm=test_tfm,files = None):
        super(customDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
            
        return im,label

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.resnet = models.resnet50(weights=None)
        
        in_features = self.resnet.fc.out_features  # 原始 fc 的输出特征维度 #1000
        print('in_features ', in_features)
        self.new_classifier = nn.Sequential(    # 新分类器    
            nn.Linear(in_features, 512),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    '''def forward(self, x):
        features = self.resnet(x) # 通过 ResNet50 提取特征
        x = self.new_classifier(features) # 通过新的分类器进行输出
        return x, features'''
    def forward(self, x):
        x_resnet = self.resnet(x)  # ResNet 特征提取
        x_before_last = self.new_classifier[:-1](x_resnet)  # 提取第二倒数层的输出（不包括最后一层）
        x = self.new_classifier(x_resnet)  # 完整地通过分类器进行分类
        return x, x_before_last, x_resnet # 返回第二倒数层的输出和最终输出

if __name__ == '__main__':
    batch_size = 128
    train_set = customDataset("./hw1_data/p1_data/office/train", tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    outputs_list = []
    features_list = []
    backbone_list = []
    labels_list = []
    model = CustomModel(num_classes=65)
    #ckpt_path = './p1_C_last_249.ckpt'
    ckpt_path = './p1_C_epoch_0.ckpt'
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    #model.eval()  
    model.train()
    with torch.no_grad(): 
        for images, labels in train_loader:  # 迭代训练数据
            images = images.to(device)
            labels = labels.to(device)
            outputs, features, backbone = model(images)
            outputs_list.append(outputs.cpu().numpy())  
            features_list.append(features.cpu().numpy()) 
            backbone_list.append(backbone.cpu().numpy())
            labels_list.append(labels.cpu().numpy())  

    # 将所有特征和标签连接成数组
    features_array = np.concatenate(features_list, axis=0)
    outputs_array = np.concatenate(outputs_list, axis=0)
    backbone_array = np.concatenate(backbone_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    # 运行 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features_array)

    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels_array, cmap='jet', s=5)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of the Train Set")
    plt.savefig("s_feature_first.jpg")
    #plt.savefig("tsne_0.jpg")
    plt.close()

    
    # 运行 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    outputs_tsne = tsne.fit_transform(outputs_array)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(outputs_tsne[:, 0], outputs_tsne[:, 1], c=labels_array, cmap='jet', s=5)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of the Train Set")
    plt.savefig("s_output_first.jpg")
    plt.close()
    
    # 运行 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    backbone_tsne = tsne.fit_transform(backbone_array)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(backbone_tsne[:, 0], backbone_tsne[:, 1], c=labels_array, cmap='jet', s=5)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of the Train Set")
    plt.savefig("s_backbone_first.jpg")
    plt.close()