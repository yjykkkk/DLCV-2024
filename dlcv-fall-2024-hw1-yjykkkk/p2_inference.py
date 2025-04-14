import numpy as np
import torch
import torch.nn as nn
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
from mean_iou_evaluate import read_masks, mean_iou_score
import PIL
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import sys

#batch_size = 8
batch_size = 1

class satDataset(Dataset):
    def __init__(self, transform, path):
        self.transform = transform
        self.path = path
        self.train_id_list = sorted(list(set([i.split("_")[0] for i in os.listdir(self.path)])))
        self.fname = []
        for file_prefix in self.train_id_list:
            self.fname.append(file_prefix)
    def __getitem__(self, index):
        file_path = os.path.join(self.path, self.fname[index]+"_sat.jpg") 
        image = PIL.Image.open(file_path)
        image = self.transform(image)
        return image, self.fname[index]
    def __len__(self):
        return len(self.train_id_list)

def save_file(file_path, img_prefix, predicted):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    RGB_masks = np.empty((len(predicted), 512, 512, 3))
    for i, pred in enumerate(predicted):
        RGB_masks[i, pred == 0] = [0, 255, 255]
        RGB_masks[i, pred == 1] = [255, 255, 0]
        RGB_masks[i, pred == 2] = [255, 0, 255]
        RGB_masks[i, pred == 3] = [0, 255, 0]
        RGB_masks[i, pred == 4] = [0, 0, 255]
        RGB_masks[i, pred == 5] = [255, 255, 255]
        RGB_masks[i, pred == 6] = [0, 0, 0]
    RGB_masks = RGB_masks.astype(np.uint8)
    for i, rgb_m in enumerate(RGB_masks):
        filename = img_prefix[i] + "_mask.png"
        path = os.path.join(file_path, filename)
        img = PIL.Image.fromarray(rgb_m) #把 numpy 轉成 PIL.Image
        img.save(path)

if __name__=='__main__':
    checkpoint_path = './p2_B_best.pth'
    img_path = sys.argv[1]
    file_path = sys.argv[2]

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.408507245, 0.37851192, 0.280892825], std=[0.14260272161078436, 0.10846701523901407, 0.09821331597487479])
    ])

    test_dataset = satDataset(transform=img_transform, path=img_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    #device = "cpu"
    criterion = nn.CrossEntropyLoss()

    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    out_channels = 7
    model.classifier = DeepLabHead(2048, out_channels)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    model.eval()
    for i, data in enumerate(tqdm(test_loader)):
            images = data[0].to(device)
            img_prefix = data[1]
            output = model(images)['out']
            predicted = torch.argmax(output, dim=1).cpu().numpy()
            output.max(dim=1)[1].data.cpu().numpy()
            save_file(file_path=file_path, img_prefix=img_prefix, predicted=predicted)
    '''pred = read_masks(file_path)
    lbl = read_masks("./hw1_data/p2_data/validation/")
    miou = mean_iou_score(pred=pred, labels=lbl)
    print("testing mean iou= {:.4f}".format(miou))'''
