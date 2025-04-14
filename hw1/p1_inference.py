import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from PIL import Image
import pandas as pd
import sys
from tqdm.auto import tqdm

class customDataset(Dataset):
    def __init__(self, csv_file, path, tfm=None):
        self.path = path
        self.data = pd.read_csv(csv_file)  # Read the CSV file
        self.transform = tfm
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # Get filename and id from CSV
        filename = self.data.iloc[idx]['filename']
        id = self.data.iloc[idx]['id']
        label = self.data.iloc[idx]['label']  # Not used in inference but kept for completeness
        image = Image.open(os.path.join(self.path, filename))
        if self.transform:
            image = self.transform(image)
        return image, id, filename

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.resnet = models.resnet50(weights=None)
        in_features = self.resnet.fc.out_features  #1000
        self.new_classifier = nn.Sequential(     
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.resnet(x)
        x = self.new_classifier(x) 
        return x

image_size = 128
if __name__ == '__main__':
    csv_file = sys.argv[1] # './hw1_data/p1_data/office/val.csv'
    image_folder = sys.argv[2] # './hw1_data/p1_data/office/val'
    output_path = sys.argv[3] # './test_inference.csv'
    checkpoint_path = 'p1_C_best.ckpt'

    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), #convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #normalize
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use DataLoaders to load the data
    batch_size = 1 #256
    num_class = 65

    #testing
    test_set = customDataset(csv_file, image_folder, tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model_best = CustomModel(num_class).to(device)
    checkpoint = torch.load(checkpoint_path)
    model_best.load_state_dict(checkpoint['model_state_dict'])
    
    model_best.eval()
    predictions = []
    with torch.no_grad():
        for images, id, filenames in tqdm(test_loader):
            images = images.to(device)
            outputs = model_best(images)
            predicts = outputs.argmax(dim=-1)  
            predictions.extend(zip(id.cpu().numpy(), filenames, predicts.cpu().numpy()))
    # Save the predictions results in csv file
    result_df = pd.DataFrame(predictions, columns=['id', 'filename', 'label'])
    result_df.to_csv(output_path, index=False)