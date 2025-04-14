import torch
import os
from byol_pytorch import BYOL
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, transforms
from PIL import Image
import numpy as np
import random

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class UnlabeledDataset(Dataset): 
    def __init__(self, path, transform=None, files=None):
        super(UnlabeledDataset).__init__()
        self.path = path
        self.transform = transform
        if files is None:
            self.files = sorted([f for f in os.listdir(path) if f.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))
        else:
            self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        img_path = os.path.join(self.path, self.files[idx])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img

dataset_path = './hw1_data/p1_data/mini/train'
dataset = UnlabeledDataset(path=dataset_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

resnet = models.resnet50(weights=None).to(device)
learner = BYOL(
    resnet,
    image_size=128,  
    hidden_layer='avgpool', 
).to(device)


optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)  
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2)

num_epochs = 100
best_train_loss = 0.8

for epoch in range(num_epochs):
    epoch_loss = 0.0  # Initialize epoch loss
    for images in data_loader:
        images = images.to(device) 
        loss = learner(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        learner.update_moving_average() # Update moving average of target encoder
        epoch_loss += loss.item()  # Accumulate loss
    avg_epoch_loss = epoch_loss / len(data_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], avg_loss: {avg_epoch_loss:.4f}')
    if avg_epoch_loss < best_train_loss:
        best_train_loss = avg_epoch_loss
        checkpoint_path = './pretrain_best.pt'
        print(f'best avg_loss: {best_train_loss:.4f}')
        torch.save(resnet.state_dict(), checkpoint_path)
    scheduler.step()
