_exp_name = "p1_A" 
import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm.auto import tqdm
import random
import torchvision.models as models
from argparse import ArgumentParser


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

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    #transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)), 
    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=(-0.2, 0.2)),
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
    def __init__(self, num_classes, pretrain_path):
        super(CustomModel, self).__init__()
        self.resnet = models.resnet50(weights=None)
        if pretrain_path: 
            self.resnet.load_state_dict(torch.load(pretrain_path))  
            print('pretrain path: ', pretrain_path)
        in_features = self.resnet.fc.out_features  # 原始 fc 的输出特征维度 #1000
        self.new_classifier = nn.Sequential(    # 新分类器    
            nn.Linear(in_features, 512),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.resnet(x) 
        x = self.new_classifier(x) 
        return x

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

if __name__ == '__main__':
    parser = ArgumentParser('HW1 p1')
    parser.add_argument('--pretrain_path',
                        default=None, type=str, help='Please enter the path of the pretrained checkpoint.')
    parser.add_argument('--data_path',
                        default='./hw1_data/p1_data/office/', type=str, help='Please enter the path of dataset.')
    args = parser.parse_args()
    pretrain_path = args.pretrain_path
    data_path = args.data_path

    batch_size = 128
    n_epochs = 250
    patience = 100  #early stop
    criterion = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CustomModel(num_classes=65, pretrain_path=pretrain_path)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4) 
    checkpoint_path = None
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    #dataloader
    train_set = customDataset(os.path.join(data_path, "train"), tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_set = customDataset(os.path.join(data_path, "val"), tfm=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    #training
    stale = 0
    best_acc = 0
    #resume from checkpoint
    #stale = 
    #best_acc = 

    for epoch in range(start_epoch, n_epochs):

        # ---------- Training ----------
        model.train()
        train_loss = []
        train_accs = []
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader):
            imgs, labels = batch
            logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels.to(device)).sum().item()
            train_total += labels.size(0)

            train_loss.append(loss.item())
            

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = train_correct / train_total
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        model.eval()

        valid_loss = []
        correct = 0
        total = 0

        for batch in tqdm(valid_loader):
            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            
            preds = logits.argmax(dim=-1)
            correct += (preds == labels.to(device)).sum().item()
            total += labels.size(0)

            # Record the loss and accuracy.
            valid_loss.append(loss.item())

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = correct / total

        if valid_acc > best_acc: 
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:    
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            checkpoint_path = f"{_exp_name}_best.ckpt"
            torch.save(checkpoint, checkpoint_path) 
            
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience or epoch == n_epochs-1:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                checkpoint_path = f"{_exp_name}_last_{epoch}.ckpt"
                torch.save(checkpoint, checkpoint_path) 
                break

    print(f"best_acc = {best_acc:.5f}")
