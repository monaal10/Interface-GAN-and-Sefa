#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import math
from torch import nn
import torchvision.models as models
from tqdm import tqdm


# In[ ]:


sys.executable


# In[ ]:



transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# In[ ]:


class Data(Dataset):
    def __init__(self, base_dir='/common/home/ms3185/Sefa/Dataset', transforms=None, is_train=True):
        self.base_dir = base_dir
        label_file = os.path.join(base_dir, 'imageLabels.txt')
        with open(label_file) as f:
            self.labels = [x.strip().lower() for x in f.read().split('\n') if x]
        self.images_path = os.path.join(base_dir, 'images')
        self.is_train = is_train
        self.transforms = transforms
        self.thresh = int(len(self.labels) * 0.8)
        self.indices = list(range(self.thresh)) if self.is_train else list(range(self.thresh, len(self.labels)))
        
    def __len__(self):
        return len(self.indices)
    
    def create_tensor(self, label):
        return torch.tensor([float(x) for x in label.split()])
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        label = self.labels[idx]
        image_name = os.path.join(self.images_path, str(idx + 1).zfill(5) + '.jpg')
        image = Image.open(image_name).convert('RGB')
        return self.transforms(image), self.create_tensor(label)


# In[ ]:


class Classifier(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.main = models.resnet18(pretrained=True)
        num_feat = self.main.fc.in_features
        self.main.fc = nn.Linear(num_feat, classes)

    def forward(self, img):
        return self.main(img)


# In[9]:


def train(model, train_data, val_data):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = 'cuda:2'

    for epoch in range(50):
        model.train()
        train_loss = 0
        train_total = 0
        val_total = 0
        print(f'Train epoch: {epoch}')
        for image, labels in (train_data):
            optimizer.zero_grad()
            image = image.to(device)
            labels = labels.to(device)

            outputs = model(image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total += 1
        
        train_loss /= train_total
        print(f'{train_loss=}')
        tr=[]
        tr.append(train_loss)
        with open("tr.txt", "w") as output:
            output.write(str(tr))
        torch.save(tr,'tr.pth')
        model.eval()
        val_loss = 0
        
        device='cuda:2'
        
        with torch.no_grad():
            for image, labels in tqdm(val_data):
                image = image.to(device)
                labels = labels.to(device)
                val_total += 1
                outputs = model(image)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        val_loss /= val_total
        print(f'{val_loss}')
        val=[]
        val.append(val_loss)
        with open("val.txt", "w") as output:
            output.write(str(val))
        torch.save(val,'val.pth')
        x=model.state_dict()
        torch.save(x,'model.pth')


# In[10]:


device='cuda:2'
model =Classifier().to(device)
train_dataset = PizzaData(transforms=transform)
test_dataset = PizzaData(transforms=transform, is_train=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=0)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=0)


# In[ ]:


train(model, train_loader, val_loader)


# In[ ]:




