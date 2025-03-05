import json
import math
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.transforms import *

# load data
split_new_label_path = './data/cuhk03/archive/splits_new_labeled.json'
with open(split_new_label_path, 'r') as f:
    data = json.load(f)

data = list(data[0].values())

# load config
config_path = './src/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

def process_image_paths(data):
    for i in range(3):
        for idx, (image_path, pid, cam_id) in enumerate(data[i]):
            image_path = image_path.split('\\')
            image_path = os.path.join(image_path[0], image_path[1], 'archive', image_path[2], image_path[3])
            data[i][idx][0] = image_path
            
process_image_paths(data)

# Train dataset
train_image_paths = []
train_labels = []
train_cam_paths = []
for image_path, pid, cam_id in data[0]:
    train_image_paths.append(image_path)
    train_labels.append(pid)
    train_cam_paths.append(cam_id)
    
# query dataset
query_image_paths = []
query_labels = []
query_cam_paths = []
for image_path, pid, cam_id in data[1]:
    query_image_paths.append(image_path)
    query_labels.append(pid)
    query_cam_paths.append(cam_id)

# Gallery dataset
gallery_image_paths = []
gallery_label_paths = []
gallery_cam_paths = []
for image_path, pid, cam_id in data[2]:
    gallery_image_paths.append(image_path)
    gallery_label_paths.append(pid)
    gallery_cam_paths.append(cam_id)

trans = {
    "train": transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((256, 128), scale=(0.8, 1), ratio=(0.25, 4)),
        RandomPatch(),
        
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    "query": transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128))
    ])
}

class CUHK03_Dataset(Dataset):
    def __init__(self, image_paths, labels, trans=None, get_image_path=False):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.trans = trans
        self.get_image_path = get_image_path
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pid = self.labels[index]
        
        if self.trans:
            image = self.trans(image)
        
        if self.get_image_path:
            return image, pid, self.image_paths[index]
        else:
            return image, pid
    
def get_traindata():
    train_dataset = CUHK03_Dataset(train_image_paths, train_labels, trans=trans['train'])
    train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True)
    
    return train_loader

def get_query():
    query_dataset = CUHK03_Dataset(query_image_paths, query_labels, trans=trans['query'])
    query_loader = DataLoader(query_dataset, batch_size=config['dataset']['batch_size'], shuffle=True)

    return query_loader

def get_gallery(get_image_path=False):   
    gallery_dataset = CUHK03_Dataset(gallery_image_paths, gallery_label_paths, trans=trans['query'], get_image_path=get_image_path)
    gallery_loader = DataLoader(gallery_dataset, batch_size=config['dataset']['batch_size'], shuffle=False)
    
    return gallery_loader

def get_total_pids():
    return np.max(train_labels) + 1