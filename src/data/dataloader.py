import os
import json
import cv2
import matplotlib.pyplot as plt
import random
import math
import sys
import numpy as np

sys.path.append(os.getcwd())
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.data.transforms import *

split_new_label_path = './data/cuhk03/archive/splits_new_labeled.json'

with open(split_new_label_path, 'r') as f:
    data = json.load(f)

data = list(data[0].values())

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
    
# Test dataset
test_image_paths = []
test_labels = []
test_cam_paths = []
for image_path, pid, cam_id in data[1]:
    test_image_paths.append(image_path)
    test_labels.append(pid)
    test_cam_paths.append(cam_id)

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
        RandomPatch()
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128))
    ])
}

class CUHK04_Dataset(Dataset):
    def __init__(self, image_paths, labels, trans=None):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.trans = trans
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pid = self.labels[index]
        
        if self.trans:
            image = self.trans(image)
        
        return image, pid
    
def get_traindata():
    train_dataset = CUHK04_Dataset(train_image_paths, train_labels, trans=trans['train'])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    return train_loader

def get_total_pids():
    return np.max(train_labels) + 1