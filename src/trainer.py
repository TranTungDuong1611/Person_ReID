import argparse
import json
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from src.data.dataloader import *
from src.model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    return config

config = read_config('./src/config.json')

def training_loop(model, train_loader, lr, epochs, weight_decay, config=config, device=device):
    os.makedirs('checkpoints', exist_ok=True)
    model = model.to(device)
    
    criterion = CrossEntropyLoss(label_smoothing=0.1)
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    
    train_losses = []
    best_weight = 1e9
    for epoch in range(epochs):
        model.train()
        
        batch_loss = []
        for i, (image, label) in tqdm(enumerate(train_loader)):
            image = image.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
            
            if (i+1) in config['osnet']['decay_time']:
                scheduler.step()
        
        train_loss = sum(batch_loss) / len(batch_loss)
        train_losses.append(train_loss)
        
        if train_loss < best_weight:
            best_weight = train_loss
            torch.save(model.state_dict(), './checkpoints/best_weight.pt')
        
        print(f"Epoch {epoch}/{epochs}\ttrain_loss: {train_loss:.4f}")
        
    plt.plot(range(len(train_losses)), train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.title('Training loss')
    os.makedirs('figures', exist_ok=True)
    plt.savefig('./figures/training_loss.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=config['osnet']['epochs'], help="Number training epochs")
    parser.add_argument('--lr', type=np.float32, default=config['osnet']['lr'], help="Learning rate of training model")
    parser.add_argument('--weight_decay', type=np.float32, default=config['osnet']['weight_decay'], help="Weight decay of training model")
    
    args = parser.parse_args()
    
    # define model
    num_classes = get_total_pids()
    model = OSNet_model(num_classes=num_classes)
    
    # train dataset
    train_loader = get_traindata()
    
    training_loop(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay
    )
    
if __name__ == '__main__':
    main()