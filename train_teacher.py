
from dataset import DepthEstimationDataset
from util.transform import TransformManager 
from dpt import DepthAnything
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import loss_dep

   
   



# 主训练脚本
def main():
    
    config = {
        'encoder': 'vits',
        'features': 256,
        'out_channels': [256, 512, 1024, 1024],
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True
    }
    
    model = DepthAnything(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    
    encoder_lr = 5e-6
    decoder_lr = 5e-5

    encoder_params = list(model.pretrained.parameters())
    decoder_params = list(model.depth_head.parameters())
    
    
    criterion = loss_dep.InvariantLoss  
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': decoder_params, 'lr': decoder_lr}
    ])
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    image_dir = './dataset/imgs/'
    depth_dir = './dataset/gts/'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    transform_manager = TransformManager(device,resize=(518, 518))
    # Dataset
    dataset = DepthEstimationDataset(image_dir, depth_dir)
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader
    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
  
    num_epochs = 20
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
       
        model.train() 
        train_loss = 0.0
        for images, depths in tqdm(train_loader,desc="Trainning", leave=False):
            images, depths = transform_manager(images, depths) 
            images, depths = images.to(device), depths.to(device) 
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
       
        model.eval() 
        val_loss = 0.0
        with torch.no_grad():  
             for images, depths in tqdm(val_loader, desc="Validation", leave=False):
                images, depths = transform_manager(images, depths)  
                images, depths = images.to(device), depths.to(device) 
                outputs = model(images)
                loss = criterion(outputs, depths)
                val_loss += loss.item()
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        scheduler.step()
            
        # save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_teacher_model.pth')
            print('Model saved!')
if __name__ == "__main__":
    main()










