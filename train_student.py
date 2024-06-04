
from dataset import DepthEstimationDataset,UDataset
from util.transform import TransformManager 
from dpt import DepthAnything,DPT_DINOv2_Encoder
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import loss_dep
import warnings

   
   



def main():
    #arg
    config = {
        'encoder': 'vits',
        'features': 256,
        'out_channels': [256, 512, 1024, 1024],
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True
    }
    teacher_config = {
        'encoder': 'vits',
        'features': 256,
        'out_channels': [256, 512, 1024, 1024],
        'use_bn': False,
        'use_clstoken': False,
        'localhub': True
    }
    l_image_dir = './dataset/imgs/'
    l_depth_dir = './dataset/gts/'
    u_imge_dir='./dataset/u_imgs/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # model load
    model = DepthAnything(config)
    teacher_model = DepthAnything(teacher_config)
    teacher_model.load_state_dict(torch.load('best_teacher_model.pth'))
    semantic_encoder=DPT_DINOv2_Encoder()
    
    
    model = model.to(device)
    teacher_model=teacher_model.to(device)
    semantic_encoder=semantic_encoder.to(device)
   
    
    
    
    
    
    
    # Dataset
    dataset = DepthEstimationDataset(l_image_dir, l_depth_dir)
    udataset=UDataset(u_imge_dir)
    # Split 
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
   
    
    
    # DataLoader
    batch_size =2
    ubatch_size=4
    l_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
    u_loader=DataLoader(udataset,batch_size=ubatch_size,shuffle=True,num_workers=1,drop_last=True)
    
    #train arg
    alpha=0.15    #semantic threshold
    num_epochs = 20
    best_val_loss = float('inf')
    criterion1 = loss_dep.InvariantLoss  
    criterion2 =loss_dep.SemanticConstraint
    encoder_lr = 5e-6
    decoder_lr = 5e-5
    encoder_params = list(model.pretrained.parameters())
    decoder_params = list(model.depth_head.parameters())
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': decoder_params, 'lr': decoder_lr}
    ])
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    transform_manager = TransformManager(device,resize=(518, 518))
    #train
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        teacher_model.eval()
        model.train() 
        train_loss = 0.0
    
        l_iter = iter(l_loader)
        u_iter = iter(u_loader)
        pbar = tqdm(total=len(l_loader), desc="Training")
        while True:
            try:
                
                l_data = next(l_iter)
                u_data = next(u_iter)
            except StopIteration:
                break
            #unlabel data
            u_images= u_data
            u_images=u_images.to(device)
            with torch.no_grad():
                u_pseudo_labels = teacher_model(u_images)
            #trans unlabel
            u_images,_=transform_manager(u_images)
            # batch/2
            half_batch_size = ubatch_size // 2
            u_images_a, u_images_b = torch.split(u_images, half_batch_size)
            u_labels_a, u_labels_b = torch.split(u_pseudo_labels, half_batch_size)
            
            #cutmix unlabel
            mixed_images_a, mixed_labels_a, mixed_images_b, mixed_labels_b = transform_manager.batch_cutMix(
                u_images_a, u_labels_a, u_images_b, u_labels_b
            )
            
            # concat cutmix
            u_images = torch.cat([mixed_images_a, mixed_images_b], dim=0)
            u_pseudo_labels = torch.cat([mixed_labels_a, mixed_labels_b], dim=0)
            
            #label data
            l_images, l_labels = l_data
            l_images, l_labels=l_images.to(device), l_labels.to(device)
            l_images, l_labels=transform_manager(l_images,l_labels)
            #concat label and unlabel
            combined_images = torch.cat((l_images, u_images), dim=0)
            combined_labels = torch.cat((l_labels, u_pseudo_labels), dim=0)
            
            
            optimizer.zero_grad()
            outputs = model(combined_images)
            loss = criterion1(outputs, combined_labels)+criterion2(model.get_features(u_images),semantic_encoder(u_images),1-alpha)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            pbar.update(1)  
        pbar.close() 
        #val
        model.eval() 
        val_loss = 0.0
        with torch.no_grad():  
             for images, depths in tqdm(val_loader, desc="Validation", leave=False):
                images, depths = transform_manager(images, depths)  
                images, depths = images.to(device), depths.to(device) 
                outputs = model(images)
                loss = criterion1(outputs, depths)
                val_loss += loss.item()
        train_loss /= len(l_loader)+len(u_loader)
        val_loss /= len(val_loader)
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        scheduler.step()
        # save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_student_model.pth')
            print('save model')
if __name__ == "__main__":
    print(torch.__version__) 
    warnings.filterwarnings("ignore")
    main()










