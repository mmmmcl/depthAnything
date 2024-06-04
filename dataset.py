import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DepthEstimationDataset(Dataset):
    def __init__(self, image_dir, depth_dir, depth_ext='.png', resize=(518, 518)):
      
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.depth_ext = depth_ext
        self.images = os.listdir(image_dir)
        self.resize = resize
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        
      
        img_base_name, _ = os.path.splitext(self.images[idx])
        
 
        depth_name = os.path.join(self.depth_dir, img_base_name + self.depth_ext)
        
       
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0  
        image=np.ascontiguousarray(image).astype(np.float32)
        
        depth = cv2.imread(depth_name, cv2.IMREAD_GRAYSCALE)  
        depth=np.ascontiguousarray(depth).astype(np.float32)
        
        to_tensor = transforms.ToTensor()
        resize_transform = transforms.Resize(self.resize)
        if not isinstance(image, torch.Tensor):
            image = to_tensor(image)
        if not isinstance(depth, torch.Tensor):
            depth = to_tensor(depth)
        image = resize_transform(image)
        depth = resize_transform(depth)
        
        return image, depth
class UDataset(Dataset):
    def __init__(self, image_dir, resize=(518, 518)):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.resize = resize
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0  
        image=np.ascontiguousarray(image).astype(np.float32)
        
       
        
        to_tensor = transforms.ToTensor()
        resize_transform = transforms.Resize(self.resize)
        if not isinstance(image, torch.Tensor):
            image = to_tensor(image)
        image = resize_transform(image)
        
        
        return image       
       







