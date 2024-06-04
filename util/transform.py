import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class TransformManager:
    def __init__(self, device, resize):
        self.device = device
        self.resize = resize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, images, depths=None):
        images = images.to(self.device)
        images = torch.nn.functional.interpolate(images, size=self.resize, mode='bilinear', align_corners=False)
        images = self.normalize(images)
        if depths is None:
            return images, None
        depths = depths.to(self.device)
        depths = torch.nn.functional.interpolate(depths, size=self.resize, mode='bilinear')
        depths = depths.squeeze(1)
        disparities = self.depth_to_disparity(depths)
        disparities = (disparities - disparities.min()) / (disparities.max() - disparities.min())
        return images, disparities
    
    def depth_to_disparity(self, depths):
        disparities = torch.where(depths > 0, 1.0 / depths, torch.tensor(1e6, device=depths.device))
        return disparities
    
    def generate_random_mask(self, batch_size, height, width, device):
        masks = torch.zeros(batch_size, 1, height, width, device=device)
        
        for i in range(batch_size):
            rect_height = torch.randint(low=int(0.2 * height), high=int(0.5 * height), size=(1,)).item()
            rect_width = torch.randint(low=int(0.2 * width), high=int(0.5 * width), size=(1,)).item()
            start_y = torch.randint(low=0, high=height - rect_height, size=(1,)).item()
            start_x = torch.randint(low=0, high=width - rect_width, size=(1,)).item()

            masks[i, 0, start_y:start_y + rect_height, start_x:start_x + rect_width] = 1
            
        return masks

    def batch_cutMix(self, batch_uimage_a, batch_ulabel_a, batch_uimage_b, batch_ulabel_b):
        if batch_uimage_a.size() != batch_uimage_b.size():
            raise ValueError("CutMix: Input batches must have the same dimensions.")
        
        batch_uimage_a, batch_uimage_b = batch_uimage_a.to(self.device), batch_uimage_b.to(self.device)
        batch_ulabel_a, batch_ulabel_b = batch_ulabel_a.to(self.device), batch_ulabel_b.to(self.device)
        # 50% 原图
        if torch.rand(1).item() < 0.5:
            return batch_uimage_a, batch_ulabel_a, batch_uimage_b, batch_ulabel_b
        batch_size, channels, height, width = batch_uimage_a.shape
        
        masks = self.generate_random_mask(batch_size, height, width, self.device)
        
        # expand masks for images
        masks_expanded = masks.expand(batch_size, channels, height, width)
        
        mixed_images_a = batch_uimage_a * masks_expanded + batch_uimage_b * (1 - masks_expanded)
        mixed_images_b = batch_uimage_b * masks_expanded + batch_uimage_a * (1 - masks_expanded)
        
        # masks for labels
        masks = masks.squeeze(1) 
        
        mixed_labels_a = batch_ulabel_a * masks + batch_ulabel_b * (1 - masks)
        mixed_labels_b = batch_ulabel_b * masks + batch_ulabel_a * (1 - masks)
        
        return mixed_images_a, mixed_labels_a, mixed_images_b, mixed_labels_b