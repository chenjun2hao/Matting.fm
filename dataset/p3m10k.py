import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import Resize, NormalizeImage, PrepareForNet, Crop
from PIL import Image
import numpy as np
import pickle
from glob import glob
import os.path as osp


class P3M10K(Dataset):
    def __init__(self, filelist_path, mode, size=512, data_name=None):
        
        self.mode = mode
        self.size = size
        self.filelist = []

        if filelist_path.endswith(".pkl"):
            with open(filelist_path, 'rb') as f:
                targets = pickle.loads(f.read())
            for target in targets:
                img_name = target['image_name']
                alpha_path = target['segmentation']
                self.filelist.append([img_name, alpha_path])
        elif filelist_path.endswith(".txt"):
            raise NotImplementedError
        
        if data_name=='real636':
            images = glob("/data1/Segmentation/matting/RealWorldPortrait-636/image/*") * 4
            alpha_dir = "/data1/Segmentation/matting/RealWorldPortrait-636/alpha/"
            for fg in images:
                alpha = alpha_dir + osp.basename(fg).replace('.jpg', '.png')
                self.filelist.append([fg, alpha])
        
        
        net_w, net_h = size, size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            PrepareForNet(),
        ] + ([Crop(size)] if self.mode == 'train' else []))
    
    def __getitem__(self, item):
        img_path = self.filelist[item][0]
        depth_path = self.filelist[item][1]

        image = np.asarray(Image.open(img_path).convert('RGB'))
        alpha = Image.open(depth_path).convert("L")
        alpha = np.asarray(alpha) / 255.0
        
        sample = self.transform({'image': image, 'mask': alpha})
        image = sample['image'] / 127.5 - 1

        sample['rgb_norm'] = torch.from_numpy(image)
        sample['alpha']  = torch.from_numpy(sample['mask']).unsqueeze(0)
        
        sample['image_path'] = self.filelist[item][0]
        
        return sample

    def __len__(self):
        return len(self.filelist)