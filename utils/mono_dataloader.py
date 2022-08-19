import os
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import torch
import torch.utils.data as data
from torchvision import transforms

class MonoDataset(data.Dataset):
    def __init__(self, config, valid=None):
        self.height, self.width = config.H, config.W
        # Right image
        right_path = config.right if not valid else config.val_right
        right = os.listdir(right_path)
        right.sort()
        self.rights = [os.path.join(right_path, path) for path in right]
        # Left images
        left_path = config.left if not valid else config.val_left
        left = os.listdir(left_path)
        left.sort()
        self.lefts = [os.path.join(left_path, path) for path in left]

        self.to_tensor = transforms.ToTensor()
        
        assert (len(self.rights) == len(self.lefts))

    def __len__(self):
        return len(self.lefts)

    def preprocess(self, path):
        x = Image.open(path)
        x = x.convert('RGB')
        x = x.resize((self.height, self.width))
        return self.to_tensor(x)

    def __getitem__(self, index):
      
        right = self.preprocess(self.rights[index])
        left = self.preprocess(self.lefts[index])
        return right, left


