import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

class MonoDataset(data.Dataset):
    def __init__(self, config, transforms=None):
        self.height, self.width = config.H, config.W
        self.transforms = transforms
        # Right image
        right = os.listdir(config.right)
        right.sort()
        self.rights = [os.path.join(config.right, path) for path in right]
        # Left images
        left = os.listdir(config.left)
        left.sort()
        self.lefts = [os.path.join(config.left, path) for path in left]
        # target depth
        target = os.listdir(config.depth)
        target.sort()
        self.target = [os.path.join(config.depth, path) for path in target]
        
        assert (len(self.rights) == len(self.lefts))

    def __len__(self):
        return len(self.lefts)

    def __getitem__(self, index):
        data = {}
        data["right"] = cv2.imread(self.rights[index])
        data["left"] = cv2.imread(self.lefts[index])
        data["disp"] = cv2.imread(self.target[index])[:, :, 0]

        if self.transforms:
            data = self.transforms(data)
        return data



class RandomCrop():

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        new_h, new_w = self.output_size
        h, w, _ = sample['left'].shape
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        for key in sample:
            sample[key] = sample[key][top: top + new_h, left: left + new_w]

        return sample


class Normalize():
    '''
    RGB mode
    '''

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['left'] = sample['left'] / 255.0
        sample['right'] = sample['right'] / 255.0

        sample['left'] = self.__normalize(sample['left'])
        sample['right'] = self.__normalize(sample['right'])

        return sample

    def __normalize(self, img):
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        return img


class ToTensor():

    def __call__(self, sample):
        left = sample['left']
        right = sample['right']

        # H x W x C ---> C x H x W
        sample['left'] = torch.from_numpy(left.transpose([2, 0, 1])).type(torch.FloatTensor)
        sample['right'] = torch.from_numpy(right.transpose([2, 0, 1])).type(torch.FloatTensor)

        if 'disp' in sample:
            sample['disp'] = torch.from_numpy(sample['disp']).type(torch.FloatTensor)

        return sample





