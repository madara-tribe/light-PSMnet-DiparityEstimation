from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from os.path import join
import cv2
import numpy as np
import glob


class DispDataLoder(Dataset):

    def __init__(self, cfg, mode, transform=None):
        super().__init__()

        self.mode = mode
        self.transform = transform

        if mode == 'train':
            self.left_imgs = sorted(glob.glob(cfg.left_dir))
            self.right_imgs = sorted(glob.glob(cfg.right_dir))
            self.disp = sorted(glob.glob(cfg.disp_dir))
        elif mode == 'val' or mode == 'test':
            N = 200
            left_imgs = sorted(glob.glob(cfg.val_left_dir))
            self.left_imgs = left_imgs[:N]
            right_imgs = sorted(glob.glob(cfg.val_right_dir))
            self.right_imgs = right_imgs[:N]
            disp = sorted(glob.glob(cfg.val_disp_dir))
            self.disp = disp[:N]


    def __len__(self):
        return len(self.left_imgs)
    
    def preprocess(self, path, disp_is=False):
        if disp_is:
            x = cv2.imread(path)[:, :, 0]
        else:
            x = cv2.imread(path)
        x = cv2.resize(x, (640, 320))
        return x

    def __getitem__(self, idx):
        data = {}
        # bgr mode
        if self.mode == "train":
            data['left'] = cv2.imread(self.left_imgs[idx])
            data['right'] = cv2.imread(self.right_imgs[idx])
            data['disp'] = cv2.imread(self.disp[idx])[:, :, 0]
        elif self.mode == "val" or self.mode == "test":
            data['left'] = self.preprocess(self.left_imgs[idx])
            data['right'] = self.preprocess(self.right_imgs[idx])
            if self.mode != 'test':
                data['disp'] = self.preprocess(self.disp[idx], disp_is=True)

        if self.transform:
            data = self.transform(data)

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


class Pad():
    def __init__(self, H, W):
        self.w = W
        self.h = H

    def __call__(self, sample):
        pad_h = self.h - sample['left'].size(1)
        pad_w = self.w - sample['left'].size(2)

        left = sample['left'].unsqueeze(0)  # [1, 3, H, W]
        left = F.pad(left, pad=(0, pad_w, 0, pad_h))
        right = sample['right'].unsqueeze(0)  # [1, 3, H, W]
        right = F.pad(right, pad=(0, pad_w, 0, pad_h))
        disp = sample['disp'].unsqueeze(0).unsqueeze(1)  # [1, 1, H, W]
        disp = F.pad(disp, pad=(0, pad_w, 0, pad_h))

        sample['left'] = left.squeeze()
        sample['right'] = right.squeeze()
        sample['disp'] = disp.squeeze()

        return sample



