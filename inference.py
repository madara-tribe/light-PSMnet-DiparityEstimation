import argparse
import os 
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from psmnet.PSMnetPlus import PSMNetPlus
from utils.disp_dataloader import ToTensor, Normalize
import torch.nn.functional as F
from utils.utils import disp2np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#python3 inference.py --right pic/test_right.png --left pic/test_left.png

parser = argparse.ArgumentParser(description='PSMNet inference')
parser.add_argument('--maxdisp', type=int, default=192, help='max diparity')
parser.add_argument('--left', default=None, help='path to the left image')
parser.add_argument('--right', default=None, help='path to the right image')
args = parser.parse_args()


mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
device_ids = [0, 1, 2, 3]
device = torch.device('cuda:{}'.format(device_ids[0]))


def main(model_path):
    left = cv2.imread(args.left)
    right = cv2.imread(args.right)
    left = cv2.resize(left, (1024, 256))
    right = cv2.resize(right, (1024, 256))
    
    pairs = {'left': left, 'right': right}

    transform = T.Compose([Normalize(mean, std), ToTensor()]) #Pad(192, 640)])
    pairs = transform(pairs)
    left = pairs['left'].to(device).unsqueeze(0)
    right = pairs['right'].to(device).unsqueeze(0)
    print(left.shape, right.shape)
    model = PSMNetPlus(args.maxdisp).to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    state = torch.load(model_path)
    if len(device_ids) == 1:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['state_dict'].items():
            namekey = k[7:] # remove `module.`
            new_state_dict[namekey] = v
        state['state_dict'] = new_state_dict

    model.load_state_dict(state['state_dict'])
    print('load model from {}'.format(model_path))
    print('epoch: {}'.format(state['epoch']))
    print('3px-error: {}%'.format(state['error']))

    model.eval()

    with torch.no_grad():
        _, _, disp = model(left, right)

    disp = disp.squeeze(0).detach().cpu().numpy()
    disp = disp2np(disp) 
    print("disp", disp.shape, left.shape, right.shape)
    cv2.imwrite("pic/test_disp.png", disp.astype(np.uint8))


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
        # disp = sample['disp'].unsqueeze(0).unsqueeze(1)  # [1, 1, H, W]
        # disp = F.pad(disp, pad=(0, pad_w, 0, pad_h))

        sample['left'] = left.squeeze()
        sample['right'] = right.squeeze()
        # sample['disp'] = disp.squeeze()

        return sample


if __name__ == '__main__':
    model_path = os.path.join("checkpoint", "best_model.ckpt")
    main(model_path)
