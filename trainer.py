import numpy as np
import time
import cv2

from torchsummary import summary
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms 

from layers import *
from utils.utils import *
from utils.mono_dataloader import MonoDataset, RandomCrop, Normalize, ToTensor
from GCNet.GCNetPlus import GCNetPlus
from GCNet.losses import SmoothL1Loss, SSIM

mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
train_transform = transforms.Compose([RandomCrop([256, 512]), Normalize(mean, std), ToTensor()])

def get_dataloder(cfg, num_worker=1):
    train_dst = MonoDataset(cfg, transforms=train_transform)
    val_dst = MonoDataset(cfg, transforms=None)

    train_loader = DataLoader(train_dst, batch_size=cfg.bs, shuffle=True, num_workers=num_worker, pin_memory=True)
    val_loader = DataLoader(val_dst, batch_size=1, shuffle=True, num_workers=0, pin_memory=None)
    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))
    num_train_samples = len(train_dst)
    
    return train_loader, val_loader, num_train_samples


class Trainer:
    def __init__(self, cfg, device, weight_path=None):
        self.cfg = cfg
        self.tfwriter = SummaryWriter(log_dir=cfg.TRAIN_TENSORBOARD_DIR)
        self.parameters_to_train = []
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
        self.device = device
        
        # load GCNetPlus
        self.model = GCNetPlus(cfg.max_disp).to(self.device)
        if weight_path:
            self.model.load_state_dict(torch.load(weight_path))
        summary(self.model, [(3, cfg.H, cfg.W), (3, cfg.H, cfg.W)])
        #if torch.cuda.device_count() > 1:
         #   self.model = nn.DataParallel(self.model)

        self.parameters_to_train += list(self.model.parameters())
        self.optimizer = optim.Adam(self.parameters_to_train, lr=cfg.lr)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, cfg.scheduler_step_size, 0.1)

        
        
        self.train_loader, self.val_loader, num_train_samples = get_dataloder(cfg, num_worker=cfg.num_worker)
        self.num_total_steps = num_train_samples // cfg.bs * cfg.num_epochs
        
        self.criterion = SmoothL1Loss().to(self.device)
        if cfg.use_ssim:
            self.ssim = nn.CrossEntropyLoss()

        self.num_epochs = cfg.num_epochs
        self.save_frequency = cfg.save_frequency

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()
        print("Training")
        self.model.train()
        for idx, data in enumerate(self.train_loader):
            rinp = data["right"].to(device=self.device, dtype=torch.float32)
            linp = data["left"].to(device=self.device, dtype=torch.float32)
            target = data["disp"].to(device=self.device, dtype=torch.float32)
            #print("shapes", rinp.shape, linp.shape, target.shape)
            #print("rl inp", rinp.max(), rinp.min(), linp.max(), linp.min())
            #print("target max min", target[mask].max(), target[mask].min())

            out1, out2, out3 = self.process_batch(rinp, linp)
            total_loss = self.compute_reprojection_loss(out1, out2, out3, target)
            if self.step %50==0:
                save_depth("R", idx, target, out1, rinp)
                save_depth("L", idx, target, out1, linp)
                print("losses", total_loss)
            #self.model_optimizer.zero_grad()
            #total_loss.backward()
            #self.optimizer.step()


            #late_phase = self.step % 400 == 0

            #if early_phase or late_phase:
            #    self.log_time(batch_idx, duration, losses["loss"].cpu().data)

              #  if "depth_gt" in inputs:
              #      self.compute_depth_losses(inputs, outputs, losses)

              #  self.log("train", inputs, outputs, losses)
              #  self.val()

            self.step += 1

    def process_batch(self, rinp, linp):
        """Pass a minibatch through the network and generate images and losses
        """
        out1, out2, out3 = self.model(rinp, linp)
        return out1, out2, out3

    def compute_reprojection_loss(self, out1, out2, out3, target):
        mask = (target > 0)
        mask = mask.detach_() 
        l1_loss1, l1_loss2, l1_loss3 = self.criterion(out1[mask], out2[mask], out3[mask], target[mask])
        #ssim_loss1 = self.ssim(out1, target)
        #ssim_loss2 = self.ssim(out2.unsqueeze(1), target.unsqueeze(1)).mean()
        #ssim_loss3 = self.ssim(out3.unsqueeze(1), target.unsqueeze(1)).mean()
        #reprojection_loss1 = 1 * ssim_loss1 + 1 * l1_loss1
        #reprojection_loss2 = 1 * ssim_loss2 + 1 * l1_loss2
        #reprojection_loss3 = 1 * ssim_loss3 + 1 * l1_loss3
        #total_loss = 0.5 * reprojection_loss1 + 0.7 * reprojection_loss2 + 1.0 * reprojection_loss3
        total_loss = 0.5 * l1_loss1 + 0.7 * l1_loss2 + 1.0 * l1_loss3
        return total_loss



