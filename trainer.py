import numpy as np
import time
from torchsummary import summary
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
from utils.utils import *
from utils.mono_dataloader import MonoDataset
from kitti_utils import *
from layers import *
from test_depth import test_depth
import networks
from GCNet.GCNetPlus import GCNetPlus
from GCNet.smoothloss import SmoothL1Loss

def get_dataloder(cfg, num_worker=1):
    train_dst = MonoDataset(cfg, valid=None)
    val_dst = MonoDataset(cfg, valid=True)

    train_loader = DataLoader(train_dst, batch_size=cfg.bs, shuffle=True, num_workers=num_worker, pin_memory=True)
    val_loader = DataLoader(val_dst, batch_size=1, shuffle=True, num_workers=0, pin_memory=None)
    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))
    num_train_samples = len(train_dst)
    
    return train_loader, val_loader, num_train_samples


class Trainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.tfwriter = SummaryWriter(log_dir=cfg.TRAIN_TENSORBOARD_DIR)
        self.parameters_to_train = []
        self.device = device
        # load pretrained monodepth model
        self.teacher_enc_model, self.teacher_depth_model = test_depth()
        self.teacher_enc_model.to(self.device)
        self.teacher_enc_model.eval()
        self.teacher_depth_model.to(self.device)
        self.teacher_depth_model.eval()
        
        # load GCNetPlus
        self.model = GCNetPlus(cfg.max_disp).to(self.device)
        #summary(self.model, [(3, cfg.H, cfg.W), (3, cfg.H, cfg.W)])
        self.parameters_to_train += list(self.model.parameters())
        
        self.optimizer = optim.Adam(self.parameters_to_train, lr=cfg.lr)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, cfg.scheduler_step_size, 0.1)

        #if self.opt.load_weights_folder is not None:
        #    self.load_model()
        #print(self.models)
        
        
        self.train_loader, self.val_loader, num_train_samples = get_dataloder(cfg, num_worker=cfg.num_worker)
        self.num_total_steps = num_train_samples // cfg.bs * cfg.num_epochs
        
        self.criterion = SmoothL1Loss()
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
        for idx, (rinp, linp) in enumerate(self.train_loader):
         
            rinp = rinp.to(device=self.device, dtype=torch.float32)
            linp = linp.to(device=self.device, dtype=torch.float32)
            
            right_depth = self.teacher_depth_model(self.teacher_enc_model(rinp))
            right_depth = F.interpolate(right_depth[("disp", 0)],
                (self.cfg.H, self.cfg.W), mode="bilinear", align_corners=False)
            left_depth = self.teacher_depth_model(self.teacher_enc_model(linp))
            left_depth = F.interpolate(left_depth[("disp", 0)],
                (self.cfg.H, self.cfg.W), mode="bilinear", align_corners=False)
            #print("r max min", rinp.shape, right_depth.shape)
            #print("l max min", linp.shape, left_depth.shape)
            #save_depth("R", idx, right_depth, rinp)
            #save_depth("L", idx, left_depth, linp)
            
            losses = self.process_batch(rinp, linp, right_depth, left_depth)
            print("losses", losses.keys())
            #self.model_optimizer.zero_grad()
            #losses.backward()
            #self.model_optimizer.step()


            #late_phase = self.step % 400 == 0

            #if early_phase or late_phase:
            #    self.log_time(batch_idx, duration, losses["loss"].cpu().data)

              #  if "depth_gt" in inputs:
              #      self.compute_depth_losses(inputs, outputs, losses)

              #  self.log("train", inputs, outputs, losses)
              #  self.val()

           # self.step += 1

    def process_batch(self, rinp, linp, right_depth, left_depth):
        """Pass a minibatch through the network and generate images and losses
        """
        rinp = rinp.to(device=self.device, dtype=torch.float32)
        linp = linp.to(device=self.device, dtype=torch.float32)
        right_depth = right_depth.to(device=self.device, dtype=torch.float32)
        left_depth = left_depth.to(device=self.device, dtype=torch.float32)
        out1, out2, out3 = self.model(rinp, linp)
        loss1, loss2, loss3 = self.criterion(out1, out2, out3)
        losses = loss1, loss2, loss3
        return losses


