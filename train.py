import os
from trainer import Trainer
from cfg import Cfg
import torch

if __name__ == "__main__":
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    cfg = Cfg
    trainer = Trainer(cfg, device)
    trainer.train()
