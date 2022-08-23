import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.bs = 1
Cfg.H = 256
Cfg.W = 512
Cfg.lr = 1e-4
Cfg.num_epochs = 30
Cfg.val_interval = 500
Cfg.gpu_id = '1'
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
#Cfg.TRAIN_OPTIMIZER = 'sgd'
Cfg.TRAIN_OPTIMIZER = 'adam'
Cfg.use_ssim = True
Cfg.num_worker = 0
Cfg.max_disp = 192
Cfg.scheduler_step_size = 15
## dataset
Cfg.right = "data/train/Right"
Cfg.left = "data/train/Left"
Cfg.depth = "data/train/target2"
Cfg.val_right = "data/valid/Right"
Cfg.val_left = "data/valid/Left"
Cfg.val_depth = "data/valid/target2"
Cfg.save_frequency = 20
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')

