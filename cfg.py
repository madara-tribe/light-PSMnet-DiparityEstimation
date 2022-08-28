import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.batch_size = 1
Cfg.validate_batch_size=1
Cfg.H = 256
Cfg.W = 512
Cfg.lr = 0.001
Cfg.num_epochs = 300
Cfg.gpu_id = '1'
Cfg.use_ssim = True
Cfg.num_workers = 4
Cfg.maxdisp = 192
## dataset
Cfg.right_dir = "data/train/Right/*.jpg"
Cfg.left_dir = "data/train/Left/*.jpg"
Cfg.disp_dir = "data/train/disp/*.png"
Cfg.val_right_dir = "data/valid/Right/*.jpg"
Cfg.val_left_dir = "data/valid/Left/*.jpg"
Cfg.val_disp_dir = "data/valid/disp/*.png"
Cfg.log_frequency = 2
Cfg.step_save_frequency = 40
Cfg.epoch_save_frequency = 2
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')

