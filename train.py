from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
from cfg import Cfg

options = MonodepthOptions()
opts = options.parse()
cfg = Cfg

if __name__ == "__main__":
    trainer = Trainer(opts, cfg)
    trainer.train()


