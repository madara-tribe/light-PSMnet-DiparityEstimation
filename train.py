import os
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torchsummary import summary
from torch.utils.data import DataLoader
from psmnet.PSMnetPlus import PSMNetPlus
#from models.PSMnet import PSMNet as GCNetPlus
from psmnet.smoothloss import SmoothL1Loss, SSIM
from psmnet.ssim import SSIM
from utils.disp_dataloader import DispDataLoder, RandomCrop, ToTensor, Normalize, Pad
from utils.utils import save_depth 
from cfg import Cfg
import tensorboardX as tX

# imagenet
mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
device_ids = [0, 1, 2, 3]
model_path = None
model_dir = "checkpoint"
os.makedirs(model_dir, exist_ok=True)
cfg = Cfg
writer = tX.SummaryWriter(log_dir=cfg.TRAIN_TENSORBOARD_DIR, comment='FSMNet')
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
print(device)


def main():
    train_transform = T.Compose([RandomCrop([256, 512]), Normalize(mean, std), ToTensor()])
    train_dataset = DispDataLoder(cfg, mode="train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    validate_transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(384, 1248)])
    validate_dataset = DispDataLoder(cfg, mode="val", transform=validate_transform)
    validate_loader = DataLoader(validate_dataset, batch_size=cfg.validate_batch_size, num_workers=cfg.num_workers)

    step = 0
    best_error = 100.0

    model = PSMNetPlus(cfg.maxdisp).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    summary(model, [(3, cfg.H, cfg.W), (3, cfg.H, cfg.W)])
    criterion = SmoothL1Loss().to(device)
    #ssim = SSIM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    if model_path is not None:
        state = torch.load(model_path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        best_error = state['error']
        print('load model from {}'.format(model_path))

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        step = train(model, train_loader, optimizer, criterion, step, epoch)
        adjust_lr(optimizer, epoch)

        if epoch % cfg.epoch_save_frequency == 0:
            model.eval()
            error = validate(model, validate_loader, epoch)
            best_error = save(model, optimizer, epoch, step, error, best_error)

def validate(model, validate_loader, epoch):
    '''
    validate 40 image pairs
    '''
    num_batches = len(validate_loader)
    idx = np.random.randint(num_batches)

    avg_error = 0.0
    for i, batch in enumerate(validate_loader):
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)

        mask = (target_disp > 0)
        mask = mask.detach_()

        with torch.no_grad():
            _, _, disp = model(left_img, right_img)

        delta = torch.abs(disp[mask] - target_disp[mask])
        error_mat = (((delta >= 3.0) + (delta >= 0.05 * (target_disp[mask]))) == 2)
        error = torch.sum(error_mat).item() / torch.numel(disp[mask]) * 100

        avg_error += error
        if i == idx:
            left_save = left_img
            disp_save = disp

    avg_error = avg_error / num_batches
    print('epoch: {:03} | 3px-error: {:.5}%'.format(epoch, avg_error))
    writer.add_scalar('error/3px', avg_error, epoch)
    save_image(left_save[0], disp_save[0], epoch)

    return avg_error


def save_image(left_image, disp, epoch):
    for i in range(3):
        left_image[i] = left_image[i] * std[i] + mean[i]
    b, r = left_image[0], left_image[2]
    left_image[0] = r  # BGR --> RGB
    left_image[2] = b
    # left_image = torch.from_numpy(left_image.cpu().numpy()[::-1])

    disp_img = disp.detach().cpu().numpy()
    fig = plt.figure(12.84, 3.84)
    plt.axis('off')  # hide axis
    plt.imshow(disp_img)
    plt.colorbar()

    writer.add_figure('image/disp', fig, global_step=epoch)
    writer.add_image('image/left', left_image, global_step=epoch)


def train(model, train_loader, optimizer, criterion, step, epoch):
    '''
    train one epoch
    '''
    for idx, batch in enumerate(train_loader):
        step += 1
        optimizer.zero_grad()

        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)

        mask = (target_disp > 0)
        mask = mask.detach_()

        disp1, disp2, disp3 = model(left_img, right_img)
        if step % cfg.step_save_frequency ==0:
            save_depth("L", step, disp3, target_disp, left_img)
            save_depth("R", step, disp3, target_disp, right_img)
        loss1, loss2, loss3 = criterion(disp1[mask], disp2[mask], disp3[mask], target_disp[mask])
        #reprojection_loss = compute_reprojection_loss(step, disp1, disp2, disp3, target_disp)
        total_loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3 # reprojection_loss

        total_loss.backward()
        optimizer.step()

        # print(step)

        if step % cfg.log_frequency == 0:
            writer.add_scalar('loss/loss1', loss1, step)
            writer.add_scalar('loss/loss2', loss2, step)
            writer.add_scalar('loss/loss3', loss3, step)
            #writer.add_scalar('loss/reprojection_loss', reprojection_loss, step)
            writer.add_scalar('loss/total_loss', total_loss, step)
            print('step/Epochs: {:05}/{:05} | total loss: {:.5} | loss1: {:.5} | loss2: {:.5} | loss3: {:.5}, reprojection_loss: {:.5}'.format(step, epoch, total_loss.item(), loss1.item(), loss2.item(), loss3.item(), 0.0))

    return step


def adjust_lr(optimizer, epoch):
    if epoch == 200:
        lr = 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def save(model, optimizer, epoch, step, error, best_error):
    path = os.path.join(model_dir, '{:03}.ckpt'.format(epoch))
    # torch.save(model.state_dict(), path)
    # model.save_state_dict(path)

    state = {}
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['error'] = error
    state['epoch'] = epoch
    state['step'] = step

    torch.save(state, path)
    print('save model at epoch{}'.format(epoch))

    if error < best_error:
        best_error = error
        best_path = os.path.join(model_dir, 'best_model.ckpt'.format(epoch))
        shutil.copyfile(path, best_path)
        print('best model in epoch {}'.format(epoch))

    return best_error

def compute_reprojection_loss(step, disp1, disp2, disp3, target_disp):
    target = target_disp #.unsqueeze(1)
    disp1_ = disp1#.unsqueeze(1)
    disp2_ = disp2#.unsqueeze(1)
    disp3_ = disp3#.unsqueeze(1)
    ssim_loss1 = SSIM(target, target)#.mean(1, True)
    ssim_loss2 = SSIM(target, target)#.mean(1, True)
    ssim_loss3 = SSIM(target, target)#.mean(1, True)
    
    l1_loss1 = F.l1_loss(disp1_, target)#.mean()
    l1_loss2 = F.l1_loss(disp2_, target)#.mean(1, True)
    l1_loss3 = F.l1_loss(disp3_, target)#.mean(1, True)
    repro_loss1 = ssim_loss1.mean() + l1_loss1.mean()
    repro_loss2 = ssim_loss2.mean() + l1_loss2.mean()
    repro_loss3 = ssim_loss3.mean() + l1_loss3.mean()
#    print('step {:05} l1loss1 {:.5} | l1loss2: {:.5} | l1loss3: {:.5} | rep1: {:.5} | rep2: {:.5}, rep3: {:.5}'.format(step, l1_loss1, l1_loss2, l1_loss3, repro_loss1.item(), repro_loss2.item(), repro_loss3.item()))
    reprojection_loss = 0.5 * repro_loss1 + 0.7 * repro_loss2 + 1.0 * repro_loss3
    return reprojection_loss
 
if __name__ == '__main__':
    main()
    writer.close()

