import torch
import torch.nn as nn


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, window_size=3):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(window_size, 1, padding=0)
        self.mu_y_pool = nn.AvgPool2d(window_size, 1, padding=0)
        self.sigma_x_pool = nn.AvgPool2d(window_size, 1, padding=0)
        self.sigma_y_pool = nn.AvgPool2d(window_size, 1, padding=0)
        self.sigma_xy_pool = nn.AvgPool2d(window_size, 1, padding=0)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.clip = int((window_size -1)/2)

    def forward(self, x, y):
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        x = x[:,:,self.clip:-self.clip,self.clip:-self.clip]
        y = y[:,:,self.clip:-self.clip,self.clip:-self.clip]

        sigma_x = self.sigma_x_pool((x  - mu_x)**2)
        sigma_y = self.sigma_y_pool((y - mu_y)** 2)

        sigma_xy = self.sigma_xy_pool((x- mu_x) * (y-mu_y))

        mu_x = mu_x[:,:,self.clip:-self.clip,self.clip:-self.clip]
        mu_y = mu_y[:,:,self.clip:-self.clip,self.clip:-self.clip]

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) , 0, 2)



def compute_reprojection_loss(step, disp1, disp2, disp3, target_disp, ssim):
    target = target_disp.unsqueeze(1).to(torch.float32)
    disp1_ = disp1.unsqueeze(1).to(torch.float32)
    disp2_ = disp2.unsqueeze(1).to(torch.float32)
    disp3_ = disp3.unsqueeze(1).to(torch.float32)
    ssim_loss1 = ssim(disp1_, target).mean()
    ssim_loss2 = ssim(disp2_, target).mean()
    ssim_loss3 = ssim(disp3_, target).mean()
    
    l1_loss1 = F.l1_loss(disp1, target_disp)
    l1_loss2 = F.l1_loss(disp2, target_disp)
    l1_loss3 = F.l1_loss(disp3, target_disp)
    repro_loss1 = 0.85 * ssim_loss1 + 0.15 * l1_loss1
    repro_loss2 = 0.85 * ssim_loss2 + 0.15 * l1_loss2
    repro_loss3 = 0.85 * ssim_loss3 + 0.15 * l1_loss3 
    #print('step {:05} l1loss1 {:.5} | l1loss2: {:.5} | l1loss3: {:.5} | rep1: {:.5} | rep2: {:.5}, rep3: {:.5}'.format(step, l1_loss1, l1_loss2, l1_loss3, repro_loss1.item(), repro_loss2.item(), repro_loss3.item()))
    reprojection_loss = 0.5 * repro_loss1 + 0.7 * repro_loss2 + 1.0 * repro_loss3
    return reprojection_loss



