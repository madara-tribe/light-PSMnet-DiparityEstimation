import torch 
import torch.nn.functional as F
import torch.nn as nn


class SmoothL1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, disp1, disp2, disp3, target):
        loss1 = F.smooth_l1_loss(disp1, target)
        loss2 = F.smooth_l1_loss(disp2, target)
        loss3 = F.smooth_l1_loss(disp3, target)

        return loss1, loss2, loss3

def SSIM(x, y,window_size=3):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        clip_size = (window_size -1)/2

        mu_x = nn.functional.avg_pool2d(x, window_size, 1, padding=0)
        mu_y = nn.functional.avg_pool2d(y, window_size, 1, padding=0)

        x = x[:,:,clip_size:-clip_size,clip_size:-clip_size]
        y = y[:,:,clip_size:-clip_size,clip_size:-clip_size]

        sigma_x = nn.functional.avg_pool2d((x  - mu_x)**2, window_size, 1, padding=0)
        sigma_y = nn.functional.avg_pool2d((y - mu_y)** 2, window_size, 1, padding=0)

        sigma_xy = (
            nn.functional.avg_pool2d((x- mu_x) * (y-mu_y), window_size, 1, padding=0)
        )

        mu_x = mu_x[:,:,clip_size:-clip_size,clip_size:-clip_size]
        mu_y = mu_y[:,:,clip_size:-clip_size,clip_size:-clip_size]

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        loss = torch.clamp((1 - SSIM) , 0, 2)
        #save_image(loss, 'SSIM_GRAY.png')

        return  torch.mean(loss)
