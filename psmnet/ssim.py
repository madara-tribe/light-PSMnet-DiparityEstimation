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


