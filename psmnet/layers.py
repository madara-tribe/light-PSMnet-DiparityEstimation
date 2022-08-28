import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttntion(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation="relu"):
        super(SelfAttntion,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class Squeeze_excitation_layer(nn.Module):
    def __init__(self, filters, se_ratio=4):
        super(Squeeze_excitation_layer, self).__init__()
        reduction = filters // se_ratio
        self.se = nn.Sequential(nn.Conv2d(filters, reduction, kernel_size=1, bias=True),
                                nn.SiLU(),
                                nn.Conv2d(reduction, filters, kernel_size=1, bias=True),
                                nn.Sigmoid())
    def forward(self, inputs):
        x = torch.mean(inputs, [2, 3], keepdim=True)
        x = self.se(x)
        return torch.multiply(inputs, x)

