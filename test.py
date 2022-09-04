import torch
from psmnet.PSMnetPlus import PSMNetPlus

torch.manual_seed(2.0)
disp_max = 16
model = PSMNetPlus(disp_max).cuda()
left = torch.randn(2, 3, 256, 256).cuda()
right = torch.randn(2, 3, 256, 256).cuda()
print(left[:, :, 0, 0])

out1, out2, out3 = model(left, right)
print(out2[0, :3, :3])
