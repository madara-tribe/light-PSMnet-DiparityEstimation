import os
import numpy as np
import cv2

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

def disp2np(x):
    vmax = np.percentile(x, 95)
    normalizer = mpl.colors.Normalize(vmin=x.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_tr = (mapper.to_rgba(x)[:, :, :3] * 255).astype(np.uint8)
    tr = pil.fromarray(colormapped_tr)
    return np.array(tr, dtype=np.uint8)

def save_depth(direction, idx, disp_resized, target, inp, output_directory="depth"):
    #print("disp_resized, target, inp", disp_resized.shape, target.shape, inp.shape)
    os.makedirs(output_directory, exist_ok=True)
    disp_resized_np = disp_resized[0].cpu().detach().numpy().copy()
    disp = disp2np(disp_resized_np)
    targets = target[0].cpu().detach().numpy().copy()
    tr_disp = disp2np(targets)
    pred = inp[0].cpu().detach().numpy().copy()
    pred = (pred * 255).transpose(1, 2, 0).astype(np.float32)
    cimg = np.hstack([disp, tr_disp, pred])
    name_inp_im = os.path.join(output_directory, "{}_{}_inp.jpeg".format(direction, idx))
    cv2.imwrite(name_inp_im, cimg)


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

