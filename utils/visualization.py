import math
import cv2
import numpy as np
# torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use('Agg')
def draw_attn(txt_attn, real_fake = 'real',slot_iter=0,mask_iter=0, sample_num= 0, color_map = ['blue','red']):
    # cmap = LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])
    cmap = LinearSegmentedColormap.from_list('blue_red', color_map)
    plt.imshow(txt_attn, cmap=cmap, interpolation='none')
    # plt.colorbar()
    plt.axis('off')
    plt.savefig(f'./results/{sample_num}_sample_{real_fake}_mask{mask_iter}_iter{slot_iter}.png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

def visualize_attn(attn, attn_shape,iter):
    # attn_vis = attn.detach().cpu().numpy().reshape(b, int(math.sqrt(n)), int(math.sqrt(n)), -1)
    attn_vis = attn.detach().cpu().numpy().reshape(attn_shape)

    for n, sample in enumerate(attn_vis):
        fake_txt_attn = sample[..., 0]
        real_txt_attn = sample[..., 1]
        for i in range(attn_vis.shape[-1]):
            if i <attn_vis.shape[-1]//2 :
                draw_attn(sample[...,i],real_fake='fake',slot_iter=iter,mask_iter=i, sample_num=n)
                # draw_attn(sample[..., i], real_fake='fake', slot_iter=iter, mask_iter=i, sample_num=n, color_map=['black','red'])
                # draw_attn(sample[..., i], real_fake='fake', slot_iter=iter, mask_iter=i, sample_num=n, color_map=['black','yellow'])
            elif i >= attn_vis.shape[-1] // 2:
                draw_attn(sample[..., i], real_fake='real',slot_iter=iter,mask_iter=i, sample_num=n)
                # draw_attn(sample[..., i], real_fake='real',slot_iter=iter,mask_iter=i, sample_num=n, color_map=['black','red'])
                # draw_attn(sample[..., i], real_fake='real',slot_iter=iter,mask_iter=i, sample_num=n, color_map=['black','yellow'])