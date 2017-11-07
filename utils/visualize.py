import torch
import numpy as np
from PIL import Image
from . import transforms


def vis_patch(img, skg, texture_location, color='lab'):
    batch_size, _, _, _ = img.size()
    if torch.cuda.is_available():
        img = img.cpu()
        skg = skg.cpu()

    img = img.numpy()
    skg = skg.numpy()

    if color == 'lab':
        ToRGB = transforms.toRGB()
        
    elif color =='rgb':
        ToRGB = transforms.toRGB('RGB')
        
    img_np = ToRGB(img)
    skg_np = ToRGB(skg)

    vis_skg = np.copy(skg_np)
    vis_img = np.copy(img_np)

    # print np.shape(vis_skg)
    for i in range(batch_size):
        for text_loc in texture_location[i]:
            xcenter, ycenter, size = text_loc
            xcenter = max(xcenter-int(size/2),0) + int(size/2)
            ycenter = max(ycenter-int(size/2),0) + int(size/2)
            vis_skg[
                i, :,
                int(xcenter-size/2):int(xcenter+size/2),
                int(ycenter-size/2):int(ycenter+size/2)
            ] = vis_img[
                    i, :,
                    int(xcenter-size/2):int(xcenter+size/2),
                    int(ycenter-size/2):int(ycenter+size/2)
                ]

    return vis_skg
    
def vis_image(img, color='lab'):
    if torch.cuda.is_available():
        img = img.cpu()

    img = img.numpy()

    if color == 'lab':
        ToRGB = transforms.toRGB()
    elif color =='rgb':
        ToRGB = transforms.toRGB('RGB')

    # print np.shape(img)
    img_np = ToRGB(img)

    return img_np
