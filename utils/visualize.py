import torch
import numpy as np
from PIL import Image
from . import transforms


def vis_patch(img, skg, texture_location, color='lab'):
    batch_size, _, _, _ = img.size()
    if color == 'lab':
        ToRGB = transforms.toRGB()
        
    elif color =='rgb':
        ToRGB = transforms.toRGB('RGB')
        img = img.cpu().numpy()
        skg = skg.cpu().numpy()
        
    img_np = ToRGB(img)
    skg_np = ToRGB(skg)
    
    vis_skg = np.copy(skg_np)
    vis_img = np.copy(img_np)
    # print np.shape(vis_skg)
    for i in range(batch_size):
        for text_loc in texture_location[i]:
            xcenter,ycenter,size = text_loc
            vis_skg[i, :, xcenter-size/2:xcenter+size/2, ycenter-size/2:ycenter+size/2] = vis_img[i, :,xcenter-size/2:xcenter+size/2,ycenter-size/2:ycenter+size/2]

    return vis_skg
    
def vis_image(img, color='lab'):
    if color == 'lab':
        ToRGB = transforms.toRGB()
        
    elif color =='rgb':
        ToRGB = transforms.toRGB('RGB')
    #print np.shape(img)   
    img_np = ToRGB(img)

    return img_np
