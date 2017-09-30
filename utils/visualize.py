import torch
import numpy as np
from PIL import Image
import transforms


def vis_patch(img,skg,xcenter=64,ycenter=64,size=40,color='lab'):
    if color == 'lab':
        ToRGB = transforms.toRGB()
        
    elif color =='rgb':
        ToRGB = transforms.toRGB('RGB')
        img = img.cpu().numpy()
        skg = skg.cpu().numpy()
    img_np = ToRGB(img)[0]
    skg_np = ToRGB(skg)[0]
    
    vis_skg = np.copy(skg_np)
    vis_img = np.copy(img_np)
    #print np.shape(vis_skg)
    vis_skg[:,xcenter-size/2:xcenter+size/2,ycenter-size/2:ycenter+size/2] = vis_img[:,xcenter-size/2:xcenter+size/2,ycenter-size/2:ycenter+size/2]
    
    return (vis_skg)
    
def vis_image(img,color='lab'):
    
    if color == 'lab':
        ToRGB = transforms.toRGB()
        
    elif color =='rgb':
        ToRGB = transforms.toRGB('RGB')
    #print np.shape(img)   
    img_np = ToRGB(img)[0]
   
    
    return (img_np)