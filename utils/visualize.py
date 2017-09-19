import torch
import numpy as np
from PIL import Image
import transforms

def vis_patch(img,skg,xcenter=64,ycenter=64,size=40):
    ToRGB = transforms.toRGB()
    img_np = ToRGB(img[0])
    skg_np = ToRGB(skg[0])
    
    vis_img = np.copy(skg_np)
    vis_img[xcenter-size/2:xcenter+size/2,ycenter-size/2:ycenter+size/2,:] = img_np[xcenter-size/2:xcenter+size/2,ycenter-size/2:ycenter+size/2,:]
    
    return vis_img
    
def vis_image(img):
    ToRGB = transforms.toRGB()
    img_np = ToRGB(img[0])
   
    
    return img_np