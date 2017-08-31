from skimage import color
from PIL import Image
import numpy as np
import torch

class toLAB(object):
    """
    Transform to convert loaded into LAB space. 
    """
    
    def __init__(self):
        self.space = 'LAB'
        
    def __call__(self, image):
        lab_image = color.rgb2lab(np.array(image)/255.0)
        return lab_image
    
class toRGB(object):
    """
    Transform to convert loaded into RGB color space. 
    """
    
    def __init__(self):
        self.space = 'RGB'
        
    def __call__(self, img):
        npimg = np.transpose(img.numpy(), (1, 2, 0))
        rgb_img = color.lab2rgb(np.array(npimg))
        return rgb_img
    
class toTensor(object):
    
    def __init__(self):
        self.space = 'RGB'
        
    def __call__(self, pic):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img
        
