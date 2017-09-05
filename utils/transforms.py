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
    """Transforms a Numpy image to torch tensor"""
    
    def __init__(self):
        self.space = 'RGB'
        
    def __call__(self, pic):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img

def normalize_lab(lab_img):
    """
    Normalizes the LAB image to lie in range 0-1
    
    Args:
    lab_img : torch.Tensor img in lab space
    
    Returns:
    lab_img : torch.Tensor Normalized lab_img 
    """
    mean = torch.zeros(lab_img.size())
    stds = torch.zeros(lab_img.size())
    
    mean[:,0,:,:] = 50
    mean[:,1,:,:] = 0
    mean[:,2,:,:] = 0
    
    stds[:,0,:,:] = 60
    stds[:,1,:,:] = 160
    stds[:,2,:,:] = 160
    
    return (lab_img.double() - mean.double())/stds.double()

   
    
def denormalize_lab(lab_img):
    """
    Normalizes the LAB image to lie in range 0-1
    
    Args:
    lab_img : torch.Tensor img in lab space
    
    Returns:
    lab_img : torch.Tensor Normalized lab_img 
    """
    mean = torch.zeros(lab_img.size())
    stds = torch.zeros(lab_img.size())
    
    mean[:,0,:,:] = 50
    mean[:,1,:,:] = 0
    mean[:,2,:,:] = 0
    
    stds[:,0,:,:] = 60
    stds[:,1,:,:] = 160
    stds[:,2,:,:] = 160
    
    return lab_img.double() *stds.double() + mean.double()
   
    
