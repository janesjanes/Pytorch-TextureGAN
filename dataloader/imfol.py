import torch.utils.data as data

from PIL import Image
import glob
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, opt):
    # opt: 'train' or 'val'
    img = glob.glob(dir + opt + '_img/*/*.jpg')
    img = sorted(img)
    skg = glob.glob(dir + opt + '_skg/*/*.jpg')
    skg = sorted(skg)
    seg = glob.glob(dir + opt + '_seg/*/*.jpg')
    seg = sorted(seg)
    
    return zip(img, skg, seg)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
   
    return pil_loader(path)


class ImageFolder(data.Dataset):

    def __init__(self, opt, root, transform=None, target_transform=None,
                 loader=default_loader):
     
        self.root = root
        self.imgs = make_dataset(root, opt)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        
        img_path, skg_path, seg_path = self.imgs[index]
        
        img = self.loader(img_path)
        skg = self.loader(skg_path)
        seg = self.loader(seg_path)
        
        
        if self.transform is not None:
            img,skg,seg = self.transform([img,skg,seg])
            
            
        return img, skg, seg

    def __len__(self):
        return len(self.imgs)