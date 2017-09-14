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


def make_dataset(dir):
    train_img=glob.glob(dir+'train_img/wendy/*.jpg')
    train_img=sorted(train_img)
    train_skg=glob.glob(dir+'train_skg/wendy/*.jpg')
    train_skg=sorted(train_skg)
    train_seg=glob.glob(dir+'train_seg/wendy/*.jpg')
    train_seg=sorted(train_seg)
    
    return zip(train_img,train_skg,train_seg)


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

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
     
        imgs = make_dataset(root)
        

        self.root = root
        self.imgs = imgs
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
            #TODO transform need to be applied to all three in exactly the same ways.
            img = self.transform(img)
            skg = self.transform(skg)
            seg = self.transform(seg)
            
        return img, skg, seg

    def __len__(self):
        return len(self.imgs)