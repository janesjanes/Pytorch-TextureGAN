import torch.utils.data as data

from PIL import Image
import glob
import os
import os.path as osp


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(directory):
    classes = [d for d in os.listdir(directory) if osp.isdir(os.path.join(directory, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


<<<<<<< f95da7a4b63785219f6c035fd3feb590c4fa0561
def make_dataset(dir, opt):
    # opt: 'train' or 'val'
    img = glob.glob(dir + opt + '_img/*/*.jpg')
    img = sorted(img)
    skg = glob.glob(dir + opt + '_skg/*/*.jpg')
    skg = sorted(skg)
    seg = glob.glob(dir + opt + '_seg/*/*.jpg')
    seg = sorted(seg)
    txt = glob.glob(dir + opt + '_txt/*/*.jpg')
    txt = sorted(txt)
    
    return zip(img, skg, seg ,txt)
=======
#TODO add val folder
def make_dataset(directory):
    train_img = glob.glob(osp.join(directory, 'train_img/wendy/*.jpg'))
    train_img = sorted(train_img)
    train_skg = glob.glob(osp.join(directory, 'train_skg/wendy/*.jpg'))
    train_skg = sorted(train_skg)
    train_seg = glob.glob(osp.join(directory, 'train_seg/wendy/*.jpg'))
    train_seg = sorted(train_seg)
    
    return zip(train_img, train_skg, train_seg)
>>>>>>> refactoring of code


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
<<<<<<< f95da7a4b63785219f6c035fd3feb590c4fa0561

    def __init__(self, opt, root, transform=None, target_transform=None,
                 loader=default_loader):
     
=======
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
     
        imgs = make_dataset(root)

>>>>>>> refactoring of code
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
        
        img_path, skg_path, seg_path, txt_path = self.imgs[index]
        
        img = self.loader(img_path)
        skg = self.loader(skg_path)
        seg = self.loader(seg_path)
        txt = self.loader(txt_path)
        
        if self.transform is not None:
<<<<<<< f95da7a4b63785219f6c035fd3feb590c4fa0561
            img,skg,seg,txt = self.transform([img,skg,seg,txt])
=======
            img, skg, seg = self.transform([img, skg, seg])
>>>>>>> refactoring of code
            
            
        return img, skg, seg, txt

    def __len__(self):
        return len(self.imgs)