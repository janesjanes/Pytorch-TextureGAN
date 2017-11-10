import torch.utils.data as data

from PIL import Image
import glob
import os
import os.path as osp
import random


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


def make_dataset(directory, opt, erode_seg=True):
    # opt: 'train' or 'val'
    img = glob.glob(osp.join(directory, opt + '_img/*/*.jpg'))
    img = sorted(img)
    skg = glob.glob(osp.join(directory, opt + '_skg/*/*.jpg'))
    skg = sorted(skg)
    seg = glob.glob(osp.join(directory, opt + '_seg/*/*.jpg'))
    seg = sorted(seg)
    txt = glob.glob(osp.join(directory, opt + '_txt/*/*.jpg'))
    #txt = glob.glob(osp.join(directory, opt + '_dtd_txt/*/*.jpg'))
    extended_txt = []
    #import pdb; pdb.set_trace()
    for i in range(len(skg)):
        extended_txt.append(txt[i%len(txt)])
    random.shuffle(extended_txt)
    

    if erode_seg:
        eroded_seg = glob.glob(osp.join(directory, 'eroded_' + opt + '_seg/*/*.jpg'))
        eroded_seg = sorted(eroded_seg)
        return list(zip(img, skg, seg , eroded_seg, extended_txt))
    else:
        return list(zip(img, skg, seg, extended_txt))


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
                 loader=default_loader, erode_seg=True):
     
        self.root = root
        self.imgs = make_dataset(root, opt, erode_seg=erode_seg)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.erode_seg = erode_seg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        if self.erode_seg:
            img_path, skg_path, seg_path, eroded_seg_path, txt_path = self.imgs[index]
        else:
            img_path, skg_path, seg_path, txt_path = self.imgs[index]
        
        img = self.loader(img_path)
        skg = self.loader(skg_path)
        seg = self.loader(seg_path)
        txt = self.loader(txt_path)

        if self.erode_seg:
            eroded_seg = self.loader(eroded_seg_path)
        else:
            eroded_seg = None

        if self.transform is not None:
            if self.erode_seg:
                img, skg, seg, eroded_seg, txt = self.transform([img, skg, seg, eroded_seg, txt])
            else:
                img, skg, seg, txt = self.transform([img, skg, seg, txt])
                eroded_seg = seg

        return img, skg, seg, eroded_seg, txt


    def __len__(self):
        return len(self.imgs)
