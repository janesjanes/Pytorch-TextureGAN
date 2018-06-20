import torch
import torch.nn as nn
import torch.optim as optim

import sys, os
import visdom
import torchvision.models as models
from torch.utils.data.sampler import SequentialSampler

from torch.utils.data import DataLoader
from dataloader.imfol import ImageFolder

from utils import transforms as custom_transforms
from models import scribbler, discriminator, texturegan, define_G, weights_init, \
    scribbler_dilate_128, FeatureExtractor, load_network
from train import train
import argparser

def get_transforms(args):
    transforms_list = [
        custom_transforms.RandomSizedCrop(args.image_size, args.resize_min, args.resize_max),
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.toTensor()
    ]
    if args.color_space == 'lab':
        transforms_list.insert(2, custom_transforms.toLAB())
    elif args.color_space == 'rgb':
        transforms_list.insert(2, custom_transforms.toRGB('RGB'))

    transforms = custom_transforms.Compose(transforms_list)
    return transforms


def get_models(args):
    sigmoid_flag = 1
    if args.gan == 'lsgan':
        sigmoid_flag = 0

    if args.model == 'scribbler':
        netG = scribbler.Scribbler(5, 3, 32)
    elif args.model == 'texturegan':
        netG = texturegan.TextureGAN(5, 3, 32)
    elif args.model == 'pix2pix':
        netG = define_G(5, 3, 32)
    elif args.model == 'scribbler_dilate_128':
        netG = scribbler_dilate_128.ScribblerDilate128(5, 3, 32)
    else:
        print(args.model + ' not support. Using Scribbler model')
        netG = scribbler.Scribbler(5, 3, 32)

    if args.color_space == 'lab':
        netD = discriminator.Discriminator(1, 32, sigmoid_flag)
        netD_local = discriminator.LocalDiscriminator(2, 32, sigmoid_flag)
    elif args.color_space == 'rgb':
        netD = discriminator.Discriminator(3, 32, sigmoid_flag)

    if args.load == -1:
        netG.apply(weights_init)
    else:
        load_network(netG, 'G', args.load_epoch, args.load, args)

    if args.load_D == -1:
        netD.apply(weights_init)
    else:
        load_network(netD, 'D', args.load_epoch, args.load_D, args)
        load_network(netD_local, 'D_local', args.load_epoch, args.load_D, args)
    return netG, netD, netD_local


def get_criterions(args):
    if args.gan == 'lsgan':
        criterion_gan = nn.MSELoss()
    elif args.gan == 'dcgan':
        criterion_gan = nn.BCELoss()
    else:
        print("Undefined GAN type. Defaulting to LSGAN")
        criterion_gan = nn.MSELoss()

    # criterion_l1 = nn.L1Loss()
    criterion_pixel_l = nn.MSELoss()
    criterion_pixel_ab = nn.MSELoss()
    criterion_style = nn.MSELoss()
    criterion_feat = nn.MSELoss()
    criterion_texturegan = nn.MSELoss()

    return criterion_gan, criterion_pixel_l, criterion_pixel_ab, criterion_style, criterion_feat, criterion_texturegan


def main(args):
    #with torch.cuda.device(args.gpu):
    layers_map = {'relu4_2': '22', 'relu2_2': '8', 'relu3_2': '13','relu1_2': '4'}

    vis = visdom.Visdom(port=args.display_port)

    loss_graph = {
        "g": [],
        "gd": [],
        "gf": [],
        "gpl": [],
        "gpab": [],
        "gs": [],
        "d": [],
        "gdl": [],
        "dl": [],
    }

    # for rgb the change is to feed 3 channels to D instead of just 1. and feed 3 channels to vgg.
    # can leave pixel separate between r and gb for now. assume user use the same weights
    transforms = get_transforms(args)

    if args.color_space == 'rgb':
        args.pixel_weight_ab = args.pixel_weight_rgb
        args.pixel_weight_l = args.pixel_weight_rgb

    rgbify = custom_transforms.toRGB()

    train_dataset = ImageFolder('train', args.data_path, transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = ImageFolder('val', args.data_path, transforms)
    indices = torch.randperm(len(val_dataset))
    val_display_size = args.batch_size
    val_display_sampler = SequentialSampler(indices[:val_display_size])
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_display_size, sampler=val_display_sampler)
    # renormalize = transforms.Normalize(mean=[+0.5+0.485, +0.5+0.456, +0.5+0.406], std=[0.229, 0.224, 0.225])

    feat_model = models.vgg19(pretrained=True)
    netG, netD, netD_local = get_models(args)

    criterion_gan, criterion_pixel_l, criterion_pixel_ab, criterion_style, criterion_feat,criterion_texturegan = get_criterions(args)


    real_label = 1
    fake_label = 0

    optimizerD = optim.Adam(netD.parameters(), lr=args.learning_rate_D, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizerD_local = optim.Adam(netD_local.parameters(), lr=args.learning_rate_D_local, betas=(0.5, 0.999))

    with torch.cuda.device(args.gpu):
        netG.cuda()
        netD.cuda()
        netD_local.cuda()
        feat_model.cuda()
        criterion_gan.cuda()
        criterion_pixel_l.cuda()
        criterion_pixel_ab.cuda()
        criterion_feat.cuda()
        criterion_texturegan.cuda()

        input_stack = torch.FloatTensor().cuda()
        target_img = torch.FloatTensor().cuda()
        target_texture = torch.FloatTensor().cuda()
        segment = torch.FloatTensor().cuda()
        label = torch.FloatTensor(args.batch_size).cuda()
        label_local = torch.FloatTensor(args.batch_size).cuda()
        extract_content = FeatureExtractor(feat_model.features, [layers_map[args.content_layers]])
        extract_style = FeatureExtractor(feat_model.features,
                                         [layers_map[x.strip()] for x in args.style_layers.split(',')])

        model = {
            "netG": netG,
            "netD": netD,
            "netD_local": netD_local,
            "criterion_gan": criterion_gan,
            "criterion_pixel_l": criterion_pixel_l,
            "criterion_pixel_ab": criterion_pixel_ab,
            "criterion_feat": criterion_feat,
            "criterion_style": criterion_style,
            "criterion_texturegan": criterion_texturegan,
            "real_label": real_label,
            "fake_label": fake_label,
            "optimizerD": optimizerD,
            "optimizerD_local": optimizerD_local,
            "optimizerG": optimizerG
        }

        for epoch in range(args.load_epoch, args.num_epoch):
            train(model, train_loader, val_loader, input_stack, target_img, target_texture,
                  segment, label, label_local,extract_content, extract_style, loss_graph, vis, epoch, args)
            #break
if __name__ == '__main__':
    args = argparser.parse_arguments()
    main(args)
