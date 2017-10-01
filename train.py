#from __future__ import print_function
#from __future__ import absolute_import
#from __future__ import division


import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dataset
import visdom
import sys,argparse,os

from models import scribbler, discriminator, texturegan, localDiscriminator
import torch.optim as optim

from skimage import color
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import visualize
import torch.nn as nn
from torch.autograd import Variable
from IPython.display import display
import torchvision.models as models
from dataloader import imfol
from torch.utils.data.sampler import SequentialSampler

from utils.visualize import vis_patch, vis_image
        
from torch.utils.data import DataLoader
from dataloader.imfol import ImageFolder, make_dataset
from utils import transforms as custom_trans
import torchvision.transforms as tforms
import utils.transforms as utforms

from networks import define_G, weights_init
from models import scribbler 
import visdom


def main(args):
    
    

    with torch.cuda.device(args.gpu):
        layers_map = {'relu4_2':'22','relu2_2':'8', 'relu3_2':'13'}

        vis=visdom.Visdom(port=args.display_port)

        Loss_g_graph=[]
        Loss_gd_graph=[]
        Loss_gf_graph = []
        Loss_gpl_graph = []
        Loss_gpab_graph = []    
        Loss_gs_graph = []
        Loss_d_graph=[]
        #for rgb the change is to feed 3 channels to D instead of just 1. and feed 3 channels to vgg. 
        #can leave pixel separate between r and gb for now. assume user use the same weights
        if args.color_space == 'lab':
            transform=custom_trans.Compose([custom_trans.RandomSizedCrop(args.image_size,args.resize_min,args.resize_max),
                                     custom_trans.RandomHorizontalFlip() ,custom_trans.toLAB(), custom_trans.toTensor()])
        elif args.color_space == 'rgb':
            transform=custom_trans.Compose([custom_trans.RandomSizedCrop(args.image_size,args.resize_min,args.resize_max),
                                     custom_trans.RandomHorizontalFlip() ,custom_trans.toRGB('RGB'), custom_trans.toTensor()])
            args.pixel_weight_ab = args.pixel_weight_rgb
            args.pixel_weight_l = args.pixel_weight_rgb
        rgbify = custom_trans.toRGB()
        trainDset = ImageFolder('train', args.data_path, transform)
        trainLoader = DataLoader(dataset=trainDset, batch_size=args.batch_size, shuffle=True)

        valDset = ImageFolder('val', args.data_path, transform)
        indices = torch.randperm(len(valDset))
        val_display_size = args.batch_size
        val_display_sampler = SequentialSampler(indices[:val_display_size])
        valLoader = DataLoader(dataset=valDset, batch_size=val_display_size,sampler=val_display_sampler)
       # renormalize = transforms.Normalize(mean=[+0.5+0.485, +0.5+0.456, +0.5+0.406], std=[0.229, 0.224, 0.225])

        sigmoid_flag = 1
        if args.gan =='lsgan':
            sigmoid_flag = 0 

        if args.model=='scribbler':
            netG=scribbler.Scribbler(5,3,32)
        elif args.model == 'texturegan':
             netG = texturegan.TextureGAN(5, 3, 32)    
        elif args.model=='pix2pix':
            netG=define_G(5,3,32)
        else:
            print(args.model+ ' not support. Using pix2pix model')
            netG=define_G(5,3,32)
        if args.color_space =='lab':
            netD=discriminator.Discriminator(1,32,sigmoid_flag) 
        elif args.color_space =='rgb':
            netD=discriminator.Discriminator(3,32,sigmoid_flag) 
        feat_model=models.vgg19(pretrained=True)
        if args.load == -1:
            netG.apply(weights_init)
        else:

            load_network(netG,'G',args.load,args.load_dir)
            print('Loaded G from itr:' + str(args.load))
        if args.load_D == -1:
            netD.apply(weights_init)  
        else:
            load_network(netD,'D',args.load_D,args.load_dir)
            print('Loaded D from itr:' + str(args.load_D))

        if args.gan =='lsgan':
            criterion_gan = nn.MSELoss()
        elif args.gan =='dcgan':
            criterion_gan = nn.BCELoss()

        #criterion_l1 = nn.L1Loss()
        criterion_pixel_l = nn.L1Loss()
        criterion_pixel_ab = nn.L1Loss()
        criterion_style = nn.L1Loss()
        criterion_feat = nn.L1Loss()

        input_stack = torch.FloatTensor()
        target_img = torch.FloatTensor()
        segment = torch.FloatTensor()


        label = torch.FloatTensor(args.batch_size)
        real_label = 1
        fake_label = 0

        optimizerD = optim.Adam(netD.parameters(), lr=args.learning_rate_D, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        netG.cuda()
        netD.cuda()
        feat_model.cuda()
        criterion_gan.cuda()
        criterion_pixel_l.cuda()
        criterion_pixel_ab.cuda()
        criterion_feat.cuda()
        input_stack, target_img, segment, label = input_stack.cuda(), target_img.cuda(),segment.cuda(), label.cuda()

        Extract_content = FeatureExtractor(feat_model.features, [layers_map[args.content_layers]])
        Extract_style = FeatureExtractor(feat_model.features, [layers_map[x.strip()] for x in args.style_layers.split(',')])
        for epoch in range(args.num_epoch):
            for i, data in enumerate(trainLoader, 0):


                #Detach is apparently just creating new Variable with cut off reference to previous node, so shouldn't effect the original 
                #But just in case, let's do G first so that detaching G during D update don't do anything weird
                ############################
                # (1) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()

                img, skg,seg = data #LAB with negeative value
                #output img/skg/seg rgb between 0-1
                #output img/skg/seg lab between 0-100, -128-128 
                if args.color_space =='lab':
                    img=utforms.normalize_lab(img)
                    skg=utforms.normalize_lab(skg)
                elif args.color_space =='rgb':
                    img=utforms.normalize_rgb(img)
                    skg=utforms.normalize_rgb(skg)  

                inp,_ = gen_input_rand(img,skg,args.crop_size_min,args.crop_size_max)


                img=img.cuda()
                skg=skg.cuda()
                seg=seg.cuda()

                inp = inp.cuda()

                input_stack.resize_as_(inp.float()).copy_(inp)
                target_img.resize_as_(img.float()).copy_(img)
                segment.resize_as_(seg.float()).copy_(seg)

                inputv = Variable(input_stack)
                targetv = Variable(target_img)

                outputG = netG(inputv)

                outputl,outputa,outputb=torch.chunk((outputG),3,dim=1)

                targetl,targeta,targetb = torch.chunk(targetv,3,dim=1)
                outputab = torch.cat((outputa,outputb),1)
                targetab = torch.cat((targeta,targetb),1)

                #TODO renormalize with the right mean (but shouldn't matter much, it's around 0.5 anyway)
                if args.color_space =='lab':
                    outputlll= (torch.cat((outputl,outputl,outputl),1))
                    targetlll = (torch.cat((targetl,targetl,targetl),1))
                elif args.color_space =='rgb':
                    outputlll= outputG#(torch.cat((outputl,outputl,outputl),1))
                    targetlll = targetv#(torch.cat((targetl,targetl,targetl),1))                

                ##################Pixel L Loss############################
                err_pixel_l = args.pixel_weight_l*criterion_pixel_l(outputl,targetl)

                ##################Pixel ab Loss############################
                err_pixel_ab = args.pixel_weight_ab*criterion_pixel_ab(outputab,targetab)


                ##################feature Loss############################
                out_feat = Extract_content(outputlll)[0]

                gt_feat = Extract_content(targetlll)[0]
                err_feat = args.feature_weight*criterion_feat(out_feat,gt_feat.detach())   


                ##################style Loss############################


                if args.local_texture_size == -1: #global
                    output_feat_ = Extract_style(outputlll)
                    target_feat_ = Extract_style(targetlll)
                else:
                    patchsize = args.local_texture_size
                    x = int( rand_between(patchsize,args.image_size-patchsize))
                    y = int( rand_between(patchsize,args.image_size-patchsize))

                    texture_patch = outputlll[:,:,x:(x+patchsize),y:(y+patchsize)]
                    gt_texture_patch = targetlll[:,:,x:(x+patchsize),y:(y+patchsize)]
                    output_feat_ = Extract_style(texture_patch)
                    target_feat_ = Extract_style(gt_texture_patch)




                gram = GramMatrix()

                err_style = 0
                for m in range(len(output_feat_)): 
                    gram_y = gram(output_feat_[m])
                    gram_s = gram(target_feat_[m])

                    err_style += args.style_weight * criterion_style(gram_y, gram_s.detach())



                ##################D Loss############################
                netD.zero_grad()
                label_ = Variable(label)
                if args.color_space =='lab':
                    outputD = netD(outputl)
                elif args.color_space =='rgb':
                    outputD = netD(outputG)
                #D_G_z2 = outputD.data.mean()

                label.resize_(outputD.data.size())
                labelv = Variable(label.fill_(real_label))

                err_gan = args.discriminator_weight*criterion_gan(outputD, labelv)

                ####################################
                err_G = err_pixel_l+err_pixel_ab + err_gan + err_feat + err_style
                err_G.backward()

                optimizerG.step() 

                Loss_g_graph.append(err_G.data[0])
                Loss_gpl_graph.append(err_pixel_l.data[0])
                Loss_gpab_graph.append(err_pixel_ab.data[0])
                Loss_gd_graph.append(err_gan.data[0])
                Loss_gf_graph.append(err_feat.data[0])
                Loss_gs_graph.append(err_style.data[0])
                #plt.imshow(vis_image(inputv.data.double().cpu()))

                print 'G:', i, err_G.data[0]            


                ############################
                # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real

                netD.zero_grad()

                labelv = Variable(label)
                if args.color_space =='lab':
                    outputD = netD(targetl)
                elif args.color_space =='rgb':
                    outputD = netD(targetv)

                label.resize_(outputD.data.size())
                labelv = Variable(label.fill_(real_label))

                errD_real = criterion_gan(outputD, labelv)
                errD_real.backward()

                score = Variable(torch.ones(args.batch_size))
                _,cd,wd,hd = outputD.size()
                D_output_size = cd*wd*hd

                clamped_output_D = outputD.clamp(0,1)
                clamped_output_D = torch.round(clamped_output_D)
                for acc_i in range(args.batch_size):
                    score[acc_i] = torch.sum(clamped_output_D[acc_i])/D_output_size

                real_acc = torch.mean(score)

                ##################################
                #TODO add threshold to stop updating D
                if args.color_space =='lab':
                    outputD = netD(outputl.detach())
                elif args.color_space =='rgb':
                    outputD = netD(outputG.detach())
                label.resize_(outputD.data.size())
                labelv = Variable(label.fill_(fake_label))

                errD_fake = criterion_gan(outputD, labelv)
                errD_fake.backward()
                score = Variable(torch.ones(args.batch_size))
                _,cd,wd,hd = outputD.size()
                D_output_size = cd*wd*hd

                clamped_output_D = outputD.clamp(0,1)
                clamped_output_D = torch.round(clamped_output_D)
                for acc_i in range(args.batch_size):
                    score[acc_i] = torch.sum(clamped_output_D[acc_i])/D_output_size

                fake_acc = torch.mean(1-score)

                D_acc = ( real_acc + fake_acc )/2

                if D_acc.data[0] <args.threshold_D_max:
                    #D_G_z1 = output.data.mean()
                    errD = errD_real + errD_fake
                    Loss_d_graph.append(errD.data[0])
                    optimizerD.step()
                else:
                    Loss_d_graph.append(0)
                #TODO add discriminator accuracy



                print 'D:', 'real_acc', "%.2f" %real_acc.data[0], 'fake_acc', "%.2f" %fake_acc.data[0] ,'D_acc',D_acc.data[0]
                if(i%args.save_every==0):
                    save_network(netG,'G',i,args.gpu,args.save_dir)
                    save_network(netD,'D',i,args.gpu,args.save_dir)



                if(i%args.visualize_every==0):
                    imgs = []
                    for ii, data in enumerate(valLoader, 0):

                        img, skg, seg = data #LAB with negeative value
                        #this is in LAB value 0/100, -128/128 etc
                        img=utforms.normalize_lab(img)
                        skg=utforms.normalize_lab(skg)
                        #norm to 0-1 minus mean

                        inp,texture_loc = gen_input_rand(img,skg,args.crop_size_min,args.crop_size_max)

                        img=img.cuda()
                        skg=skg.cuda()
                        seg=seg.cuda()

                        inp = inp.cuda()

                        input_stack.resize_as_(inp.float()).copy_(inp)
                        target_img.resize_as_(img.float()).copy_(img)
                        segment.resize_as_(seg.float()).copy_(seg)

                        inputv = Variable(input_stack)
                        targetv = Variable(target_img)

                        outputG = netG(inputv)

                        #segment_img=vis_image((seg.cpu()))
                        #segment_img=(segment_img*255).astype('uint8')
                        #segment_img=np.transpose(segment_img,(2,0,1))
                        #imgs.append(segment_img)

                        #inp_img= vis_patch(utforms.denormalize_lab(img.cpu()), utforms.denormalize_lab(skg.cpu()), xcenter, ycenter, crop_size)
                        #inp_img=(inp_img*255).astype('uint8')
                        #inp_img=np.transpose(inp_img,(2,0,1))
                        #imgs.append(inp_img)

                        #out_img= vis_image(utforms.denormalize_lab(outputG.data.double().cpu()))
                        #out_img=(out_img*255).astype('uint8')
                        #out_img=np.transpose(out_img,(2,0,1))   
                        #imgs.append(out_img)

                        #tar_img=vis_image(utforms.denormalize_lab(img.cpu()))
                        #tar_img=(tar_img*255).astype('uint8')
                        #tar_img=np.transpose(tar_img,(2,0,1))
                        #imgs.append(tar_img)

                    if args.color_space == 'lab':
                        out_img=vis_image(utforms.denormalize_lab(outputG.data.double().cpu()),args.color_space)
                        inp_img=vis_patch(utforms.denormalize_lab(img.cpu()),utforms.denormalize_lab(skg.cpu()),texture_loc,args.color_space)
                        tar_img=vis_image(utforms.denormalize_lab(img.cpu()),args.color_space)
                    elif args.color_space =='rgb':
                        out_img=vis_image(utforms.denormalize_rgb(outputG.data.double().cpu()),args.color_space)
                        inp_img=vis_patch(utforms.denormalize_rgb(img.cpu()),utforms.denormalize_rgb(skg.cpu()),texture_loc,args.color_space)
                        tar_img=vis_image(utforms.denormalize_rgb(img.cpu()),args.color_space)                    
                    out_img=[x*255 for x in out_img]#(out_img*255)#.astype('uint8')
                    #out_img=np.transpose(out_img,(2,0,1))

                    inp_img=[x*255 for x in inp_img]#(inp_img*255)#.astype('uint8')
                    #inp_img=np.transpose(inp_img,(2,0,1))


                    tar_img=[x*255 for x in tar_img]#(tar_img*255)#.astype('uint8')
                    #tar_img=np.transpose(tar_img,(2,0,1))

                    segment_img=vis_image((seg.cpu()),args.color_space)
                    segment_img=[x*255 for x in segment_img]#segment_img=(segment_img*255)#.astype('uint8')
                    #segment_img=np.transpose(segment_img,(2,0,1))

                    for i_ in range(len(out_img)):
                        imgs.append(segment_img[i_])
                        imgs.append(inp_img[i_])
                        imgs.append(out_img[i_])
                        imgs.append(tar_img[i_])

                    vis.images(imgs,win='output',opts=dict(title='Output images'))
                    #vis.image(inp_img,win='input',opts=dict(title='input'))  
                    #vis.image(tar_img,win='target',opts=dict(title='target'))
                    #vis.image(segment_img,win='segment',opts=dict(title='segment'))
                    vis.line(np.array(Loss_gs_graph),win='gs',opts=dict(title='G-Style Loss'))
                    vis.line(np.array(Loss_g_graph),win='g',opts=dict(title='G Total Loss'))
                    vis.line(np.array(Loss_gd_graph),win='gd',opts=dict(title='G-Discriminator Loss'))
                    vis.line(np.array(Loss_gf_graph),win='gf',opts=dict(title='G-Feature Loss'))
                    vis.line(np.array(Loss_gpl_graph),win='gpl',opts=dict(title='G-Pixel Loss-L'))
                    vis.line(np.array(Loss_gpab_graph),win='gpab',opts=dict(title='G-Pixel Loss-AB'))
                    vis.line(np.array(Loss_d_graph),win='d',opts=dict(title='D Loss'))





                

#all in one place funcs, need to organize these:
def rand_between(a,b):
    return a + torch.round(torch.rand(1)*(b-a))[0]

def gen_input(img,skg,xcenter=64,ycenter=64,size=40):
    #generate input skg with random patch from img
    #input img,skg [bsx3xwxh], xcenter,ycenter, size 
    #output bsx5xwxh

    w,h = img.size()[1:3]
    #print w,h
    xstart = max(xcenter-size/2,0)
    ystart = max(ycenter-size/2,0)
    xend = min(xcenter + size/2,w)
    yend = min(ycenter + size/2,h)


    input_texture = torch.ones(img.size())*(1)
    input_sketch = skg[0:1,:,:] #L channel from skg
    input_mask = torch.ones(input_sketch.size())*(-1)
    
    #print xstart, xend, ystart, yend

    input_mask[:,xstart:xend,ystart:yend] = 1
    input_texture[:,xstart:xend,ystart:yend] = img[:,xstart:xend,ystart:yend].clone()
    #print input_mask.size()
    #print input_texture.size()
    #print input_mask.size()
    return torch.cat((input_sketch.cpu().float(),input_texture.float(),input_mask),0)
    #return input_mask,input_texture,input_sketch
def gen_input_rand(img,skg,size_min=40,size_max=60):
    #generate input skg with random patch from img
    #input img,skg [bsx3xwxh], xcenter,ycenter, size 
    #output bsx5xwxh

    bs,c,w,h = img.size()
    results = torch.Tensor(bs,5,w,h)
    text_info = [] 
    
    for i in range(bs):
            crop_size = int( rand_between(size_min, size_max))
            xcenter = int( rand_between(crop_size/2,w-crop_size/2))
            ycenter = int( rand_between(crop_size/2,h-crop_size/2))   
            text_info.append([xcenter,ycenter,crop_size])
            #print xcenter, ycenter
            #print i, xcenter, ycenter
            results[i,:,:,:] = gen_input(img[i],skg[i],xcenter,ycenter,crop_size)
    return results,text_info


class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a , b, c * d)  # resise F_XL into \hat F_XL

        G = torch.bmm(features, features.transpose(1,2))  # compute the gram product

        # normalize the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div( b * c * d)
    
class FeatureExtractor(nn.Module):
    # Extract features from intermediate layers of a network

    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.extracted_layers=extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]
    
def save_network(model, network_label, epoch_label, gpu_id, save_dir):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(model.cpu().state_dict(), save_path)
    model.cuda(device_id=gpu_id)
def load_network(model, network_label, epoch_label,save_dir):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    model.load_state_dict(torch.load(save_path))    
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
###############added options#######################################
    parser.add_argument('-learning_rate', default=1e-5, type=float,
                    help='Learning rate for the generator')
    parser.add_argument('-learning_rate_D',  default=1e-4,type=float,
                    help='Learning rate for the discriminator')    
    
    parser.add_argument('-gan', default='lsgan',type=str,choices=['dcgan', 'lsgan'],
                    help='dcgan|lsgan') #todo wgan/improved wgan    
    
    parser.add_argument('-model', default='pix2pix',type=str,choices=['scribbler', 'pix2pix', 'texturegan'],
                    help='scribbler|pix2pix|texturegan')
    
    parser.add_argument('-num_epoch',  default=1,type=int,
                    help='texture|scribbler')   
    
    parser.add_argument('-visualize_every',  default=10,type=int,
                    help='no. iteration to visualize the results')      

    #all the weights ratio, might wanna make them sum to one
    parser.add_argument('-feature_weight', default=100,type=float,
                       help='weight ratio for feature loss')
    parser.add_argument('-pixel_weight_l', default=400,type=float,
                       help='weight ratio for pixel loss for l channel')
    parser.add_argument('-pixel_weight_ab', default=800,type=float,
                   help='weight ratio for pixel loss for ab channel')
    parser.add_argument('-pixel_weight_rgb', default=800,type=float,
                   help='weight ratio for pixel loss for ab channel')
    
    parser.add_argument('-discriminator_weight', default=2e1,type=float,
                   help='weight ratio for the discriminator loss')
    parser.add_argument('-style_weight', default = 1, type=float, 
                        help='weight ratio for the texture loss')


    parser.add_argument('-gpu', default=1,type=int,
                   help='id of gpu to use') #TODO support cpu

    parser.add_argument('-display_port', default=7780,type=int,
               help='port for displaying on visdom (need to match with visdom currently open port)')

    parser.add_argument('-data_path', default='/home/psangkloy3/training_catdog/',type=str,
                   help='path to the data directory, expect train_skg, train_img, val_skg, val_img')

    parser.add_argument('-save_dir', default='/home/psangkloy3/texturegan/save_dir_scribbler',type=str,
                   help='path to save the model')
    
    parser.add_argument('-load_dir', default='/home/psangkloy3/texturegan/catdog/',type=str,
                   help='path to save the model')
    
    parser.add_argument('-save_every',  default=1000,type=int,
                    help='no. iteration to save the models')
    
    parser.add_argument('-load', default=14000,type=int,
                   help='load generator and discrminator from iteration n')
    parser.add_argument('-load_D', default=-1,type=float,
                   help='load discriminator from iteration n, priority over load')
    
    parser.add_argument('-image_size',default=128,type=int,
                    help='Training images size, after cropping')        
    parser.add_argument('-resize_max',  default=1,type=int,
                    help='max resize, ratio of the original image, max value is 1')        
    parser.add_argument('-resize_min',  default=0.6,type=int,
                    help='min resize, ratio of the original image, min value 0')   
    parser.add_argument('-crop_size_min',default=20,type=int,
                    help='minumum texture patch size')   
    parser.add_argument('-crop_size_max',default=40,type=int,
                    help='max texture patch size')  
    
    parser.add_argument('-batch_size', default=8)     
    
    parser.add_argument('-local_texture_size', default=50,type=int,
                   help='use local texture loss instead of global, set -1 to use global')
    parser.add_argument('-color_space',  default='lab',type=str,choices=['lab','rgb'],
                help='lab|rgb')
    
    parser.add_argument('-threshold_D_max',  default=0.8,type=int,
                    help='stop updating D when accuracy is over max')
    
    parser.add_argument('-content_layers',  default='relu4_2',type=str,
                    help='Layer to attach content loss.')
    parser.add_argument('-style_layers',  default='relu3_2, relu4_2',type=str,
    help='Layer to attach content loss.')   
    
############################################################################
############################################################################
############TODO: TO ADD#################################################################
    parser.add_argument('-tv_weight', default=1,type=float,
                   help='weight ratio for total variation loss')

    
    parser.add_argument('-threshold_D_min',  default=0.3,type=int,
                    help='stop updating G when accuracy is below min')

    
    parser.add_argument('-mode',  default='texture',type=str,choices=['texture','scribbler'],
                    help='texture|scribbler') 
    
   
    parser.add_argument('-crop',  default='random',type=str,choices=['random','center'],
                    help='random|center')
    
    parser.add_argument('-contrast',  default=True,type=bool,
                    help='randomly adjusting contrast on sketch')
    
    parser.add_argument('-occlude', default=False,type=bool,
                       help='randomly occlude part of the sketch')
    
    
    parser.add_argument('-checkpoints_path', default='data/',type=str,
                   help='output directory for results and models')
    

    
    parser.add_argument('-noise_gen', default=False,type=bool,
                   help='whether or not to inject noise into the network')
    
    
    parser.add_argument('-absolute_load', default='',type=str,
                   help='load saved generator model from absolute location')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    