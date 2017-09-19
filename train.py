#from __future__ import print_function
#from __future__ import absolute_import
#from __future__ import division


import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dataset
import visdom
import sys,argparse,os

from models import scribbler, discriminator
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

#TODO: visdom show the whole batch and on test set
#TODO: index to name mapping for vgg layers
#TODO: rgb/lab option

    with torch.cuda.device(args.gpu):

        vis=visdom.Visdom(port=args.display_port)

        Loss_g_graph=[]
        Loss_gd_graph=[]
        Loss_gf_graph = []
        Loss_gpl_graph = []
        Loss_gpab_graph = []    
        Loss_gs_graph = []
        Loss_d_graph=[]

        ts=custom_trans.Compose([custom_trans.RandomSizedCrop(args.image_size,args.resize_min,args.resize_max),custom_trans.RandomHorizontalFlip() ,custom_trans.toLAB(), custom_trans.toTensor()])
        rgbify = custom_trans.toRGB()
        dset = ImageFolder(args.data_path,ts)
        dataloader=DataLoader(dataset=dset, batch_size=args.batch_size, shuffle=True)

       # renormalize = transforms.Normalize(mean=[+0.5+0.485, +0.5+0.456, +0.5+0.406], std=[0.229, 0.224, 0.225])

        sigmoid_flag = 1
        if args.gan =='lsgan':
            sigmoid_flag = 0 

        if args.model=='scribbler':
            netG=scribbler.Scribbler(5,3,32)
        elif args.model=='pix2pix':
            netG=define_G(5,3,32)
        else:
            print(args.model+ ' not support. Using pix2pix model')
            netG=define_G(5,3,32)
        netD=discriminator.Discriminator(1,32,sigmoid_flag)  
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

        Extract_content = FeatureExtractor(feat_model.features, ['11'])
        Extract_style = FeatureExtractor(feat_model.features, ['0','5','10','19','28'])
        for epoch in range(args.num_epoch):
            for i, data in enumerate(dataloader, 0):


                #Detach is apparently just creating new Variable with cut off reference to previous node, so shouldn't effect the original 
                #But just in case, let's do G first so that detaching G during D update don't do anything weird
                ############################
                # (1) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()

                img, skg,seg = data #LAB with negeative value

                img=utforms.normalize_lab(img)
                skg=utforms.normalize_lab(skg)
                #skg = torch.round(skg)
               # break
                #randomize patch position/size
                crop_size = int( rand_between(args.crop_size_min, args.crop_size_max))
                xcenter = int( rand_between(crop_size/2,args.image_size-crop_size/2))
                ycenter = int( rand_between(crop_size/2,args.image_size-crop_size/2))
                inp = gen_input(img,skg,xcenter,ycenter,crop_size)


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
                outputlll= (torch.cat((outputl,outputl,outputl),1))
                targetlll = (torch.cat((targetl,targetl,targetl),1))

                ##################Pixel L Loss############################
                err_pixel_l = args.pixel_weight_l*criterion_pixel_l(outputl,targetl)

                ##################Pixel ab Loss############################
                err_pixel_ab = args.pixel_weight_ab*criterion_pixel_ab(outputab,targetab)


                ##################feature Loss############################
                out_feat = Extract_content(outputlll)[0]

                gt_feat = Extract_content(targetlll)[0]
                err_feat = args.feature_weight*criterion_feat(out_feat,gt_feat.detach())   


                ##################style Loss############################
                output_feat_ = Extract_style(outputlll)
                target_feat_ = Extract_style(targetlll)

                gram = GramMatrix()

                err_style = 0
                for m in range(len(output_feat_)): 
                    gram_y = gram(output_feat_[m])
                    gram_s = gram(target_feat_[m])

                    err_style += args.style_weight * criterion_style(gram_y, gram_s.detach())



                ##################D Loss############################
                netD.zero_grad()
                label_ = Variable(label)
                outputD = netD(outputl)

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

                print i, err_G.data[0]            


                ############################
                # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                netD.zero_grad()

                labelv = Variable(label)

                outputD = netD(targetl)

                label.resize_(outputD.data.size())
                labelv = Variable(label.fill_(real_label))

                errD_real = criterion_gan(outputD, labelv)
                errD_real.backward()

                #D_x = output_.data.mean()

                # train with fake
                #noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
                #noisev = Variable(noise)

                ##################################
                #TODO add threshold to stop updating D
                outputD = netD(outputl.detach())
                label.resize_(outputD.data.size())
                labelv = Variable(label.fill_(fake_label))

                errD_fake = criterion_gan(outputD, labelv)
                errD_fake.backward()
                #D_G_z1 = output.data.mean()
                errD = errD_real + errD_fake
                Loss_d_graph.append(errD.data[0])
                optimizerD.step()
                #TODO add discriminator accuracy

                if(i%args.save_every==0):
                    save_network(netG,'G',i,args.gpu,args.save_dir)
                    save_network(netD,'D',i,args.gpu,args.save_dir)


                #TODO test on test set
                if(i%args.visualize_every==0):

                    out_img=vis_image(utforms.denormalize_lab(outputG.data.double().cpu()))
                    out_img=(out_img*255).astype('uint8')
                    out_img=np.transpose(out_img,(2,0,1))

                    inp_img=vis_patch(utforms.denormalize_lab(img.cpu()),utforms.denormalize_lab(skg.cpu()),xcenter,ycenter,crop_size)
                    inp_img=(inp_img*255).astype('uint8')
                    inp_img=np.transpose(inp_img,(2,0,1))

                    tar_img=vis_image(utforms.denormalize_lab(img.cpu()))
                    tar_img=(tar_img*255).astype('uint8')
                    tar_img=np.transpose(tar_img,(2,0,1))

                    segment_img=vis_image((seg.cpu()))
                    segment_img=(segment_img*255).astype('uint8')
                    segment_img=np.transpose(segment_img,(2,0,1))

                    vis.image(out_img,win='output',opts=dict(title='output'))
                    vis.image(inp_img,win='input',opts=dict(title='input'))  
                    vis.image(tar_img,win='target',opts=dict(title='target'))
                    vis.image(segment_img,win='segment',opts=dict(title='segment'))
                    vis.line(np.array(Loss_gs_graph),win='gs',opts=dict(title='G-Style Loss'))
                    vis.line(np.array(Loss_g_graph),win='g',opts=dict(title='G Total Loss'))
                    vis.line(np.array(Loss_gd_graph),win='gd',opts=dict(title='G-Discriminator Loss'))
                    vis.line(np.array(Loss_gf_graph),win='gf',opts=dict(title='G-Feature Loss'))
                    vis.line(np.array(Loss_gpl_graph),win='gpl',opts=dict(title='G-Pixel Loss-L'))
                    vis.line(np.array(Loss_gpab_graph),win='gpab',opts=dict(title='G-Pixel Loss-AB'))
                    vis.line(np.array(Loss_d_graph),win='d',opts=dict(title='D Loss'))

#TODO, need to organize these func:
def rand_between(a,b):
    return a + torch.round(torch.rand(1)*(b-a))[0]

def gen_input(img,skg,xcenter=64,ycenter=64,size=40):
    #generate input skg with random patch from img
    #input img,skg [bsx3xwxh], xcenter,ycenter, size 
    #output bsx5xwxh
       
    w,h = img.size()[2:4]

    xstart = max(xcenter-size/2,0)
    ystart = max(ycenter-size/2,0)
    xend = min(xcenter + size/2,w)
    yend = min(ycenter + size/2,h)


    input_texture = torch.ones(img.size())*(1)
    input_sketch = skg[:,0:1,:,:] #L channel from skg
    input_mask = torch.ones(input_sketch.size())*(-1)

    input_mask[:,:,xstart:xend,ystart:yend] = 1
    input_texture[:,:,xstart:xend,ystart:yend] = img[:,:,xstart:xend,ystart:yend].clone()

    return torch.cat((input_sketch.float(),input_texture.float(),input_mask),1)
     
    
#TODO: move to utils
def clamp_image(img):
    img[:,0,:,:].clamp_(0,1)
    img[:,1,:,:].clamp_(-1.5,1.5)
    img[:,2,:,:].clamp_(-1.5,1.5)
    return img 

#TODO: move to model function
def save_network(model, network_label, epoch_label, gpu_id, save_dir):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(model.cpu().state_dict(), save_path)
    model.cuda(device_id=gpu_id)
#TODO: move to model function    
def load_network(model, network_label, epoch_label,save_dir):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    model.load_state_dict(torch.load(save_path))
    
#TODO move to utils
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
    
#TODO move to separate loss file
class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # normalize the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
###############added options#######################################
    parser.add_argument('-learning_rate', default=1e-5, type=float,
                    help='Learning rate for the generator')
    parser.add_argument('-learning_rate_D',  default=1e-4,type=float,
                    help='Learning rate for the discriminator')    
    
    parser.add_argument('-gan', default='lsgan',type=str,choices=['dcgan', 'lsgan'],
                    help='dcgan|lsgan') #todo wgan/improved wgan    
    
    parser.add_argument('-model', default='scribbler',type=str,choices=['scribbler', 'pix2pix'],
                   help='scribbler|pix2pix')
    
    parser.add_argument('-num_epoch',  default=1,type=int,
                    help='texture|scribbler')   
    
    parser.add_argument('-visualize_every',  default=10,type=int,
                    help='no. iteration to visualize the results')      

    #all the weights ratio, might wanna make them sum to one
    parser.add_argument('-feature_weight', default=1,type=float,
                       help='weight ratio for feature loss')
    parser.add_argument('-pixel_weight_l', default=100,type=float,
                       help='weight ratio for pixel loss for l channel')
    parser.add_argument('-pixel_weight_ab', default=500,type=float,
                   help='weight ratio for pixel loss for ab channel')
   
    parser.add_argument('-discriminator_weight', default=1e1,type=float,
                   help='weight ratio for the discriminator loss')
    parser.add_argument('-style_weight', default = 1, type=float, 
                        help='weight ratio for the texture loss')


    parser.add_argument('-gpu', default=1,type=int,
                   help='id of gpu to use') #TODO support cpu

    parser.add_argument('-display_port', default=8889,type=int,
               help='port for displaying on visdom (need to match with visdom currently open port)')

    parser.add_argument('-data_path', default='/home/psangkloy3/training_handbags_pretrain/',type=str,
                   help='path to the data directory, expect train_skg, train_img, val_skg, val_img')

    parser.add_argument('-save_dir', default='/home/psangkloy3/texturegan/save_dir_first_test',type=str,
                   help='path to save the model')
    
    parser.add_argument('-load_dir', default='/home/psangkloy3/texturegan/save_dir_first_test',type=str,
                   help='path to save the model')
    
    parser.add_argument('-save_every',  default=1000,type=int,
                    help='no. iteration to save the models')
    
    parser.add_argument('-load', default=17100,type=int,
                   help='load generator and discrminator from iteration n')
    parser.add_argument('-load_D', default=17100,type=float,
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
    
    parser.add_argument('-batch_size', default=8) #fixed batch size 1    
############################################################################
############################################################################
############TODO: TO ADD#################################################################
    parser.add_argument('-tv_weight', default=1,type=float,
                   help='weight ratio for total variation loss')
    parser.add_argument('-content_layers',  default='relu2_2',type=str,
                    help='Layer to attach content loss.')
    


    
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
    
    
    
    
##################################################################################################################################    
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    