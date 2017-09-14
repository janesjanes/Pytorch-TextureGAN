import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dataset
import visdom
import sys,argparse

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

import visdom

def main(args):

    with torch.cuda.device(args.gpu):

        vis=visdom.Visdom(port=args.display_port)

        Loss_g_graph=[]
        Loss_gd_graph=[]
        Loss_gf_graph = []
        Loss_gp_graph = []
        Loss_d_graph=[]

        ts=tforms.Compose([custom_trans.toLAB(), custom_trans.toTensor()])
        rgbify = custom_trans.toRGB()
        dset = ImageFolder(args.data_path,ts)
        dataloader=DataLoader(dataset=dset, batch_size=2, shuffle=True)

        sigmoid_flag = 1
        if args.gan =='lsgan':
            sigmoid_flag = 0 


        if args.model=='scribbler':
            netG=scribbler.Scribbler(3,3,32)
        elif args.model=='pix2pix':
            netG=define_G(3,3,32)
        else:
            print(argv.model+ ' not support. Using pix2pix model')
            netG=define_G(3,3,32)
            
        netD=discriminator.Discriminator(3,32,sigmoid_flag)  
        feat_model=models.vgg19(pretrained=True)

        netG.apply(weights_init)
        netD.apply(weights_init)    


        if args.gan =='lsgan':
            criterion_gan = nn.MSELoss()
        elif args.gan =='dcgan':
            criterion_gan = nn.BCELoss()

        criterion_l1 = nn.L1Loss()

        criterion_feat = nn.L1Loss()



        input_skg = torch.FloatTensor(2, 3, 256, 256)
        output_img = torch.FloatTensor(2, 3, 256, 256)
        label = torch.FloatTensor(2)
        real_label = 1
        fake_label = 0

        optimizerD = optim.Adam(netD.parameters(), lr=args.learning_rate_D, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        netG.cuda()
        netD.cuda()
        feat_model.cuda()
        criterion_gan.cuda()
        criterion_l1.cuda()
        criterion_feat.cuda()
        input_skg, output_img, label = input_skg.cuda(), output_img.cuda(), label.cuda()


        for epoch in range(args.num_epoch):
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                netD.zero_grad()
                img, skg = data
                img=utforms.normalize_lab(img)
                skg=utforms.normalize_lab(skg)
                img=img.cuda()
                skg=skg.cuda()
                input_skg.resize_as_(skg.float()).copy_(skg)
                output_img.resize_as_(img.float()).copy_(img)
                inputv = Variable(input_skg)
                outputv = Variable(output_img)
                labelv = Variable(label)
                #print labelv.data.size()

                output = netD(inputv)

                label.resize_(output.data.size())
                labelv = Variable(label.fill_(real_label))
                errD_real = criterion_gan(output, labelv)
                errD_real.backward()
                D_x = output.data.mean()

                # train with fake
                #noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
                #noisev = Variable(noise)
                fake = netG(inputv)
                output = netD(fake.detach())
                label.resize_(output.data.size())
                labelv = Variable(label.fill_(fake_label))

                errD_fake = criterion_gan(output, labelv)
                errD_fake.backward()
                D_G_z1 = output.data.mean()
                errD = errD_real + errD_fake
                Loss_d_graph.append(errD.data[0])
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
                output = netD(fake)

                multer=torch.ones(inputv.data.size())
                multer[:,1:,:,:]=args.pixel_weight_ab*multer[:,1:,:,:]
                multer=multer.cuda()
                multer=Variable(multer)

                multer_=torch.ones(inputv.data.size())
                multer_[:,0,:,:]=args.pixel_weight_l*multer_[:,0,:,:]
                multer_=multer_.cuda()
                multer_=Variable(multer_)

                #print outputv.size(), multer.size()
                woutputv = outputv*multer*multer_
                err_pixel = criterion_l1(fake*multer, woutputv)
                #####################################
                D_G_z2 = output.data.mean()

                label.resize_(output.data.size())
                labelv = Variable(label.fill_(real_label))

                err_gan = args.discriminator_weight*criterion_gan(output, labelv)

                ####################################
                #TODO normalize and minus mean?
                L,A,B=torch.chunk(fake,3,dim=1)
                LLL=torch.cat((L,L,L),1)
                out_feat=feat_model.features(LLL)

                #print(LLL.size())
                #break
                gt_feat = feat_model.features(outputv)

                gt_feat = gt_feat.detach() #don't require grad for this

                err_feat = args.feature_weight*criterion_feat(out_feat,gt_feat)
                #err_feat.backward()

                err_G = err_pixel + err_gan + err_feat
                err_G.backward()

                optimizerG.step()
                Loss_g_graph.append(err_G.data[0])
                Loss_gp_graph.append(err_pixel.data[0])
                Loss_gd_graph.append(err_gan.data[0])
                Loss_gf_graph.append(err_feat.data[0])
                #plt.imshow(vis_image(inputv.data.double().cpu()))

                print i, err_G.data[0]
                #TODO test on test set
                if(i%args.visualize_every==0):
                    test_img=clamp_image(fake.data.double().cpu())
                    test_img=utforms.denormalize_lab(test_img)
                    test_img=vis_image(test_img)
                    test_img=(test_img*255).astype('uint8')
                    test_img=np.transpose(test_img,(2,0,1))

                    inp_img=vis_patch(utforms.denormalize_lab(img.cpu()),utforms.denormalize_lab(skg.cpu()))
                    inp_img=(inp_img*255).astype('uint8')
                    inp_img=np.transpose(inp_img,(2,0,1))

                    target_img=vis_image(utforms.denormalize_lab(img.cpu()))
                    target_img=(target_img*255).astype('uint8')
                    target_img=np.transpose(target_img,(2,0,1))

                    vis.image(test_img,win='output',opts=dict(title='output'))
                    vis.image(inp_img,win='input',opts=dict(title='input'))  
                    vis.image(inp_img,win='input',opts=dict(title='input'))
                    vis.line(np.array(Loss_g_graph),win='g',opts=dict(title='G Total Loss'))
                    vis.line(np.array(Loss_gd_graph),win='gd',opts=dict(title='G-Discriminator Loss'))
                    vis.line(np.array(Loss_gf_graph),win='gf',opts=dict(title='G-Feature Loss'))
                    vis.line(np.array(Loss_d_graph),win='d',opts=dict(title='D Loss'))

#TODO: move to utils
def clamp_image(img):
    img[:,0,:,:].clamp_(0,1)
    img[:,1,:,:].clamp_(-1.5,1.5)
    img[:,2,:,:].clamp_(-1.5,1.5)
    return img    
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
###############added options#######################################
    parser.add_argument('-learning_rate', default=1e-4, type=float,
                    help='Learning rate for the generator')
    parser.add_argument('-learning_rate_D',  default=1e-4,type=float,
                    help='Learning rate for the discriminator')    
    
    parser.add_argument('-gan', default='dcgan',type=str,choices=['dcgan', 'lsgan'],
                    help='dcgan|lsgan') #todo wgan/improved wgan    
    
    parser.add_argument('-model', default='pix2pix',type=str,choices=['scribbler', 'pix2pix'],
                   help='scribbler|pix2pix')
    
    parser.add_argument('-num_epoch',  default=1,type=int,
                    help='texture|scribbler')   
    
    parser.add_argument('-visualize_every',  default=10,type=int,
                    help='no. iteration to visualize the results')      

    #all the weights ratio, might wanna make them sum to one
    parser.add_argument('-feature_weight', default=10,type=float,
                       help='weight ratio for feature loss')
    parser.add_argument('-pixel_weight_l', default=1,type=float,
                       help='weight ratio for pixel loss for l channel')
    parser.add_argument('-pixel_weight_ab', default=10,type=float,
                   help='weight ratio for pixel loss for ab channel')
    parser.add_argument('-tv_weight', default=1,type=float,
                   help='weight ratio for total variation loss')
    parser.add_argument('-discriminator_weight', default=0,type=float,
                   help='weight ratio for the discriminator loss')

    parser.add_argument('-gpu', default=1,type=int,
                   help='id of gpu to use') #TODO support cpu

    parser.add_argument('-display_port', default=7779,type=int,
               help='port for displaying on visdom (need to match with visdom currently open port)')

    parser.add_argument('-data_path', default='/home/psangkloy3/training_handbags_pretrain/',type=str,
                   help='path to the data directory, expect train_skg, train_img, val_skg, val_img')


############################################################################
############################################################################
############TODO: TO ADD#################################################################
    parser.add_argument('-content_layers',  default='relu2_2',type=str,
                    help='Layer to attach content loss.')
    
    parser.add_argument('-batch_size', default=1) #fixed batch size 1
    
    parser.add_argument('-image_size',default=128,type=int,
                    help='Training images size, after cropping')        
    parser.add_argument('-resize_max',  default=256,type=int,
                    help='max resize size')        
    parser.add_argument('-resize_min',  default=128,type=int,
                    help='min resize size')   
    

    
    parser.add_argument('-mode',  default='texture',type=str,choices=['texture','scribbler'],
                    help='texture|scribbler') 
    
    parser.add_argument('-save_every',  default=50000,type=int,
                    help='no. iteration to save the models')
    
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
    
    parser.add_argument('-load', default=1,type=int,
                   help='load generator and discrminator from iteration n')
    parser.add_argument('-load_D', default=1,type=float,
                   help='load discriminator from iteration n, priority over load')
    
    parser.add_argument('-absolute_load', default='',type=str,
                   help='load saved generator model from absolute location')
    
    
    
    
##################################################################################################################################    
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    