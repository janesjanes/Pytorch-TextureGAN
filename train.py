import torch
from torch.autograd import Variable
import numpy as np
from utils import transforms as custom_transforms
from models import save_network, GramMatrix
from utils.visualize import vis_image, vis_patch
import time
#import cv2
import math
import random

def rand_between(a, b):
    return a + torch.round(torch.rand(1) * (b - a))[0]


def gen_input(img, skg, ini_texture, ini_mask, xcenter=64, ycenter=64, size=40):
    # generate input skg with random patch from img
    # input img,skg [bsx3xwxh], xcenter,ycenter, size
    # output bsx5xwxh

    w, h = img.size()[1:3]
    # print w,h
    xstart = max(int(xcenter - size / 2), 0)
    ystart = max(int(ycenter - size / 2), 0)
    xend = min(int(xcenter + size / 2), w)
    yend = min(int(ycenter + size / 2), h)

    input_texture = ini_texture  # torch.ones(img.size())*(1)
    input_sketch = skg[0:1, :, :]  # L channel from skg
    input_mask = ini_mask  # torch.ones(input_sketch.size())*(-1)

    input_mask[:, xstart:xend, ystart:yend] = 1

    input_texture[:, xstart:xend, ystart:yend] = img[:, xstart:xend, ystart:yend].clone()

    return torch.cat((input_sketch.cpu().float(), input_texture.float(), input_mask), 0)

def get_coor(index, size):
    index = int(index)
    #get original coordinate from flatten index for 3 dim size
    w,h = size
    
    return ((index%(w*h))/h, ((index%(w*h))%h))

def gen_input_rand(img, skg, seg, size_min=40, size_max=60, num_patch=1):
    # generate input skg with random patch from img
    # input img,skg [bsx3xwxh], xcenter,ycenter, size
    # output bsx5xwxh
    
    bs, c, w, h = img.size()
    results = torch.Tensor(bs, 5, w, h)
    texture_info = []

    # text_info.append([xcenter,ycenter,crop_size])
    seg = seg / torch.max(seg) #make sure it's 0/1
    
    seg[:,0:int(math.ceil(size_min/2)),:] = 0
    seg[:,:,0:int(math.ceil(size_min/2))] = 0
    seg[:,:,int(math.floor(h-size_min/2)):h] = 0
    seg[:,int(math.floor(w-size_min/2)):w,:] = 0
    
    counter = 0
    for i in range(bs):
        counter = 0
        ini_texture = torch.ones(img[0].size()) * (1)
        ini_mask = torch.ones((1, w, h)) * (-1)
        temp_info = []
        
        for j in range(num_patch):
            crop_size = int(rand_between(size_min, size_max))
            
            seg_index_size = seg[i,:,:].view(-1).size()[0]
            seg_index = torch.arange(0,seg_index_size)
            seg_one = seg_index[seg[i,:,:].view(-1)==1]
            if len(seg_one) != 0:
                seg_select_index = int(rand_between(0,seg_one.view(-1).size()[0]-1))
                x,y = get_coor(seg_one[seg_select_index],seg[i,:,:].size())
            else:
                x,y = (w/2, h/2)
            
            temp_info.append([x, y, crop_size])
            res = gen_input(img[i], skg[i], ini_texture, ini_mask, x, y, crop_size)

            ini_texture = res[1:4, :, :]

        texture_info.append(temp_info)
        results[i, :, :, :] = res
    return results, texture_info

def gen_local_patch(patch_size, batch_size, eroded_seg, seg, img):
    # generate local loss patch from eroded segmentation
    
    bs, c, w, h = img.size()
    texture_patch = img[:, :, 0:patch_size, 0:patch_size].clone()

    if patch_size != -1:
        eroded_seg[:,0,0:int(math.ceil(patch_size/2)),:] = 0
        eroded_seg[:,0,:,0:int(math.ceil(patch_size/2))] = 0
        eroded_seg[:,0,:,int(math.floor(h-patch_size/2)):h] = 0
        eroded_seg[:,0,int(math.floor(w-patch_size/2)):w,:] = 0

    for i_bs in range(bs):
                
        i_bs = int(i_bs)
        seg_index_size = eroded_seg[i_bs,0,:,:].view(-1).size()[0]
        seg_index = torch.arange(0,seg_index_size).cuda()
        #import pdb; pdb.set_trace()
        #print bs, batch_size
        seg_one = seg_index[eroded_seg[i_bs,0,:,:].view(-1)==1]
        if len(seg_one) != 0:
            random_select = int(rand_between(0, len(seg_one)-1))
            #import pdb; pdb.set_trace()
            
            x,y = get_coor(seg_one[random_select], eroded_seg[i_bs,0,:,:].size())
            #print x,y,i_bs
        else:
            x,y = (w/2, h/2)

        if patch_size == -1:
            xstart = 0
            ystart = 0
            xend = -1
            yend = -1

        else:
            xstart = int(x-patch_size/2)
            ystart = int(y-patch_size/2)
            xend = int(x+patch_size/2)
            yend = int(y+patch_size/2)

        k = 1
        while torch.sum(seg[i_bs,0,xstart:xend,ystart:yend]) < k*patch_size*patch_size:
                
            try:
                k = k*0.9
                if len(seg_one) != 0:
                    random_select = int(rand_between(0, len(seg_one)-1))
            
                    x,y = get_coor(seg_one[random_select], eroded_seg[i_bs,0,:,:].size())
            
                else:
                    x,y = (w/2, h/2)
                xstart = (int)(x-patch_size/2)
                ystart = (int)(y-patch_size/2)
                xend = (int)(x+patch_size/2)
                yend = (int)(y+patch_size/2)
            except:
                break
                
            
        texture_patch[i_bs,:,:,:] = img[i_bs, :, xstart:xend, ystart:yend]
        
    return texture_patch

def renormalize(img):
    """
    Renormalizes the input image to meet requirements for VGG-19 pretrained network
    """

    forward_norm = torch.ones(img.data.size()) * 0.5
    forward_norm = Variable(forward_norm.cuda())
    img = (img * forward_norm) + forward_norm  # add previous norm
    # return img
    mean = img.data.new(img.data.size())
    std = img.data.new(img.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    img -= Variable(mean)
    img = img / Variable(std)

    return img


def visualize_training(netG, val_loader,input_stack, target_img, target_texture,segment, vis, loss_graph, args):
    imgs = []
    for ii, data in enumerate(val_loader, 0):
        img, skg, seg, eroded_seg, txt = data  # LAB with negeative value
        if random.random() < 0.5:
            txt = img
        # this is in LAB value 0/100, -128/128 etc
        img = custom_transforms.normalize_lab(img)
        skg = custom_transforms.normalize_lab(skg)
        txt = custom_transforms.normalize_lab(txt)
        seg = custom_transforms.normalize_seg(seg)
        eroded_seg = custom_transforms.normalize_seg(eroded_seg)
        
        bs, w, h = seg.size()
        
        seg = seg.view(bs, 1, w, h)
        seg = torch.cat((seg, seg, seg), 1)
        
        eroded_seg = eroded_seg.view(bs, 1, w, h)
        eroded_seg = torch.cat((eroded_seg, eroded_seg, eroded_seg), 1)

        temp = torch.ones(seg.size()) * (1 - seg).float()
        temp[:, 1, :, :] = 0  # torch.ones(seg[:,1,:,:].size())*(1-seg[:,1,:,:]).float()
        temp[:, 2, :, :] = 0  # torch.ones(seg[:,2,:,:].size())*(1-seg[:,2,:,:]).float()

        txt = txt.float() * seg.float() + temp
      
        patchsize = args.local_texture_size
        batch_size = bs
              
        # seg=custom_transforms.normalize_lab(seg)
        # norm to 0-1 minus mean
        if not args.use_segmentation_patch:
            seg.fill_(1)
            #skg.fill_(0)
            eroded_seg.fill_(1)
        if args.input_texture_patch == 'original_image':
            inp, texture_loc = gen_input_rand(img, skg, eroded_seg[:, 0, :, :] * 100,
                                              args.patch_size_min, args.patch_size_max,
                                              args.num_input_texture_patch)
        elif args.input_texture_patch == 'dtd_texture':
            inp, texture_loc = gen_input_rand(txt, skg, eroded_seg[:, 0, :, :] * 100,
                                              args.patch_size_min, args.patch_size_max,
                                              args.num_input_texture_patch)

        img = img.cuda()
        skg = skg.cuda()
        seg = seg.cuda()
        eroded_seg = eroded_seg.cuda()
        txt = txt.cuda()
        inp = inp.cuda()

        inp.size()

        input_stack.resize_as_(inp.float()).copy_(inp)
        target_img.resize_as_(img.float()).copy_(img)
        segment.resize_as_(seg.float()).copy_(seg)
        target_texture.resize_as_(txt.float()).copy_(txt)

        inputv = Variable(input_stack)
        targetv = Variable(target_img)
        
        gtimgv = Variable(target_img)
        segv = Variable(segment)
        txtv = Variable(target_texture)

        outputG = netG(inputv)
        
        outputl, outputa, outputb = torch.chunk(outputG, 3, dim=1)
        #outputlll = (torch.cat((outputl, outputl, outputl), 1))
        gtl, gta, gtb = torch.chunk(gtimgv, 3, dim=1)
        txtl, txta, txtb = torch.chunk(txtv, 3, dim=1)
        
        gtab = torch.cat((gta, gtb), 1)
        txtab= torch.cat((txta, txtb), 1)
        
        if args.color_space == 'lab':
            outputlll = (torch.cat((outputl, outputl, outputl), 1))
            gtlll = (torch.cat((gtl, gtl, gtl), 1))
            txtlll = torch.cat((txtl, txtl, txtl), 1)
        elif args.color_space == 'rgb':
            outputlll = outputG  # (torch.cat((outputl,outputl,outputl),1))
            gtlll = gtimgv  # (torch.cat((targetl,targetl,targetl),1))
            txtlll = txtv
        if args.loss_texture == 'original_image':
            targetl = gtl
            targetab = gtab
            targetlll = gtlll
        else:
            targetl = txtl
            targetab = txtab
            targetlll = txtlll
       # import pdb; pdb.set_trace()

        texture_patch = gen_local_patch(patchsize, batch_size, eroded_seg, seg, outputlll)
        gt_texture_patch = gen_local_patch(patchsize, batch_size, eroded_seg, seg, targetlll)


    if args.color_space == 'lab':
        out_img = vis_image(custom_transforms.denormalize_lab(outputG.data.double().cpu()),
                            args.color_space)
        temp_labout = custom_transforms.denormalize_lab(texture_patch.data.double().cpu())
        temp_labout[:,1:3,:,:] = 0
        
        temp_labgt = custom_transforms.denormalize_lab(gt_texture_patch.data.double().cpu())
        temp_labgt[:,1:3,:,:] = 0
        temp_out =vis_image(temp_labout,args.color_space) #torch.cat((patches[0].data.double().cpu(),patches[0].data.double().cpu(),patches[0].data.double().cpu()),1)
        #temp_out = (temp_out + 1 )/2
                            
        temp_gt =vis_image(temp_labgt,
                            args.color_space) #torch.cat((patches[1].data.double().cpu(),patches[1].data.double().cpu(),patches[1].data.double().cpu()),1)

        if args.input_texture_patch == 'original_image':
            inp_img = vis_patch(custom_transforms.denormalize_lab(img.cpu()),
                                custom_transforms.denormalize_lab(skg.cpu()),
                                texture_loc,
                                args.color_space)
        elif args.input_texture_patch == 'dtd_texture':
            inp_img = vis_patch(custom_transforms.denormalize_lab(txt.cpu()),
                                custom_transforms.denormalize_lab(skg.cpu()),
                                texture_loc,
                                args.color_space)
        tar_img = vis_image(custom_transforms.denormalize_lab(img.cpu()),
                            args.color_space)
        skg_img = vis_image(custom_transforms.denormalize_lab(skg.cpu()),
                            args.color_space)
        txt_img = vis_image(custom_transforms.denormalize_lab(txt.cpu()),
                            args.color_space)
    elif args.color_space == 'rgb':

        out_img = vis_image(custom_transforms.denormalize_rgb(outputG.data.double().cpu()),
                            args.color_space)
        inp_img = vis_patch(custom_transforms.denormalize_rgb(img.cpu()),
                            custom_transforms.denormalize_rgb(skg.cpu()),
                            texture_loc,
                            args.color_space)
        tar_img = vis_image(custom_transforms.denormalize_rgb(img.cpu()),
                            args.color_space)

    out_final = [x*0 for x in txt_img] 
    gt_final = [x*0 for x in txt_img] 
    out_img = [x * 255 for x in out_img]  # (out_img*255)#.astype('uint8')
    skg_img = [x * 255 for x in skg_img]  # (out_img*255)#.astype('uint8')
    out_patch = [x * 255 for x in temp_out]
    gt_patch = [x * 255 for x in temp_gt]    # out_img=np.transpose(out_img,(2,0,1))
    for t_i in range(bs):
        #import pdb; pdb.set_trace()
        patchsize = int(args.local_texture_size)
        out_final[t_i][:,0:patchsize,0:patchsize] = out_patch[t_i][:,:,:]# .append(np.resize(out_patch[t_i], (3,w,h)))
        gt_final[t_i][:,0:patchsize,0:patchsize] =gt_patch[t_i][:,:,:]#gt_final.append(np.resize(gt_patch[t_i], (3,w,h)))
   
    
    # out_img=np.transpose(out_img,(2,0,1))

    txt_img = [x * 255 for x in txt_img]    
    inp_img = [x * 255 for x in inp_img]  # (inp_img*255)#.astype('uint8')
    # inp_img=np.transpose(inp_img,(2,0,1))

    tar_img = [x * 255 for x in tar_img]  # (tar_img*255)#.astype('uint8')
    # tar_img=np.transpose(tar_img,(2,0,1))
    #import pdb; pdb.set_trace()
    
    #segment_img = vis_image((eroded_seg.cpu()), args.color_space)
    #import pdb; pdb.set_trace()
    segment_img = [x * 255 for x in eroded_seg.cpu().numpy()]  # segment_img=(segment_img*255)#.astype('uint8')
    # segment_img=np.transpose(segment_img,(2,0,1))
    #import pdb; pdb.set_trace()
    for i_ in range(len(out_img)):
        #import pdb; pdb.set_trace()
        imgs.append(skg_img[i_])
        imgs.append(txt_img[i_])
        imgs.append(inp_img[i_])
        imgs.append(out_img[i_])
        imgs.append(segment_img[i_])
        imgs.append(tar_img[i_])
        imgs.append(out_final[i_])
        imgs.append(gt_final[i_])

    # for idx, img in enumerate(imgs):
    #     print(idx, type(img), img.shape)

    vis.images(imgs, win='output', opts=dict(title='Output images'))
    # vis.image(inp_img,win='input',opts=dict(title='input'))
    # vis.image(tar_img,win='target',opts=dict(title='target'))
    # vis.image(segment_img,win='segment',opts=dict(title='segment'))
    vis.line(np.array(loss_graph["gs"]), win='gs', opts=dict(title='G-Style Loss'))
    vis.line(np.array(loss_graph["g"]), win='g', opts=dict(title='G Total Loss'))
    vis.line(np.array(loss_graph["gd"]), win='gd', opts=dict(title='G-Discriminator Loss'))
    vis.line(np.array(loss_graph["gf"]), win='gf', opts=dict(title='G-Feature Loss'))
    vis.line(np.array(loss_graph["gpl"]), win='gpl', opts=dict(title='G-Pixel Loss-L'))
    vis.line(np.array(loss_graph["gpab"]), win='gpab', opts=dict(title='G-Pixel Loss-AB'))
    vis.line(np.array(loss_graph["d"]), win='d', opts=dict(title='D Loss'))
    if args.local_texture_size != -1:
        vis.line(np.array(loss_graph["dl"]), win='dl', opts=dict(title='D Local Loss'))
        vis.line(np.array(loss_graph["gdl"]), win='gdl', opts=dict(title='G D Local Loss'))
    
def train(model, train_loader, val_loader, input_stack, target_img, target_texture,
          segment, label,label_local, extract_content, extract_style, loss_graph, vis, epoch, args):

    netG = model["netG"]
    netD = model["netD"]
    netD_local = model["netD_local"]
    criterion_gan = model["criterion_gan"]
    criterion_pixel_l = model["criterion_pixel_l"]
    criterion_pixel_ab = model["criterion_pixel_ab"]
    criterion_feat = model["criterion_feat"]
    criterion_style = model["criterion_style"]
    criterion_texturegan = model["criterion_texturegan"]
    real_label = model["real_label"]
    fake_label = model["fake_label"]
    optimizerD = model["optimizerD"]
    optimizerD_local = model["optimizerD_local"]
    optimizerG = model["optimizerG"]

    for i, data in enumerate(train_loader):

        print("Epoch: {0}       Iteration: {1}".format(epoch, i))
        # Detach is apparently just creating new Variable with cut off reference to previous node, so shouldn't effect the original
        # But just in case, let's do G first so that detaching G during D update don't do anything weird
        ############################
        # (1) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()

        img, skg, seg, eroded_seg, txt = data  # LAB with negeative value
        if random.random() < 0.5:
            txt = img
        # output img/skg/seg rgb between 0-1
        # output img/skg/seg lab between 0-100, -128-128
        if args.color_space == 'lab':
            img = custom_transforms.normalize_lab(img)
            skg = custom_transforms.normalize_lab(skg)
            txt = custom_transforms.normalize_lab(txt)
            seg = custom_transforms.normalize_seg(seg)
            eroded_seg = custom_transforms.normalize_seg(eroded_seg)
            # seg = custom_transforms.normalize_lab(seg)
        elif args.color_space == 'rgb':
            img = custom_transforms.normalize_rgb(img)
            skg = custom_transforms.normalize_rgb(skg)
            txt = custom_transforms.normalize_rgb(txt)
            # seg=custom_transforms.normalize_rgb(seg)
        # print seg
        if not args.use_segmentation_patch:
            seg.fill_(1)
         
        bs, w, h = seg.size()

        seg = seg.view(bs, 1, w, h)
        seg = torch.cat((seg, seg, seg), 1)
        eroded_seg = eroded_seg.view(bs, 1, w, h)
        
        # import pdb; pdb.set_trace()

        temp = torch.ones(seg.size()) * (1 - seg).float()
        temp[:, 1, :, :] = 0  # torch.ones(seg[:,1,:,:].size())*(1-seg[:,1,:,:]).float()
        temp[:, 2, :, :] = 0  # torch.ones(seg[:,2,:,:].size())*(1-seg[:,2,:,:]).float()

        txt = txt.float() * seg.float() + temp
        #tic = time.time()
        if args.input_texture_patch == 'original_image':
            inp, _ = gen_input_rand(img, skg, eroded_seg[:, 0, :, :], args.patch_size_min, args.patch_size_max,
                                    args.num_input_texture_patch)
        elif args.input_texture_patch == 'dtd_texture':
            inp, _ = gen_input_rand(txt, skg, eroded_seg[:, 0, :, :], args.patch_size_min, args.patch_size_max,
                                    args.num_input_texture_patch)
        #print(time.time()-tic)
        batch_size, _, _, _ = img.size()

        img = img.cuda()
        skg = skg.cuda()
        seg = seg.cuda()
        eroded_seg = eroded_seg.cuda()
        txt = txt.cuda()

        inp = inp.cuda()

        input_stack.resize_as_(inp.float()).copy_(inp)
        target_img.resize_as_(img.float()).copy_(img)
        segment.resize_as_(seg.float()).copy_(seg)
        target_texture.resize_as_(txt.float()).copy_(txt)
        
        inv_idx = torch.arange(target_texture.size(0)-1, -1, -1).long().cuda()
        target_texture_inv = target_texture.index_select(0, inv_idx)

        assert torch.max(seg) <= 1
        assert torch.max(eroded_seg) <= 1

        inputv = Variable(input_stack)
        gtimgv = Variable(target_img)
        segv = Variable(segment)
        txtv = Variable(target_texture)
        txtv_inv = Variable(target_texture_inv)
        
        outputG = netG(inputv)

        outputl, outputa, outputb = torch.chunk(outputG, 3, dim=1)

        gtl, gta, gtb = torch.chunk(gtimgv, 3, dim=1)
        txtl, txta, txtb = torch.chunk(txtv, 3, dim=1)
        txtl_inv,txta_inv,txtb_inv = torch.chunk(txtv_inv,3,dim=1)

        outputab = torch.cat((outputa, outputb), 1)
        gtab = torch.cat((gta, gtb), 1)
        txtab = torch.cat((txta, txtb), 1)

        if args.color_space == 'lab':
            outputlll = (torch.cat((outputl, outputl, outputl), 1))
            gtlll = (torch.cat((gtl, gtl, gtl), 1))
            txtlll = torch.cat((txtl, txtl, txtl), 1)
        elif args.color_space == 'rgb':
            outputlll = outputG  # (torch.cat((outputl,outputl,outputl),1))
            gtlll = gtimgv  # (torch.cat((targetl,targetl,targetl),1))
            txtlll = txtv
        if args.loss_texture == 'original_image':
            targetl = gtl
            targetab = gtab
            targetlll = gtlll
        else:
            # if args.loss_texture == 'texture_mask':
            # remove baskground dtd
            #     txtl = segv[:,0:1,:,:]*txtl
            #     txtab=segv[:,1:3,:,:]*txtab
            #     txtlll=segv*txtlll
            # elif args.loss_texture == 'texture_patch':

            targetl = txtl
            targetab = txtab
            targetlll = txtlll

        ################## Global Pixel ab Loss ############################
        
        err_pixel_ab = args.pixel_weight_ab * criterion_pixel_ab(outputab, targetab)

        ################## Global Feature Loss############################
        
        out_feat = extract_content(renormalize(outputlll))[0]

        gt_feat = extract_content(renormalize(gtlll))[0]
        err_feat = args.feature_weight * criterion_feat(out_feat, gt_feat.detach())

        ################## Global D Adversarial Loss ############################
        
        netD.zero_grad()
        label_ = Variable(label)
        
        #return outputl, txtl
        if args.color_space == 'lab':
            outputD = netD(outputl)
        elif args.color_space == 'rgb':
            outputD = netD(outputG)
        # D_G_z2 = outputD.data.mean()

        label.resize_(outputD.data.size())
        labelv = Variable(label.fill_(real_label))

        err_gan = args.discriminator_weight * criterion_gan(outputD, labelv)
        err_pixel_l = 0
        ################## Global Pixel L Loss ############################
             
        err_pixel_l = args.global_pixel_weight_l * criterion_pixel_l(outputl, targetl)
        if args.local_texture_size == -1:  # global, no loss patch
            
            ################## Global Style Loss ############################
            
            output_style_feat = extract_style(outputlll)
            target_style_feat = extract_style(targetlll)
            
            gram = GramMatrix()

            err_style = 0
            for m in range(len(output_style_feat)):
                gram_y = gram(output_style_feat[m])
                gram_s = gram(target_style_feat[m])

                err_style += args.style_weight * criterion_style(gram_y, gram_s.detach())
            
            

            
            err_texturegan = 0
                        
        else: # local loss patch
            err_style = 0
            
            patchsize = args.local_texture_size
            
            netD_local.zero_grad()
             
            for p in range(args.num_local_texture_patch):
                texture_patch = gen_local_patch(patchsize, batch_size, eroded_seg,seg, outputlll)
                gt_texture_patch = gen_local_patch(patchsize, batch_size, eroded_seg,seg, targetlll)

                texture_patchl = gen_local_patch(patchsize, batch_size, eroded_seg, seg,outputl)
                gt_texture_patchl = gen_local_patch(patchsize, batch_size, eroded_seg,seg, targetl)

                ################## Local Style Loss ############################

                output_style_feat = extract_style(texture_patch)
                target_style_feat = extract_style(gt_texture_patch)

                gram = GramMatrix()


                for m in range(len(output_style_feat)):
                    gram_y = gram(output_style_feat[m])
                    gram_s = gram(target_style_feat[m])

                    err_style += args.style_weight * criterion_style(gram_y, gram_s.detach())

                ################## Local Pixel L Loss ############################

                err_pixel_l += args.local_pixel_weight_l * criterion_pixel_l(texture_patchl, gt_texture_patchl)
            
            
                ################## Local D Loss ############################
                
                label_ = Variable(label)
                err_texturegan = 0
            
                outputD_local = netD_local(torch.cat((texture_patchl, gt_texture_patchl),1))

                label_local.resize_(outputD_local.data.size())
                labelv_local = Variable(label_local.fill_(real_label))

                err_texturegan += args.discriminator_local_weight * criterion_texturegan(outputD_local, labelv_local)
            loss_graph["gdl"].append(err_texturegan.data[0])
        
        ####################################
        err_G = err_pixel_l + err_pixel_ab + err_gan + err_feat + err_style + err_texturegan
        
        err_G.backward(retain_variables=True)

        optimizerG.step()

        loss_graph["g"].append(err_G.data[0])
        loss_graph["gpl"].append(err_pixel_l.data[0])
        loss_graph["gpab"].append(err_pixel_ab.data[0])
        loss_graph["gd"].append(err_gan.data[0])
        loss_graph["gf"].append(err_feat.data[0])
        loss_graph["gs"].append(err_style.data[0])
            

        print('G:', err_G.data[0])

        ############################
        # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        
        
        netD.zero_grad()

        labelv = Variable(label)
        if args.color_space == 'lab':
            outputD = netD(gtl)
        elif args.color_space == 'rgb':
            outputD = netD(gtimgv)

        label.resize_(outputD.data.size())
        labelv = Variable(label.fill_(real_label))

        errD_real = criterion_gan(outputD, labelv)
        errD_real.backward()

        score = Variable(torch.ones(batch_size))
        _, cd, wd, hd = outputD.size()
        D_output_size = cd * wd * hd

        clamped_output_D = outputD.clamp(0, 1)
        clamped_output_D = torch.round(clamped_output_D)
        for acc_i in range(batch_size):
            score[acc_i] = torch.sum(clamped_output_D[acc_i]) / D_output_size

        real_acc = torch.mean(score)

        if args.color_space == 'lab':
            outputD = netD(outputl.detach())
        elif args.color_space == 'rgb':
            outputD = netD(outputG.detach())
        label.resize_(outputD.data.size())
        labelv = Variable(label.fill_(fake_label))

        errD_fake = criterion_gan(outputD, labelv)
        errD_fake.backward()
        score = Variable(torch.ones(batch_size))
        _, cd, wd, hd = outputD.size()
        D_output_size = cd * wd * hd

        clamped_output_D = outputD.clamp(0, 1)
        clamped_output_D = torch.round(clamped_output_D)
        for acc_i in range(batch_size):
            score[acc_i] = torch.sum(clamped_output_D[acc_i]) / D_output_size

        fake_acc = torch.mean(1 - score)

        D_acc = (real_acc + fake_acc) / 2

        if D_acc.data[0] < args.threshold_D_max:
            # D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            loss_graph["d"].append(errD.data[0])
            optimizerD.step()
        else:
            loss_graph["d"].append(0)

        print('D:', 'real_acc', "%.2f" % real_acc.data[0], 'fake_acc', "%.2f" % fake_acc.data[0], 'D_acc', D_acc.data[0])

        ############################
        # (2) Update D local network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        
        if args.local_texture_size != -1:
            patchsize = args.local_texture_size
            x1 = int(rand_between(patchsize, args.image_size - patchsize))
            y1 = int(rand_between(patchsize, args.image_size - patchsize))

            x2 = int(rand_between(patchsize, args.image_size - patchsize))
            y2 = int(rand_between(patchsize, args.image_size - patchsize))

            netD_local.zero_grad()

            labelv = Variable(label)
            if args.color_space == 'lab':
                outputD_local = netD_local(torch.cat((targetl[:, :, x1:(x1 + patchsize), y1:(y1 + patchsize)],targetl[:, :, x2:(x2 + patchsize), y2:(y2 + patchsize)]),1))#netD_local(targetl)
            elif args.color_space == 'rgb':
                outputD = netD(gtimgv)

            label.resize_(outputD_local.data.size())
            labelv = Variable(label.fill_(real_label))

            errD_real_local = criterion_texturegan(outputD_local, labelv)
            errD_real_local.backward(retain_variables=True)

            score = Variable(torch.ones(batch_size))
            _, cd, wd, hd = outputD_local.size()
            D_output_size = cd * wd * hd

            clamped_output_D = outputD_local.clamp(0, 1)
            clamped_output_D = torch.round(clamped_output_D)
            for acc_i in range(batch_size):
                score[acc_i] = torch.sum(clamped_output_D[acc_i]) / D_output_size

            realreal_acc = torch.mean(score)

            

            x1 = int(rand_between(patchsize, args.image_size - patchsize))
            y1 = int(rand_between(patchsize, args.image_size - patchsize))

            x2 = int(rand_between(patchsize, args.image_size - patchsize))
            y2 = int(rand_between(patchsize, args.image_size - patchsize))


            if args.color_space == 'lab':
                #outputD_local = netD_local(torch.cat((txtl[:, :, x1:(x1 + patchsize), y1:(y1 + patchsize)],outputl[:, :, x2:(x2 + patchsize), y2:(y2 + patchsize)]),1))#outputD = netD(outputl.detach())
                outputD_local = netD_local(torch.cat((texture_patchl, gt_texture_patchl),1))
            elif args.color_space == 'rgb':
                outputD = netD(outputG.detach())
            label.resize_(outputD_local.data.size())
            labelv = Variable(label.fill_(fake_label))

            errD_fake_local = criterion_gan(outputD_local, labelv)
            errD_fake_local.backward()
            score = Variable(torch.ones(batch_size))
            _, cd, wd, hd = outputD_local.size()
            D_output_size = cd * wd * hd

            clamped_output_D = outputD_local.clamp(0, 1)
            clamped_output_D = torch.round(clamped_output_D)
            for acc_i in range(batch_size):
                score[acc_i] = torch.sum(clamped_output_D[acc_i]) / D_output_size

            fakefake_acc = torch.mean(1 - score)

            D_acc = (realreal_acc +fakefake_acc) / 2

            if D_acc.data[0] < args.threshold_D_max:
                # D_G_z1 = output.data.mean()
                errD_local = errD_real_local + errD_fake_local
                loss_graph["dl"].append(errD_local.data[0])
                optimizerD_local.step()
            else:
                loss_graph["dl"].append(0)

            print('D local:', 'real real_acc', "%.2f" % realreal_acc.data[0], 'fake fake_acc', "%.2f" % fakefake_acc.data[0], 'D_acc', D_acc.data[0])
            #if i % args.save_every == 0:
             #   save_network(netD_local, 'D_local', epoch, i, args)

        if i % args.save_every == 0:
            save_network(netG, 'G', epoch, i, args)
            save_network(netD, 'D', epoch, i, args)
            save_network(netD_local, 'D_local', epoch, i, args)
            
        if i % args.visualize_every == 0:
            visualize_training(netG, val_loader, input_stack, target_img,target_texture, segment, vis, loss_graph, args)


