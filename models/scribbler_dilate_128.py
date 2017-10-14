import torch
import torch.nn as nn


class ScribblerDilate128(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        """
        Defines the necessary modules of the Scribbler Generator

        Input:
        - int input_nc : Input number of channels
        - int output_nc : Output number of channels
        """
        super(ScribblerDilate128, self).__init__()

        self.conv = nn.Conv2d
        self.batch_norm = nn.BatchNorm2d
        self.ngf = ngf

        self.res_block = ResidualBlock
        self.dilate_block = DilationBlock
        self.biup = UpsamplingBlock
        self.concat = ConcatTable
        self.model = self.create_model(input_nc,output_nc)

    def create_test_model(self, input_nc, output_nc):
        """
        Function which pieces together the model
        """

        model = nn.Sequential()
        ngf=self.ngf
        #model.add_module('identity',nn.Identity())
        model.add_module('res_block_1', self.res_block(output_nc))
        #model.add_module('res_block_2', self.res_block(output_nc))

        #model.add_module('tanh',nn.Tanh())
        return model
        #model.add_module('batch_9',self.batch_norm(3)) #?? why?

    def create_model(self,input_nc,output_nc):
        """
        Function which pieces together the model
        """

        model = nn.Sequential()
        ngf = self.ngf

        model.add_module('conv_1',self.dilate_block(input_nc,ngf))
        model.add_module('batch_1',self.batch_norm(ngf))
        model.add_module('norm_1',nn.ReLU(True))

        #skip connection here
        block1 = nn.Sequential()

        block1.add_module('res_block_1', self.res_block(ngf))

        block1.add_module('conv_2',self.conv(ngf,ngf*2,3,2,1))
        block1.add_module('batch_2',self.batch_norm(ngf*2))
        block1.add_module('norm_2',nn.ReLU(True))

        block1.add_module('res_block_2',self.res_block(ngf*2))

        block1.add_module('conv_3',self.conv(ngf*2,ngf*4,3,2,1))
        block1.add_module('batch_3',self.batch_norm(ngf*4))
        block1.add_module('norm_3',nn.ReLU(True))

        block1.add_module('res_block_3',self.res_block(ngf*4))

        block1.add_module('conv_4',self.conv(ngf*4,ngf*8,3,1,1))
        block1.add_module('batch_4',self.batch_norm(ngf*8))
        block1.add_module('norm_4',nn.ReLU(True))

        block1.add_module('res_block_4',self.res_block(ngf*8))
        block1.add_module('res_block_5',self.res_block(ngf*8))
        block1.add_module('res_block_6',self.res_block(ngf*8))
        block1.add_module('res_block_7',self.res_block(ngf*8))
        block1.add_module('res_block_8',self.res_block(ngf*8))

        block1.add_module('upsampl_1',self.biup(ngf*8,ngf*4,3,1,1,dil=1))
        block1.add_module('batch_5',self.batch_norm(ngf*4))
        block1.add_module('norm_5',nn.ReLU(True))
        block1.add_module('res_block_9',self.res_block(ngf*4))
        #model.add_module('res_block_10',self.res_block(ngf*4))

        block1.add_module('upsampl_2',self.biup(ngf*4,ngf*2,3,1,1,dil=1))
        block1.add_module('batch_6',self.batch_norm(ngf*2))
        block1.add_module('norm_6',nn.ReLU(True))
        block1.add_module('res_block_11',self.res_block(ngf*2))
        #model.add_module('res_block_12',self.res_block(ngf*2))
        block1.add_module('conv_7',self.conv(ngf*2,ngf,3,1,1))
        block1.add_module('batch_7',self.batch_norm(ngf))
        block1.add_module('norm_7',nn.ReLU(True))

        #block1.add_module('upsampl_3',self.biup(ngf*2,ngf,5,1,1,dil=1))
        #block1.add_module('batch_7',self.batch_norm(ngf))
        #block1.add_module('norm_7',nn.ReLU(True))

        #skip connection here
        block2 = nn.Sequential()
        block2.add_module('res_block_13',self.res_block(ngf))
        block2.add_module('res_block_14',self.res_block(ngf))
        block2.add_module('res_block_15',self.res_block(ngf))
        mlp = self.concat(block1,block2)
        model.add_module('concat',mlp)
        model.add_module('upsampl_4',self.biup(2*ngf,3,3,1,1,dil=3))
        # model.add_module('batch_8',self.batch_norm(ngf))
        # model.add_module('norm_8',nn.ReLU(True))
        model.add_module('tanh',nn.Tanh())
        # model.add_module('conv_5',self.conv(ngf,3,3,1,1))

        return model
        # model.add_module('batch_9',self.batch_norm(3)) #?? why?

    def forward(self, input):
        return self.model(input)


class UpsamplingBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel, stride, pad, dil):
        '''
        Single block of upsampling operation

        Input:
        - int input_nc    : Input number of channels
        - int output_nc   : Output number of channels
        - int kernel      : Kernel size
        - int stride	  : Stride length
        - int pad         : Padd_moduleing
        '''
        super(UpsamplingBlock, self).__init__()

        conv = nn.Conv2d
        biup = nn.UpsamplingBilinear2d

        block = nn.Sequential()
        block.add_module('conv_1',conv(input_nc, output_nc, kernel, stride, pad, dilation=dil))
        block.add_module('upsample_2',biup(scale_factor=2))

        self.biup_block = block

    def forward(self, input):
        return self.biup_block(input)


class DilationBlock(nn.Module):
    def __init__(self,input_c,output_c):
        '''
        Single block of upsampling operation

        Input:
        - int input_nc    : Input number of channels
        - int output_nc   : Output number of channels
        - int kernel      : Kernel size
        - int stride	  : Stride length
        - int pad         : Padd_moduleing
        '''
        super(DilationBlock, self).__init__()
        self.conv = nn.Conv2d
        self.batch_norm = nn.BatchNorm2d

        self.dilblock = nn.Sequential()

        self.dilblock.add_module('conv_1',self.conv(input_c,output_c,5,1,2,5))
        self.dilblock.add_module('batch_1',self.batch_norm(output_c))
        self.dilblock.add_module('norm_1',nn.ReLU(True))

        self.dilblock.add_module('conv_2',self.conv(output_c,output_c,5,1,1,5))
        self.dilblock.add_module('batch_2',self.batch_norm(output_c))
        self.dilblock.add_module('norm_2',nn.ReLU(True))

        self.dilblock.add_module('conv_3',self.conv(output_c,output_c,5,1,1,5))
        self.dilblock.add_module('batch_3',self.batch_norm(output_c))
        self.dilblock.add_module('norm_3',nn.ReLU(True))

        self.dilblock.add_module('conv_4',self.conv(output_c,output_c,3,1,1,5))
        self.dilblock.add_module('batch_4',self.batch_norm(output_c))


    def forward(self,input):
        return self.dilblock(input)#+input


class ConcatTable(nn.Module):
    def __init__(self, model1, model2):
        super(ConcatTable, self).__init__()
        self.layer1 = model1
        self.layer2 = model2

    def forward(self, x):
        y = [self.layer1(x), self.layer2(x)]
        z = torch.cat((y[0], y[1]),1)
        return z


class ResidualBlock(nn.Module):
    def __init__(self, block_size):
        '''
        Residual block for bottleneck operation

        Input:
        - int block_size : number of features in the bottleneck layer
        '''
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d
        self.batch_norm = nn.BatchNorm2d

        self.resblock = nn.Sequential()

        self.resblock.add_module('conv_1',self.conv(block_size, block_size, 3, 1, 1, 1))
        self.resblock.add_module('batch_1',self.batch_norm(block_size))
        self.resblock.add_module('norm_1',nn.ReLU(True))

        self.resblock.add_module('conv_2',self.conv(block_size, block_size, 3, 1, 1, 1))
        self.resblock.add_module('batch_2',self.batch_norm(block_size))


    def forward(self, input):
        return self.resblock(input)+input

