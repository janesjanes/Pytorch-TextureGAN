import torch
import torch.nn as nn


class Scribbler(nn.Module):
	def __init__(self, input_nc, output_nc, ngf):
		'''
		Defines the necessary modules of the Scribbler Generator

		Input:
		- int input_nc : Input number of channels
		- int output_nc : Output number of channels
		'''
		super(Scribbler, self).__init__()

		self.conv = nn.Conv2d
		self.batch_norm = nn.BatchNorm2d
		self.ngf = ngf

		self.res_block = ResidualBlock
		self.biup = UpsamplingBlock
		self.model = self.create_model(input_nc,output_nc)


	def create_model(self,input_nc,output_nc):
		'''
		Function which pieces together the model
		'''
            
		model = nn.Sequential()
		ngf=self.ngf

		model.add_module('conv_1',self.conv(input_nc,ngf,3,1,1))
		model.add_module('batch_1',self.batch_norm(ngf))
		model.add_module('norm_1',nn.ReLU(True))

		model.add_module('res_block_1', self.res_block(ngf))

		model.add_module('conv_2',self.conv(ngf,ngf*2,3,2,1))
		model.add_module('batch_2',self.batch_norm(ngf*2))
		model.add_module('norm_2',nn.ReLU(True))

		model.add_module('res_block_2',self.res_block(ngf*2))

		model.add_module('conv_3',self.conv(ngf*2,ngf*4,3,2,1))
		model.add_module('batch_3',self.batch_norm(ngf*4))
		model.add_module('norm_3',nn.ReLU(True))

		model.add_module('res_block_3',self.res_block(ngf*4))

		model.add_module('conv_4',self.conv(ngf*4,ngf*8,3,2,1))
		model.add_module('batch_4',self.batch_norm(ngf*8))
		model.add_module('norm_4',nn.ReLU(True))
		
		model.add_module('res_block_4',self.res_block(ngf*8))
		model.add_module('res_block_5',self.res_block(ngf*8))
		model.add_module('res_block_6',self.res_block(ngf*8))
		model.add_module('res_block_7',self.res_block(ngf*8))
		model.add_module('res_block_8',self.res_block(ngf*8))

		model.add_module('upsampl_1',self.biup(ngf*8,ngf*4,3,1,1))
		model.add_module('batch_5',self.batch_norm(ngf*4))
		model.add_module('norm_5',nn.ReLU(True))
		model.add_module('res_block_9',self.res_block(ngf*4))
		model.add_module('res_block_10',self.res_block(ngf*4))

		model.add_module('upsampl_2',self.biup(ngf*4,ngf*2,3,1,1))
		model.add_module('batch_6',self.batch_norm(ngf*2))
		model.add_module('norm_6',nn.ReLU(True))
		model.add_module('res_block_11',self.res_block(ngf*2))
		model.add_module('res_block_12',self.res_block(ngf*2))

		model.add_module('upsampl_3',self.biup(ngf*2,ngf,3,1,1))
		model.add_module('batch_7',self.batch_norm(ngf))
		model.add_module('norm_7',nn.ReLU(True))
		model.add_module('res_block_13',self.res_block(ngf))
		model.add_module('batch_8',self.batch_norm(ngf))

		model.add_module('res_block_14',self.res_block(ngf))
		model.add_module('conv_5',self.conv(ngf,3,3,1,1))

		model.add_module('batch_9',self.batch_norm(3)) #?? why?

		return model

	def forward(self,input):
		return self.model(input)





class UpsamplingBlock(nn.Module):
	def __init__(self,input_nc,output_nc,kernel,stride,pad):
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
		block.add_module('conv_1',conv(input_nc,output_nc,kernel,stride,pad))
		block.add_module('upsample_2',biup(scale_factor=2))

		self.biup_block = block

	def forward(self,input):

		return self.biup_block(input)



class ResidualBlock(nn.Module):
	def __init__(self,block_size):
		'''
		Residual block for bottleneck operation

		Input:
		- int block_size : number of features in the bottleneck layer
		'''
		super(ResidualBlock, self).__init__()
		self.conv = nn.Conv2d
		self.batch_norm = nn.BatchNorm2d

		self.resblock = nn.Sequential()

		self.resblock.add_module('conv_1',self.conv(block_size,block_size,3,1,1,1))
		self.resblock.add_module('batch_1',self.batch_norm(block_size))
		self.resblock.add_module('norm_1',nn.ReLU(True))

		self.resblock.add_module('conv_2',self.conv(block_size,block_size,3,1,1,1))
		self.resblock.add_module('batch_2',self.batch_norm(block_size))


	def forward(self,input):
		return self.resblock(input)+input

