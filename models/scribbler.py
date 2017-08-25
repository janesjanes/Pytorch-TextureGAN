import torch
import torch.nn as nn


class Scribbler(nn.module):
	def __init__(self, input_nc, output_nc):
		'''
		Defines the necessary modules of the Scribbler Generator

		Input:
		- int input_nc : Input number of channels
		- int output_nc : Output number of channels
		'''

		self.conv = nn.Conv2d
		self.batch_norm = nn.BatchNorm2d

		self.model = self.create_model(input_nc,output_nc)
		self.res_block = ResidualBlock
		self.biup = UpsamplingBlock

	def create_model(input_nc,output_nc):
		'''
		Function which pieces together the model
		'''

		model = nn.Sequential

		model.add(self.conv(input_nc,ngf,3,1,1))
		model.add(self.batch_norm(ngf))
		model.add(nn.ReLU(True))

		model.add(self.res_block(ngf))

		model.add(self.conv(ngf,ngf*2,3,2,1))
		model.add(self.batch_norm(ngf*2))
		model.add(nn.ReLU(True))

		model.add(self.res_block(ngf*2))

		model.add(self.conv(ngf*2,ngf*4,3,2,1))
		model.add(self.batch_norm(ngf*4))
		model.add(nn.ReLU(True))

		model.add(self.res_block(ngf*4))

		model.add(self.conv(ngf*4,ngf*8,3,2,1))
		model.add(self.batch_norm(ngf*8))


		model.add(nn.ReLU(True))
		model.add(self.res_block(ngf*8))
		model.add(self.res_block(ngf*8))
		model.add(self.res_block(ngf*8))
		model.add(self.res_block(ngf*8))
		model.add(self.res_block(ngf*8))

		model.add(self.biup(ngf*8,ngf*4,3,1,1))
		model.add(self.batch_norm(ngf*4))
		model.add(nn.ReLU(True))
		model.add(res_block(ngf*4))
		model.add(res_block(ngf*4))

		model.add(self.biup(ngf*4,ngf*2,3,1,1))
		model.add(self.batch_norm(ngf*2))
		model.add(self.ReLU(True))
		model.add(res_block(ngf*2))
		model.add(res_block(ngf*2))

		model.add(self.biup(ngf*2,ngf,3,1,1))
		model.add(self.batch_norm(ngf))
		model.add(self.ReLU(True))
		model.add(res_block(ngf))
		model.add(self.batch_norm(ngf))

		model.add(self.res_block(ngf))
		model.add(self.conv(ngf))

		model.add(self.batch_norm(3)) #?? why?

		return model

	def forward(input):
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
		- int pad         : Padding 
		'''

		conv = nn.Conv2d 
		biup = nn.BilinearUpsampling2d

		block = nn.Sequential()
		block.add(conv(input_nc,output_nc,kernel,stride,pad))
		block.add(biup(2))

		self.biup_block = block

	def forward(input):

		return self.biup_block(input)



class ResidualBlock(nn.Module):
	def __init__(self,block_size):
		'''
		Residual block for bottleneck operation

		Input:
		- int block_size : number of features in the bottleneck layer
		'''

		self.conv = nn.Conv2d
		self.batch_norm = nn.BatchNorm2d

		resblock = nn.Sequential()

		resblock.add(self.conv(block_size,block_size,3,1,1,1))
		resblock.add(self.batch_norm(block_size))
		resblock.add(nn.ReLU(true))

		resblock.add(self.conv(block_size,block_size,3,1,1,1))
		resblock.add(self.batch_norm(block_size))

		self.model = resblock

	def forward(input):
		return self.model(input)+input

