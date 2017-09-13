import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):

	def __init__(self,input_nc,ndf,use_sigmoid):
		super(Discriminator,self).__init__()

		self.input_nc = input_nc
		self.ndf = ndf
		self.conv = nn.Conv2d
		self.batch_norm = nn.BatchNorm2d

		self.model = self.create_discriminator(use_sigmoid)

	def create_discriminator(self,use_sigmoid): 
		
		norm_layer = nn.BatchNorm2d
		kw = 4
		ndf = self.ndf
		use_bias = 0
		#use_sigmoid =1 
		padw = int(np.ceil((kw-1)/2))
		sequence = [
			nn.Conv2d(self.input_nc, self.ndf, kernel_size=kw, stride=2, padding=padw),
			nn.LeakyReLU(0.2, True)
		]

		nf_mult = 1
		nf_mult_prev = 1
		n_layers=4

		for n in range(1, n_layers):
			 nf_mult_prev = nf_mult
			 nf_mult = min(2**n, 8)
			 sequence += [
				 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
						   kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				 norm_layer(ndf * nf_mult),
				 nn.LeakyReLU(0.2, True)
			 ]

		nf_mult_prev = nf_mult
		nf_mult = min(2**n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
					  kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]

		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

		if use_sigmoid:
			sequence += [nn.Sigmoid()]
		
		return nn.Sequential(*sequence) 

	def forward(self, input):
		return self.model(input)

