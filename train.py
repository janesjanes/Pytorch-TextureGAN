import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.dataset as dataset
import visdom

class wrap_tensor():
	def __init__(self, tensor, to_gpu):

		self.tensor = tensor
		self.to_gpu = to_gpu

	def __call__(self,input):

		if to_gpu:
			self.tensor=self.tensor.gpu()
		return Variable(self.tensor)

