import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def softmax(x, axis=1):
	transpose_x =  x.transpose(axis,0).contiguous()
	softmax = F.softmax(transpose_x)
	softmax_new = softmax.transpose(axis, 0)


def sample_gumbel(shape, eps=1.0e-10):

	u = torch.rand(shape).float()
	sample =  -torch.log(eps-torch.log(U+eps))
	return sample

def gumbel_softmax_sample(logits, tau=1, eps=1.0e-10):
	gumbel_noise =  sample_gumbel(logits.size(), eps = eps)
	x = logits + Variable(gumbel_noise)
	x = softmax(x/tau, axis=-1)
	return x