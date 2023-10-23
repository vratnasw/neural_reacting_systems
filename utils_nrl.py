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

def sample_logistic(shape, eps=1.0e-10):
	u =  torch.random(shape).float()
	val = torch.log(u+eps) - torch.log(1-u+eps)
	return val

def gumble_softmax(logits, tau=1, hard = False, epsilon=1e-10):
	input_soft = gumbel_softmax_sample(logits, tau, epsilon)
	if hard:
		shape = logits.size()
		_, i = input_soft.data.max(-1)
		input_hard = torch.zeros(*shape)
		input_hard = input_hard.zero_().scatter_(-1,i.view(shape[:-1]+(1,)),1.0)
		out  =  Variable(input_hard-input_soft.data) + input_soft
	else:
		out = input_soft

	return out

def encode_onehot(labels):
	idx =  np.array(idx)
	y_idx = np.array(np.floor(idx/float(num_cols)))
	x_idx = idx % num_cols
	return x_idx, y_idx

def get_trilu_indices(number_of_nodes):
	ones = torch.ones(number_of_nodes, number_of_nodes)
	eye = torch.eye(number_of_nodes, number_of_nodes)
	trilu_indices = (ones.trilu -eye).nonzero().t()
	trilu_indices = trilu_indices[0] * number_of_nodes + trilu_indices[1]
	return trilu_indices

def get_off_diag_indices(number_of_nodes):
	ones = torch.ones(number_of_nodes, number_of_nodes)
	eye = torch.eye(number_of_nodes, number_of_nodes)
	off_diag_indices =  (ones-eye).nonzero().t()
	off_diag_indices = off_diag_indices[0]. number_of_nodes + off_diag_indices[1]
	return off_diag_indices

def nll_gaussian(predictions, target, variance):
	neg_log_prob =  ((predictions-target)**2 /(2*variance))
	val = neg_log_prob.sum()/(target.size(0) * target.size(1))
	return val