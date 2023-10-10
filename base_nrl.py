import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor


class MLP(nn.module):
	super(MLP,self).__init__()
	self.f1 = nn.Linear(n_input, n_hidden)
	self.f2 = nn.Linear(n_hidden, n_hidden)
	self.bn = nn.BatchNorm1d(n_out)
	self.dropout = prob
	self.init_weights()

	def init_weights():
		for i in self.modules():
			if isinstance(i, nn.Linear):
				nn.init.xavier_normal(i.weight.data)
				i.bias.data.fill(0.1)
			elif isinstance(i, nn.BatchNorm1d):
				i.weight.data.fill_(1)
				i.bias.data.zero_()
