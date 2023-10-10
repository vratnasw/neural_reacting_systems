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
	self.dropout_prob = prob
	self.init_weights()

	def init_weights():
		for i in self.modules():
			if isinstance(i, nn.Linear):
				nn.init.xavier_normal(i.weight.data)
				i.bias.data.fill(0.1)
			elif isinstance(i, nn.BatchNorm1d):
				i.weight.data.fill_(1)
				i.bias.data.zero_()

	def batch_norm(self, inputs):
		x = inputs.view(inputs.size(0) * inputs.size(1), -1)
		x = self.bn(x)
		return x.view(inputs.size(0) * inputs.size(1), -1)


	def forward(self, inputs):
		x = F.elu(self.f1(inputs))
		x = F.dropout(x, self.dropout_prob)
		x = F.elu(self.f2(x))
		x = self.bn(x)
		return x