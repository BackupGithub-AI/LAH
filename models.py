import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
import gl

# DEBUG SWITCH
DEBUG_MODEL = False
# is the internet connection?
IS_NET_CONNECTION = True


class GraphConvolution(nn.Module):
	
	def __init__(self, in_features, out_features, bias=False):
		
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.Tensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.Tensor(1, 1, out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()
		
	def reset_parameters(self):
		
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)
	
	def forward(self, input, adj):
		support = torch.matmul(input, self.weight)
		output = torch.matmul(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output
	
	def __repr__(self):
		
		return self.__class__.__name__ + ' (' \
		       + str(self.in_features) + ' -> ' \
		       + str(self.out_features) + ')'


class GCNResnet(nn.Module):
	def __init__(self, option, model, num_classes, in_channel=300, t=0, adj_file=None):
		super(GCNResnet, self).__init__()
		self.state = {}
		self.state['use_gpu'] = torch.cuda.is_available()
		self.opt = option
		self.is_hash = option.HASH_TASK
		self.is_usemfb = option.IS_USE_MFB
		self.pooling_stride = option.pooling_stride
		self.features = nn.Sequential(
			model.conv1,
			model.bn1,
			model.relu,
			model.maxpool,
			model.layer1,
			model.layer2,
			model.layer3,
			model.layer4,
		)
		self.num_classes = num_classes
		self.pooling = nn.MaxPool2d(14, 14)
		self.gc1 = GraphConvolution(in_channel, 1024)
		self.gc2 = GraphConvolution(1024, 2048)
		self.relu = nn.LeakyReLU(0.2)
		_adj = gen_A(self.opt.threshold_p, num_classes, self.opt.threshold_tao, adj_file)
		self.A = Parameter(torch.from_numpy(_adj).float())
		
		self.image_normalization_mean = [0.485, 0.456, 0.406]  
		self.image_normalization_std = [0.229, 0.224, 0.225]  
		

		self.JOINT_EMB_SIZE = option.linear_intermediate
		assert self.JOINT_EMB_SIZE%self.pooling_stride==0, \
			'linear-intermediate value must can be divided exactly by sum pooling stride value!'
		if self.is_hash:
			self.out_in_tmp = int(self.JOINT_EMB_SIZE / self.pooling_stride)
		else:
			self.out_in_tmp = int(1)

		self.Linear_imgdataproj = nn.Linear(option.IMAGE_CHANNEL, self.JOINT_EMB_SIZE)  
		self.Linear_classifierproj = nn.Linear(option.CLASSIFIER_CHANNEL,self.JOINT_EMB_SIZE)
		
		if self.is_hash:
			self.hash_layer = nn.Linear(int(self.num_classes*self.out_in_tmp), option.HASH_BIT)
			self.use_tanh = nn.Tanh()

	
	def forward(self, feature, inp):
		feature = self.features(feature)
		feature = self.pooling(feature)
		feature = feature.view(feature.size(0), -1) 
		
		inp = inp[0]
		adj = gen_adj(self.A).detach()
		x = self.gc1(inp, adj)
		x = self.relu(x)
		x = self.gc2(x, adj)  
		
		x = th.transpose(x, 0, 1)
		if self.is_usemfb:
			if self.state['use_gpu']:
				x_out = torch.FloatTensor(torch.FloatStorage()).cuda()
			else:
				x_out = torch.FloatTensor(torch.FloatStorage())
			for i_row in range(int(feature.shape[0])):
				img_linear_row = self.Linear_imgdataproj(feature[i_row, :]).view(1, -1)
				if self.state['use_gpu']:
					out_row = torch.FloatTensor(torch.FloatStorage()).cuda()
				else:
					out_row = torch.FloatTensor(torch.FloatStorage())
				for col in range(int(x.shape[1])):  
					
					tmp_x = x[:, col].view(1, -1)  
					classifier_linear = self.Linear_classifierproj(tmp_x)  
					iq = torch.mul(img_linear_row, classifier_linear)  
					iq = F.dropout(iq, self.opt.DROPOUT_RATIO, training=self.training)
					iq = torch.sum(iq.view(1, self.out_in_tmp, -1), 2) 
					out_row = torch.cat((out_row,iq),1)
				x_out = torch.cat((x_out, out_row),0)
		else:
			x_out = th.matmul(feature, x)
		gl.GLOBAL_TENSOR = x_out
		
		if self.is_hash:
			hash_tmp = self.hash_layer(x_out)
			hash_code_out = self.use_tanh(hash_tmp)
			if gl.LOCAL_USE_TANH:
				hash_code_out[hash_code_out > 0] = 1
				hash_code_out[hash_code_out <= 0] = -1
			return hash_code_out
		
		return x_out
	
	def get_config_optim(self, lr, lrp):
		return [
			{'params': self.features.parameters(), 'lr': lr * lrp},
			{'params': self.gc1.parameters(), 'lr': lr},
			{'params': self.gc2.parameters(), 'lr': lr},
		]
	
	@property
	def display_model_hyperparameters(self):
		print("self.is_usetanh = ",self.is_usetanh)


def gcn_resnet101(opt, num_classes, t, pretrained=True, adj_file=None, in_channel=300):
	model = models.resnet101(pretrained=pretrained)	
	return GCNResnet(opt, model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)
