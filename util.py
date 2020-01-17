import math, os, sys
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import numpy as np
import torch.nn.functional as F
from cauchy_hash import *



class Warp(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		
		self.size = int(size)
		self.interpolation = interpolation
	
	def __call__(self, img):
		
		return img.resize((self.size, self.size), self.interpolation)
	
	def __str__(self):
		return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
		                                                                                        interpolation=self.interpolation)


class MultiScaleCrop(object):
	'''
	Get many images which have different scale
	'''
	
	def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
		
		self.scales = scales if scales is not None else [1, 875, .75, .66]
		self.max_distort = max_distort
		self.fix_crop = fix_crop
		self.more_fix_crop = more_fix_crop
		self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
		self.interpolation = Image.BILINEAR  
	
	def __call__(self, img):
		
		im_size = img.size
		crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
		crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
		ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
		return ret_img_group
	
	def _sample_crop_size(self, im_size):
		image_w, image_h = im_size[0], im_size[1]
		
		base_size = min(image_w, image_h)
		crop_sizes = [int(base_size * x) for x in self.scales]      
		crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
		crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]
		
		pairs = []
		for i, h in enumerate(crop_h):
			for j, w in enumerate(crop_w):
				if abs(i - j) <= self.max_distort:
					pairs.append((w, h))
		
		crop_pair = random.choice(pairs)
		if not self.fix_crop:
			w_offset = random.randint(0, image_w - crop_pair[0])
			h_offset = random.randint(0, image_h - crop_pair[1])
		else:
			w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
		
		return crop_pair[0], crop_pair[1], w_offset, h_offset
	
	def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
		offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
		return random.choice(offsets)
	
	@staticmethod
	def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
		
		w_step = (image_w - crop_w) // 4
		h_step = (image_h - crop_h) // 4
		
		ret = list()
		ret.append((0, 0))  
		ret.append((4 * w_step, 0))  
		ret.append((0, 4 * h_step))  
		ret.append((4 * w_step, 4 * h_step))  
		ret.append((2 * w_step, 2 * h_step))  
		
		if more_fix_crop:
			ret.append((0, 2 * h_step))  
			ret.append((4 * w_step, 2 * h_step)) 
			ret.append((2 * w_step, 4 * h_step))  
			ret.append((2 * w_step, 0 * h_step))  
			
			ret.append((1 * w_step, 1 * h_step))  
			ret.append((3 * w_step, 1 * h_step))  
			ret.append((1 * w_step, 3 * h_step))  
			ret.append((3 * w_step, 3 * h_step))  
		
		return ret
	
	def __str__(self):
		return self.__class__.__name__


def download_url(url, destination=None, progress_bar=True):
	
	def my_hook(t):
		last_b = [0]
		
		def inner(b=1, bsize=1, tsize=None):
			if tsize is not None:
				t.total = tsize
			if b > 0:
				t.update((b - last_b[0]) * bsize)
			last_b[0] = b
		
		return inner
	
	if progress_bar:
		with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
			filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
	else:
		filename, _ = urlretrieve(url, filename=destination)


class AveragePrecisionMeter(object):
	
	def __init__(self, difficult_examples=False):
		super(AveragePrecisionMeter, self).__init__()
		self.reset()
		self.difficult_examples = difficult_examples
	
	def reset(self):
		"""Resets the meter with empty member variables"""
		self.scores = torch.FloatTensor(torch.FloatStorage()) 
		self.targets = torch.LongTensor(torch.LongStorage())  
	
	def add(self, output, target):
		
		if not torch.is_tensor(output):
			output = torch.from_numpy(output)
		if not torch.is_tensor(target):
			target = torch.from_numpy(target)
		
		if output.dim() == 1:
			output = output.view(-1, 1) 
		else:
			assert output.dim() == 2, \
				'wrong output size (should be 1D or 2D with one column \
				per class)'
		if target.dim() == 1:
			target = target.view(-1, 1)
		else:
			assert target.dim() == 2, \
				'wrong target size (should be 1D or 2D with one column \
				per class)'
		if self.scores.numel() > 0:  
			assert target.size(1) == self.targets.size(1), \
				'dimensions for output should match previously added examples.'
		
		if self.scores.storage().size() < self.scores.numel() + output.numel():  
			new_size = math.ceil(self.scores.storage().size() * 1.5)
			self.scores.storage().resize_(int(new_size + output.numel()))
			self.targets.storage().resize_(int(new_size + output.numel()))
		
		offset = self.scores.size(0) if self.scores.dim() > 0 else 0
		self.scores.resize_(offset + output.size(0), output.size(1))
		self.targets.resize_(offset + target.size(0), target.size(1))
		self.scores.narrow(0, offset, output.size(0)).copy_(output)
		self.targets.narrow(0, offset, target.size(0)).copy_(target)
		
	
	def value(self):
		"""Returns the model's average precision for each class
		Return:
			ap (FloatTensor): 1xK tensor, with avg precision for each class k
		"""
		if self.scores.numel() == 0:
			return 0
		
		ap = torch.zeros(self.scores.size(1))
		rg = torch.arange(1, self.scores.size(0)).float()
		for k in range(self.scores.size(1)):  
			
			scores = self.scores[:, k]  
			targets = self.targets[:, k] 
			ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
		return ap
	
	@staticmethod
	def average_precision(output, target, difficult_examples=True):
		sorted, indices = torch.sort(output, dim=0, descending=True)
		pos_count = 0.
		total_count = 0.
		precision_at_i = 0.
		
		for i in indices:
			label = target[i]
			if difficult_examples and label == 0:
				continue
			if label == 1:
				pos_count += 1
			total_count += 1
			if label == 1:
				precision_at_i += pos_count / total_count
		precision_at_i /= pos_count
		return precision_at_i
	
	def overall(self):
		if self.scores.numel() == 0:
			return 0
		scores = self.scores.cpu().numpy()
		targets = self.targets.cpu().numpy()
		targets[targets == -1] = 0
		return self.evaluation(scores, targets)
	
	def overall_topk(self, k):
		targets = self.targets.cpu().numpy()
		targets[targets == -1] = 0
		n, c = self.scores.size()
		scores = np.zeros((n, c)) - 1
		index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
		tmp = self.scores.cpu().numpy()
		for i in range(n):
			for ind in index[i]:
				scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
		return self.evaluation(scores, targets)
	
	def evaluation(self, scores_, targets_):
		n, n_class = scores_.shape
		Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
		for k in range(n_class):
			scores = scores_[:, k]
			targets = targets_[:, k]
			targets[targets == -1] = 0
			Ng[k] = np.sum(targets == 1)
			Np[k] = np.sum(scores >= 0)
			Nc[k] = np.sum(targets * (scores >= 0))
		Np[Np == 0] = 1
		OP = np.sum(Nc) / np.sum(Np)
		OR = np.sum(Nc) / np.sum(Ng)
		OF1 = (2 * OP * OR) / (OP + OR)
		
		CP = np.sum(Nc / Np) / n_class
		CR = np.sum(Nc / Ng) / n_class
		CF1 = (2 * CP * CR) / (CP + CR)
		return OP, OR, OF1, CP, CR, CF1


class HashAveragePrecisionMeter(AveragePrecisionMeter):
	def __init__(self, difficult_examples=False):
		AveragePrecisionMeter.__init__(self, difficult_examples)
	
	def loss_value(self):
		"""Returns the model's average precision for each class
		Return:
			ap (FloatTensor): 1xK tensor, with avg precision for each class k
		"""
		if self.scores.numel() == 0:
			return 0
		target_gt = self.targets
		all_output = self.scores
		
		target_gt[target_gt >= 0] = 1
		target_gt[target_gt < 0] = 0
		calcloss = CauchyLoss(sij_type='IOU', normed=True)
		epoch_loss = calcloss.forward(target_gt, all_output)
		
		return epoch_loss
	
	def batch_loss_value(self, batch_target, batch_output):
		"""Returns the model's average precision for each class
		Return:
			ap (FloatTensor): 1xK tensor, with avg precision for each class k
		"""
		target_gt = batch_target
		all_output = batch_output
		
		target_gt[target_gt >= 0] = 1
		target_gt[target_gt < 0] = 0
		calcloss = CauchyLoss(sij_type='IOU', normed=True)
		batch_loss = calcloss.forward(target_gt, all_output)
		
		return batch_loss


def gen_A(p, num_classes, t, adj_file):
	'''
	generate the adjecent matrix
	:param opt: get command parameters
	:param num_classes: the amount of classes
	:param t:
	:param adj_file:    word embeding matrix???
	:return:
	'''
	import pickle
	with open(adj_file, "rb") as f:
		result = pickle.load(f)
	_adj = result['adj']
	_nums = result['nums']
	_nums = _nums[:, np.newaxis] 
	_adj = _adj / _nums
	_adj[_adj < t] = 0  
	_adj[_adj >= t] = 1
	
	_adj = _adj * p / (_adj.sum(0, keepdims=True) + 1e-6)
	_adj = _adj + np.identity(num_classes, np.int)
	return _adj


def gen_adj(A):
	'''
	:param A:
	:return:
	'''
	D = torch.pow(A.sum(1).float(), -0.5)  
	D = torch.diag(D)
	adj = torch.matmul(torch.matmul(A, D).t(), D) 
	return adj


