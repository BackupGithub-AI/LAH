import os, sys, datetime
import shutil, pickle, sys
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from util import *
import numpy as np
from tqdm import tqdm, trange
from post_process import *
import re, gl

DEBUG_FUNC_ON_END_BATCH = False

ENGINE_DEBUG = False
store_fc_data = False

conver_thre = [0.0001, 0.0001, 0.0001]

tqdm.monitor_interval = 0


class Engine(object):
	def __init__(self, state={}):
		
		self.state = state  
		
		if self._state('use_gpu') is None:
			self.state['use_gpu'] = torch.cuda.is_available()
		
		if self._state('image_size') is None:
			self.state['image_size'] = 224
		
		if self._state('batch_size') is None:
			self.state['batch_size'] = 64
		
		if self._state('workers') is None:
			self.state['workers'] = 25
		
		if self._state('device_ids') is None:
			self.state['device_ids'] = None
		
		if self._state('evaluate') is None:
			self.state['evaluate'] = False
		
		if self._state('start_epoch') is None:
			self.state['start_epoch'] = 0
		
		if self._state('max_epochs') is None:
			self.state['max_epochs'] = 100
		
		if self._state('epoch_step') is None:
			self.state['epoch_step'] = []
		
		if self._state("train_loss_set") is None:
			self.state["train_loss_set"] = []
		if self._state("test_loss_set") is None:
			self.state["test_loss_set"] = []
		if self._state('convergency_flag') is None:
			self.state['convergency_flag'] = False
		if self._state("loss_difference") is None:
			self.state["loss_difference"] = []
		if self._state("start_time") is None:
			now_time = datetime.datetime.now()
			self.state["start_time"] = nt = datetime.datetime.now().strftime('%Y%m%d%H%M')
		
		if self._state("one_epoch_cauchy_loss") == None:
			self.state['one_epoch_cauchy_loss'] = []
		
		if self._state('model') is None:
			self.state['model'] = []
		if self._state('criterion') is None:
			self.state['criterion'] = []
		
		if self._state("all_param") is None:
			self.state["all_param"] = []
		
		if self._state('better_than_before') is None:
			self.state['better_than_before'] = False
		if self._state('temp_dir') is None:
			self.state['temp_dir'] = ""
		if self._state('destination') is None:
			self.state['destination'] = ""
		if self._state('before_fc_tmp') is None:
			self.state['before_fc_tmp'] = ""
		if self._state('before_fc_destination') is None:
			self.state['before_fc_destination'] = ""
		if self._state('convergence_point') is None:
			self.state['convergence_point'] = self.state['max_epochs']
		
		self.state['meter_loss'] = tnt.meter.AverageValueMeter()
		self.state['batch_time'] = tnt.meter.AverageValueMeter()
		self.state['data_time'] = tnt.meter.AverageValueMeter()
		if self._state('use_pb') is None:
			self.state['use_pb'] = True
		if self._state('print_freq') is None:
			self.state['print_freq'] = 0
		
		if self._state('hashcode_store') is None:
			self.state['hashcode_store'] = torch.CharTensor(torch.CharStorage())
		if self._state('target_store') is None:
			self.state['target_store'] = torch.CharTensor(torch.CharStorage())
		if self._state('selectimg_store') is None:
			self.state['selectimg_store'] = []
	
	def _state(self, name):
		
		if name in self.state:
			return self.state[name]
	
	def _gpu_cpu(self, input):
		if th.is_tensor(input):
			if self.state['use_gpu']:
				return input.cuda()
		return input
	
	def _wbin_pkl(self, file_dir, content):
		'''write content into .pkl file with bin format'''
		with open(str(file_dir), 'ab') as fi:
			pickle.dump(content, fi)
	
	def _rbin_pkl(self, file_dir):
		'''read .pkl file and return the content'''
		with open(str(file_dir), 'rb') as fi:
			content = pickle.load(fi, encoding='bytes')
		return content
	
	def _deleterow(self, ts, idx):
		'''
		delete a row in tensor
		:param idx:
		:return:
		'''
		ts = ts[th.arange(ts.size(0)) != idx]
		print(ts)
	
	def _continue_load(self):
		pass
	
	def _get_all_filename(self, ):
		tmp = self.state['hashcode_pool'].split('/')[:-1]
		prefix = '/'.join(tmp) + '/'
		pool = []
		print(os.walk(prefix))
		for root, dirs, files in os.walk(prefix):
			pool = files
			break
		return pool
	
	@staticmethod
	def calc_mean_var(nlist):
		"""
		calculate the mean and var of a list which named "nlist"
		:param nlist:
		:return:
		"""
		N = float(len(nlist))
		narray = np.array(nlist)
		sum1 = float(narray.sum())
		narray2 = narray * narray
		sum2 = float(narray2.sum())
		mean = sum1 / N
		var = sum2 / N - mean ** 2  
		
		return mean, var
	
	def is_convergency(self):
		'''
		:return:
		'''
		win_size = 3  
		now_cap = len(self.state["train_loss_set"])
		if now_cap >= 23:
			s_value_last = self.state['train_loss_set'][int(0 - win_size):]
			s_aver_last, _ = self.calc_mean_var(s_value_last)
			s_value_penult = self.state['train_loss_set'][int(0 - 2 * win_size):-win_size]
			s_aver_penult, _ = self.calc_mean_var(s_value_penult)
			s_value_thirdlast = self.state['train_loss_set'][int(0 - 3 * win_size):-2 * win_size]
			s_aver_thirdlast, _ = self.calc_mean_var(s_value_thirdlast)
			
			diff1_2 = abs(s_aver_last - s_aver_penult)
			diff2_3 = abs(s_aver_penult - s_aver_thirdlast)
			diff1_3 = abs(s_aver_thirdlast - s_aver_last)
			self.state['loss_difference'].extend([":", diff1_2, diff2_3, diff1_3])
			if diff1_2 <= conver_thre[0] and diff2_3 <= conver_thre[1] and diff1_3 <= conver_thre[2]:
				print("\n***I think the trainning processing is convergency***")
				print("$$$$$$$$$$$$$$$$$$$$$$$$$ Lots and lots of dollars $$$$$$$$$$$$$$$$$$$$$$$")
				self.state["convergency_flag"] = True
			else:
				self.state['convergency_flag'] = False
		
		else:
			self.state['convergency_flag'] = False
	
	def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
		self.state['meter_loss'].reset()
		self.state['batch_time'].reset()
		self.state['data_time'].reset()
	
	def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
		
		loss = self.state['meter_loss'].value()[0]
		if display:
			if training:
				print('Epoch: [{0}]\t'
				      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
			else:
				print('Test: \t Loss {loss:.4f}'.format(loss=loss))
		return loss
	
	def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
		pass
	
	def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
		self.state['loss_batch'] = self.state['loss'].item()
		self.state['meter_loss'].add(self.state['loss_batch'])
		
		if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
			loss = self.state['meter_loss'].value()[0]
			batch_time = self.state['batch_time'].value()[0]
			data_time = self.state['data_time'].value()[0]
			if training:
				print('Epoch: [{0}][{1}/{2}]\t'
				      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
				      'Data {data_time_current:.3f} ({data_time:.3f})\t'
				      'Loss {loss_current:.4f} ({loss:.4f})'.format(
					self.state['epoch'], self.state['iteration'], len(data_loader),
					batch_time_current=self.state['batch_time_current'],
					batch_time=batch_time, data_time_current=self.state['data_time_batch'],
					data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
			else:
				print('Test: [{0}/{1}]\t'
				      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
				      'Data {data_time_current:.3f} ({data_time:.3f})\t'
				      'Loss {loss_current:.4f} ({loss:.4f})'.format(
					self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
					batch_time=batch_time, data_time_current=self.state['data_time_batch'],
					data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
	
	def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
		input_var = torch.autograd.Variable(self.state['input'])
		target_var = torch.autograd.Variable(self.state['target'])
		
		if not training:
			input_var.volatile = True
			target_var.volatile = True
		
		self.state['output'] = model(input_var)
		self.state['loss'] = criterion(self.state['output'], target_var)
		
		if training:
			optimizer.zero_grad()
			self.state['loss'].backward()
			optimizer.step()
	
	def init_learning(self, model, criterion):
		'''
		init some state value : train_transform or val_transform,
		:param model:
		:param criterion:
		:return:
		'''
		self.state['model'] = model
		if self._state('train_transform') is None:  
			normalize = transforms.Normalize(mean=model.image_normalization_mean,
			                                 std=model.image_normalization_std)
			self.state['train_transform'] = transforms.Compose([
				MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			])
		
		if self._state('val_transform') is None:
			normalize = transforms.Normalize(mean=model.image_normalization_mean,
			                                 std=model.image_normalization_std)
			self.state['val_transform'] = transforms.Compose([
				Warp(self.state['image_size']),
				transforms.ToTensor(),
				normalize,
			])
		self.state['best_score'] = 0
		if self.state['HASH_TASK']:
			self.state['best_score'] = 1000000000
			if os.path.exists(self.state['hashcode_pool']):
				os.remove(self.state['hashcode_pool'])
	
	def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):
		
		self.init_learning(model, criterion)
		
		train_dataset.transform = self.state['train_transform']
		train_dataset.target_transform = self._state('train_target_transform')
		val_dataset.transform = self.state['val_transform']
		val_dataset.target_transform = self._state('val_target_transform')
		
		train_loader = torch.utils.data.DataLoader(train_dataset,
		                                           batch_size=self.state['batch_size'], shuffle=True,
		                                           num_workers=self.state['workers'])
		val_loader = torch.utils.data.DataLoader(val_dataset,
		                                         batch_size=self.state['batch_size'], shuffle=False,
		                                         num_workers=self.state['workers'])
		
		if self._state('resume') is not None:
			if os.path.isfile(self.state['resume']):
				print("=> loading checkpoint '{}'".format(self.state['resume']))
				checkpoint = torch.load(self.state['resume'])
				self.state['start_epoch'] = checkpoint['epoch']
				self.state['best_score'] = checkpoint['best_score']
				model.load_state_dict(checkpoint['state_dict'])
				print("=> loaded checkpoint '{}' (epoch {})"
				      .format(self.state['evaluate'], checkpoint['epoch']))
			else:
				print("=> no checkpoint found at '{}'".format(self.state['resume']))
		
		if self.state['use_gpu']:
			train_loader.pin_memory = True
			val_loader.pin_memory = True
			cudnn.benchmark = True
			
			model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
			
			criterion = criterion.cuda()
		
		if self.state['evaluate']:
			self.validate(val_loader, model, criterion) 
			return
		
		# TODO define optimizer
		
		for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
			self.state['epoch'] = epoch
			lr = self.adjust_learning_rate(optimizer)
			
			self.train(train_loader, model, criterion, optimizer, epoch)  
			prec1 = self.validate(val_loader, model, criterion)  
			is_best = prec1 > self.state['best_score']
			self.state['best_score'] = max(prec1, self.state['best_score'])
			self.save_checkpoint({
				'epoch': epoch + 1,
				'arch': self._state('arch'),
				'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
				'best_score': self.state['best_score'],
			}, is_best)
			
			print(' *** best={best:.3f} \n'.format(best=self.state['best_score']))
		
		
		return self.state['best_score']
	
	def train(self, data_loader, model, criterion, optimizer, epoch):
		
		model.train()
		
		self.state['one_epoch_cauchy_loss'].clear()
		
		self.on_start_epoch(True, model, criterion, data_loader, optimizer)
		
		if self.state['use_pb']:
			data_loader = tqdm(data_loader, desc=str("(" + str(epoch) + ')Training')) 
		
		end = time.time()
		for i, (input, target) in enumerate(data_loader):
			self.state['iteration'] = i
			self.state['data_time_batch'] = time.time() - end
			self.state['data_time'].add(self.state['data_time_batch'])
			
			self.state['input'] = input  
			self.state['target'] = target  
			self.on_start_batch(True, model, criterion, data_loader, optimizer)
			
			if self.state['use_gpu']:
				self.state['target'] = self.state['target'].cuda(async=True)
			
			self.on_forward(True, model, criterion, data_loader, optimizer)
			
			self.state['batch_time_current'] = time.time() - end
			self.state['batch_time'].add(self.state['batch_time_current'])
			end = time.time()
			self.on_end_batch(True, model, criterion, data_loader, optimizer)
		
		self.on_end_epoch(True, model, criterion, data_loader, optimizer) 
	
	def validate(self, data_loader, model, criterion):
		model.eval()
		
		self.on_start_epoch(False, model, criterion, data_loader)
		
		if self.state['use_pb']:
			data_loader = tqdm(data_loader, desc='Test')
		
		end = time.time()
		for i, (input, target) in enumerate(data_loader):
			self.state['iteration'] = i
			self.state['data_time_batch'] = time.time() - end
			self.state['data_time'].add(self.state['data_time_batch'])
			
			self.state['input'] = input
			self.state['target'] = target
			
			self.on_start_batch(False, model, criterion, data_loader)
			
			if self.state['use_gpu']:
				self.state['target'] = self.state['target'].cuda(async=True)
			
			self.on_forward(False, model, criterion, data_loader)
			
			self.state['batch_time_current'] = time.time() - end
			self.state['batch_time'].add(self.state['batch_time_current'])
			end = time.time()
			self.on_end_batch(False, model, criterion, data_loader)
		
		score = self.on_end_epoch(False, model, criterion, data_loader) 
		
		return score
	
	def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
		if self._state('save_model_path') is not None:
			filename_ = filename
			filename = os.path.join(self.state['save_model_path'], filename_)
			if not os.path.exists(self.state['save_model_path']):
				os.makedirs(self.state['save_model_path'])
		print('save model {filename}\n'.format(filename=filename))
		torch.save(state, filename)
		if is_best:
			filename_best = 'model_best.pth.tar'
			if self._state('save_model_path') is not None:
				filename_best = os.path.join(self.state['save_model_path'], filename_best)
			shutil.copyfile(filename, filename_best)
			if self._state('save_model_path') is not None:
				if self._state('filename_previous_best') is not None:
					os.remove(self._state('filename_previous_best'))
				filename_best = os.path.join(self.state['save_model_path'],
				                             'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
				shutil.copyfile(filename, filename_best)
				self.state['filename_previous_best'] = filename_best
	
	def adjust_learning_rate(self, optimizer):
		"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
		lr_list = []
		decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * decay
			lr_list.append(param_group['lr'])
		return np.unique(lr_list)

class MultiLabelMAPEngine(Engine):
	def __init__(self, state):
		Engine.__init__(self, state)
		if self._state('difficult_examples') is None:
			self.state['difficult_examples'] = False
		if self.state['HASH_TASK']:
			self.state['ap_meter'] = HashAveragePrecisionMeter(self.state['difficult_examples'])
		
		else:
			self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])
	
	def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
		Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
		self.state['ap_meter'].reset()
	
	def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
		
		map = 100 * self.state['ap_meter'].value().mean()  
		loss = self.state['meter_loss'].value()[0]
		OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
		OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
		
		if display:
			if training:
				print('Epoch: [{0}]\t'
				      'Loss {loss:.4f}\t'
				      'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
				print('OP: {OP:.4f}\t'
				      'OR: {OR:.4f}\t'
				      'OF1: {OF1:.4f}\t'
				      'CP: {CP:.4f}\t'
				      'CR: {CR:.4f}\t'
				      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
				self.state["train_loss_set"].append(float(loss))
			
			else:
				print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
				print('OP: {OP:.4f}\t'
				      'OR: {OR:.4f}\t'
				      'OF1: {OF1:.4f}\t'
				      'CP: {CP:.4f}\t'
				      'CR: {CR:.4f}\t'
				      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
				print('OP_3: {OP:.4f}\t'
				      'OR_3: {OR:.4f}\t'
				      'OF1_3: {OF1:.4f}\t'
				      'CP_3: {CP:.4f}\t'
				      'CR_3: {CR:.4f}\t'
				      'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))
				self.state["test_loss_set"].append(float(loss))
			
			self.is_convergency()
		
		return map
	
	def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
		self.state['target_gt'] = self.state['target'].clone()
		self.state['target'][self.state['target'] == 0] = 1
		self.state['target'][self.state['target'] == -1] = 0
		
		input = self.state['input']
		self.state['input'] = input[0]
		self.state['name'] = input[1]
	
	def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
		'''
		store the one batch output data and the corresponding ground truth labels data
		:param training:
		:param model:
		:param criterion:
		:param data_loader:
		:param optimizer:
		:param display:
		:return:
		'''
		
		Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)
		self.state['ap_meter'].add(self.state['output'].data,
		                           self.state['target_gt'])  
		
		if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
			loss = self.state['meter_loss'].value()[0]
			batch_time = self.state['batch_time'].value()[0]
			data_time = self.state['data_time'].value()[0]
			if training:
				print('Epoch: [{0}][{1}/{2}]\t'
				      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
				      'Data {data_time_current:.3f} ({data_time:.3f})\t'
				      'Loss {loss_current:.4f} ({loss:.4f})'.format(
					self.state['epoch'], self.state['iteration'], len(data_loader),
					batch_time_current=self.state['batch_time_current'],
					batch_time=batch_time, data_time_current=self.state['data_time_batch'],
					data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
			else:
				print('Test: [{0}/{1}]\t'
				      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
				      'Data {data_time_current:.3f} ({data_time:.3f})\t'
				      'Loss {loss_current:.4f} ({loss:.4f})'.format(
					self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
					batch_time=batch_time, data_time_current=self.state['data_time_batch'],
					data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))


class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
	def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
		feature_var = torch.autograd.Variable(self.state['feature']).float()
		target_var = torch.autograd.Variable(self.state['target']).float()
		inp_var = torch.autograd.Variable(self.state['input']).float().detach()  
		if not training:
			
			with torch.no_grad():
				feature_var = torch.autograd.Variable(self.state['feature']).float()
				target_var = torch.autograd.Variable(self.state['target']).float()
				inp_var = torch.autograd.Variable(self.state['input']).float()
		
		self.state['output'] = model(feature_var, inp_var)
		
		if self.state['HASH_TASK'] == False:
			self.state['loss'] = criterion(self.state['output'], target_var) 
		else:
			self.state['loss'] = criterion(self.state['target'], self.state['output'])
		
		if training:
			optimizer.zero_grad()
			self.state['loss'].backward()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) 
			optimizer.step()
	
	def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
		
		self.state['target_gt'] = self.state['target'].clone()  
		self.state['target'][self.state['target'] == 0] = 1
		self.state['target'][self.state['target'] == -1] = 0
		
		input = self.state['input']
		self.state['feature'] = input[0]
		self.state['out'] = input[1]
		self.state['input'] = input[2]
	
	def display_all_loss(self):
		if self.state['HASH_TASK'] == False:
			print("\nTrain loss set is:\n", self.state["train_loss_set"])
			print()
			print("Test loss set is:\n", self.state["test_loss_set"])
		else:
			print('\nThe convergency point at the epoch({0})\n'.format(self.state['convergence_point']))


class GCNMultiLabelHashEngine(GCNMultiLabelMAPEngine):
	def reset(self):
		if self._state('hash_code') is None:
			self.state['hash_code'] = torch.IntTensor(torch.IntStorage())
		if self._state('target_for_hash') is None:
			self.state['target_for_hash'] = torch.IntTensor(torch.IntStorage())
		if self._state('select_img') is None:
			self.state['select_img'] = []
		
		if self._state('hash_mAP') is None:
			self.state['hash_mAP'] = []
		if self._state('mean_hash_mAP') is None:
			self.state['mean_hash_mAP'] = 0
		
		gl.GLOBAL_TENSOR = torch.FloatTensor(torch.FloatStorage())
	
	def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
		feature_var = torch.autograd.Variable(self.state['feature']).float()
		target_var = torch.autograd.Variable(self.state['target']).float()
		inp_var = torch.autograd.Variable(self.state['input']).float().detach()  
		if not training:
			with torch.no_grad():
				feature_var = torch.autograd.Variable(self.state['feature']).float()
				target_var = torch.autograd.Variable(self.state['target']).float()
				inp_var = torch.autograd.Variable(self.state['input']).float()

		
		gl.LOCAL_USE_TANH = False if training else True
		
		self.state['output'] = model(feature_var, inp_var)
		self.state['loss'] = criterion(self.state['target'], self.state['output'])
		if training == False:
			dic_temp = {"img_name": self.state['out'],
			            "target": self.state['target'].cpu(),
			            'output': self.state['output'].cpu()}
			
			self._wbin_pkl(self.state['temp_dir'], dic_temp)
			
			if store_fc_data:
				bf_temp = {"img_name": self.state['out'],
				           "target": self.state['target'].cpu(),
				           'output': gl.GLOBAL_TENSOR.cpu()}
				self._wbin_pkl(self.state['before_fc_tmp'], bf_temp)
		
		if training:
			optimizer.zero_grad()
			self.state['loss'].backward()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) 
			optimizer.step()
	
	def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
		
		loss = self.state['ap_meter'].batch_loss_value(self.state['target_gt'], self.state['output'])
		if th.is_tensor(loss):
			if loss.numel() == 1:
				loss = loss.item()
		if display:
			if training:
				print("Training loss in one batch: {loss}".format(loss=loss))
			else:
				print("Test loss in one batch: {loss}".format(loss=loss))
		self.state['one_epoch_cauchy_loss'].append(loss)
	
	def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
		loss, _ = self.calc_mean_var(self.state['one_epoch_cauchy_loss']) 
		if display:
			if training:
				print('Training Epoch({0}):\taverage Loss in this epoch is {loss}\n'. \
				      format(self.state['epoch'],
				             loss=loss))
			
			else:
				print('Test Epoch({0}):\taverage test Loss is {loss}\t'.format(self.state['epoch'], loss=loss))
		self.is_convergency()
		
		return loss
	
	def train(self, data_loader, model, criterion, optimizer, epoch):
		
		model.train()
		
		self.state['one_epoch_cauchy_loss'].clear()
		
		self.on_start_epoch(True, model, criterion, data_loader, optimizer)
		
		if self.state['use_pb']:
			data_loader = tqdm(data_loader, desc=str("(" + str(epoch) + ')Training')) 
		
		end = time.time()
		for i, (input, target) in enumerate(data_loader):
			
			self.state['iteration'] = i
			self.state['data_time_batch'] = time.time() - end
			self.state['data_time'].add(self.state['data_time_batch'])
			
			self.state['input'] = input   
			self.state['target'] = target  
			self.on_start_batch(True, model, criterion, data_loader, optimizer)
			
			if self.state['use_gpu']:
				self.state['target'] = self.state['target'].cuda(async=True)
			
			self.on_forward(True, model, criterion, data_loader, optimizer)
			
			self.state['batch_time_current'] = time.time() - end
			self.state['batch_time'].add(self.state['batch_time_current'])
			end = time.time()
			self.on_end_batch(True, model, criterion, data_loader, optimizer, display=False)
		
		self.on_end_epoch(True, model, criterion, data_loader, optimizer)  
	
	def validate(self, data_loader, model, criterion):
		print("In this function will generate all the hash-code of test set samples\n")
		model.eval()
		
		self.on_start_epoch(False, model, criterion, data_loader)
		
		if self.state['use_pb']:
			data_loader = tqdm(data_loader, desc='Test')
		
		end = time.time()
		for i, (input, target) in enumerate(data_loader):
			self.state['iteration'] = i
			self.state['data_time_batch'] = time.time() - end
			self.state['data_time'].add(self.state['data_time_batch'])
			
			self.state['input'] = input
			self.state['target'] = target
			
			self.on_start_batch(False, model, criterion, data_loader)
			
			if self.state['use_gpu']:
				self.state['target'] = self.state['target'].cuda(async=True)
			
			self.on_forward(False, model, criterion, data_loader)
			
			self.state['batch_time_current'] = time.time() - end
			self.state['batch_time'].add(self.state['batch_time_current'])
			end = time.time()
			self.on_end_batch(False, model, criterion, data_loader, display=False)
		
		score = self.on_end_epoch(False, model, criterion, data_loader) 
		
		return score
	
	def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):
		
		self.init_learning(model, criterion)
		self.reset()
		train_dataset.transform = self.state['train_transform']
		train_dataset.target_transform = self._state('train_target_transform')
		val_dataset.transform = self.state['val_transform']
		val_dataset.target_transform = self._state('val_target_transform')
		
		train_loader = torch.utils.data.DataLoader(train_dataset,
		                                           batch_size=self.state['batch_size'], shuffle=True,
		                                           num_workers=self.state['workers'])
		val_loader = torch.utils.data.DataLoader(val_dataset,
		                                         batch_size=self.state['batch_size'], shuffle=False,
		                                         num_workers=self.state['workers'])
		if self.state['HASH_TASK']:
			if self._state('overall_test_set') is None:
				self.state['overall_test_set'] = torch.utils.data.DataLoader(val_dataset,
				                                                             batch_size=self.state['test_set_amount'],
				                                                             shuffle=False,
				                                                             num_workers=self.state['workers'])
			if self._state('test_set_count') is None:
				self.state['test_set_count'] = 0
			
		
		self.state['temp_dir'] = self.state['hashcode_pool'][:-4] + "_hb" + str(self.state['hash_bit']) + '_' + \
		                         self.state['start_time'] + '_temp' + '.pkl'
		self.state['destination'] = self.state['hashcode_pool'][:-4] + "_hb" + str(self.state['hash_bit']) + '_' + \
		                            self.state['start_time'] + '.pkl'
		
		self.state['before_fc_tmp'] = self.state['hashcode_pool'][:-4] + "_hb" + str(self.state['hash_bit']) + '_' + \
			                              self.state['start_time'] + 'before_fc_temp' + '.pkl'
		self.state['before_fc_destination'] = self.state['hashcode_pool'][:-4] + "_hb" + str(
				self.state['hash_bit']) + '_' + self.state['start_time'] + 'before_fc.pkl'
		
		if self._state('resume') is not None:
			if os.path.isfile(self.state['resume']):
				print("=> loading checkpoint '{}'".format(self.state['resume']))
				checkpoint = torch.load(self.state['resume'])
				self.state['start_epoch'] = checkpoint['epoch']
				self.state['epoch'] = self.state['start_epoch']
				self.state['best_score'] = checkpoint['best_score']
				model.load_state_dict(checkpoint['state_dict'])
				print("=> loaded checkpoint '{}' (epoch {})"
				      .format(self.state['evaluate'], checkpoint['epoch']))
			else:
				print("=> no checkpoint found at '{}'".format(self.state['resume']))
		
		if self.state['use_gpu']:
			train_loader.pin_memory = True
			val_loader.pin_memory = True
			cudnn.benchmark = True
			model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
			criterion = criterion.cuda()
		
		if self.state['evaluate']:
			self.state['temp_dir'] = self.state['hashcode_pool'][:-4] + "_hb" + str(self.state['hash_bit']) + '_' + \
			                         self.state['start_time'] + '.pkl'
			if store_fc_data:
				self.state['before_fc_tmp'] = self.state['hashcode_pool'][:-4] + "_hb" + str(self.state['hash_bit']) + '_' + \
			                              self.state['start_time'] + 'before_fc' + '.pkl'
			self.validate(val_loader, model, criterion)  
			return
		
		# TODO define optimizer
		
		flag = False
		for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
			self.state['epoch'] = epoch
			lr = self.adjust_learning_rate(optimizer)
			self.train(train_loader, model, criterion, optimizer, epoch) 
			
			now_score = self.validate(val_loader, model, criterion)
			print("\nin epoch({0}) the now_score is: {1}\n".format(epoch, now_score))
			is_best = now_score < self.state['best_score']
			self.state['better_than_before'] = is_best
			print("\nin epoch({0}) the is_best = {1}\n".format(epoch, is_best))
			self.state['best_score'] = min(now_score, self.state['best_score'])
			print("\nin epoch({0}) the self.state['best_score'] is: {1}\n".format(epoch, self.state['best_score']))
			
			if is_best:
				nt = datetime.datetime.now().strftime('%Y%m%d%H%M')
				print("find a better score, this will rename temp by destination({0})\n".format(nt))
				if os.path.exists(self.state['destination']):
					os.remove(self.state['destination'])
					if store_fc_data:
						os.remove(self.state['before_fc_destination'])
				if os.path.exists(self.state['temp_dir']):
					os.rename(self.state['temp_dir'], self.state['destination'])
					if store_fc_data:
						os.rename(self.state['before_fc_tmp'], self.state['before_fc_destination'])
				self.state['convergence_point'] = epoch
			else:
				nt = datetime.datetime.now().strftime('%Y%m%d%H%M')
				str1 = str(self.state['start_time']) + '.pkl'
				filename_pool = self._get_all_filename()
				for item in filename_pool:
					if re.search(str1, str(item)) != None:
						flag = True
						break
				else:
					flag = False
				if flag:
					print("is_best is False, so delete the temp file({0})\n".format(nt))
					if os.path.exists(self.state['temp_dir']):
						os.remove(self.state['temp_dir'])
						if store_fc_data:
							os.remove(self.state['before_fc_tmp'])
				else:
					os.rename(self.state['temp_dir'], self.state['destination'])
					if store_fc_data:
						os.rename(self.state['before_fc_tmp'], self.state['before_fc_destination'])
			self.save_checkpoint({
				'epoch': epoch + 1,
				'arch': self._state('arch'),
				'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
				'best_score': self.state['best_score'],
			}, is_best)
	
	def save_checkpoint(self, state, is_best, filename='hash_checkpoint.pth.tar'):
		if self._state('save_model_path') is not None:
			filename_ = filename
			filename = os.path.join(self.state['save_model_path'], filename_)
			if not os.path.exists(self.state['save_model_path']): 
				os.makedirs(self.state['save_model_path'])
		print('save model {filename}\n'.format(filename=filename))
		torch.save(state, filename)
		if is_best:
			filename_best = "hash_model_best.pth.tar"
			if self._state('save_model_path') is not None:
				filename_best = os.path.join(self.state['save_model_path'], filename_best)
			shutil.copyfile(filename, filename_best) 
			if self._state('save_model_path') is not None:
				if self._state('filename_previous_best') is not None:
					os.remove(self._state('filename_previous_best'))
				filename_best = os.path.join(self.state['save_model_path'],
				                             'hash_model_best_{score:.4f}_{time}.pth.tar'. \
				                             format(score=state['best_score'], time=self.state["start_time"]))
				shutil.copyfile(filename, filename_best)
				self.state['filename_previous_best'] = filename_best
