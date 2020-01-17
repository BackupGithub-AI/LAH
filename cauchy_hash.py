import torch as th
import numpy as np
import sys, os


class CauchyLoss(th.nn.Module):
	def __init__(self, gamma=1, q_lambda=0.55, sij_type="IOU", normed=True):
		super(CauchyLoss, self).__init__()
		self.q_loss_img, self.cos_loss, self.loss = 0, 0, 0
		self.gamma = gamma
		self.q_lambda = q_lambda
		self.sij_type = sij_type
		self.normed = normed
		self.gpu_state = th.cuda.is_available()
		
	def forward(self, batch_label,  layer_out, LOCAL_TRAINING_FLAG=True):
		'''
		calculate the loss value
		:param self:
		:return:
		'''
		self.b_label = batch_label
		self.output_dim = layer_out.shape[1]       
		self.u = layer_out.float()              
		self.label_u = batch_label.float()        
		return self.apply_loss_function()
	
	 
	def cauchy_cross_entropy(self, v=None, label_v=None):
		label_u = self._gpu_cpu(self.label_u)
		u = self._gpu_cpu(self.u)
		normed = self.normed
		
		if v is None:
			v = self._gpu_cpu(u)
			label_v = self._gpu_cpu(label_u)
		
		s = self._gpu_cpu(self.SIJ_RES(label_u, res_type=self.sij_type))
		
		if normed:
			ip_1 = self._gpu_cpu(th.matmul(u, th.transpose(v, 0, 1)).float()) 
			
			def reduce_shaper(t):
				return th.reshape(th.sum(t, 1), [t.shape[0], 1])
			
			mod_1 = self._gpu_cpu(th.sqrt(th.matmul(reduce_shaper(u.pow(2)), th.transpose(reduce_shaper(v.pow(2)) + 0.000001,0,1))).float())
			
			dist = self._gpu_cpu(float(self.output_dim) / 2.0 * (1.0 - th.div(ip_1, mod_1) + 0.000001))
			
		else:
			r_u = self._gpu_cpu(th.reshape(th.sum(u * u, 1), [-1, 1]))  
			r_v = self._gpu_cpu(th.reshape(th.sum(v * v, 1), [-1, 1]))
			
			dist = self._gpu_cpu(r_u - 2 * th.matmul(u, th.transpose(v, 0, 1)) + th.transpose(r_v, 0, 1) + 0.001)
			
		cauchy = self._gpu_cpu(self.gamma / (dist + self.gamma))
		
		s_t = self._gpu_cpu(2.0 * th.add(s, -0.5))
		sum_1 = float(th.sum(s))
		sum_all = float(th.sum(th.abs(s_t)))
		assert sum_1!=0, 'Maybe the batch size too small, so the label vectors have no interact set'
		balance_param = self._gpu_cpu(th.add(th.abs(th.add(s, -1.0)), float(sum_all / sum_1) * s))     
		mask = self._gpu_cpu(self.create_mask(s.shape[0])).long()
		cauchy_mask = self._gpu_cpu(th.gather(cauchy,1,mask).reshape(1,-1).squeeze())   
		s_mask = self._gpu_cpu(th.gather(s,1,mask).reshape(1,-1).squeeze())
		balance_p_mask = self._gpu_cpu(th.gather(balance_param,1,mask).reshape(1,-1).squeeze())  
		all_loss = self._gpu_cpu(- s_mask * th.log(cauchy_mask) - (1.0 - s_mask) * th.log(1.0 - cauchy_mask))
		return self._gpu_cpu(th.mean(th.mul(all_loss, balance_p_mask)))
	
	 
	def apply_loss_function(self):
		self.cos_loss = self.cauchy_cross_entropy()
		self.q_loss_img = th.mean(th.pow(th.abs(self.u)-1.0,2))
		self.q_loss = self.q_lambda * self.q_loss_img

		self.loss = self.cos_loss + self.q_loss
		
		return self.loss
	
	def test_criterion(self):
		
		pass
	 
	def clip_by_tensor(self, t, t_min, t_max):
		
		t = t.float()
		result = th.clamp(t, min=0.0, max=1.0)
		return result
	
	 
	def SIJ_RES(self, lm_1, lm_2=None, res_type="IOU"):
		
		if lm_2==None:
			lm_2 = lm_1
		
		dim = lm_1.shape[0]
		silm = self._gpu_cpu(th.matmul(lm_1, th.transpose(lm_2,0,1)).float())
		em = self._gpu_cpu(th.eye(dim).float())
		dig_silm = self._gpu_cpu(th.mul(silm, em).float()) 
		cdig_silm = self._gpu_cpu(silm - dig_silm) 
		
		if res_type == "IOU":
			for i in range(dim):
				for j in range(i + 1, dim):
					cdig_silm[i][j] = cdig_silm[i][j] / (dig_silm[i][i] + dig_silm[j][j] - cdig_silm[i][j])
					cdig_silm[j][i] = cdig_silm[i][j]
			return self._gpu_cpu(cdig_silm + em)
		elif res_type =="original":
			label_ip = th.matmul(lm_1, th.transpose(lm_2, 0, 1)).float()
			s = th.clamp(label_ip, min=0.0, max=1.0)
			
			return self._gpu_cpu(s)
		else:
			assert (res_type=="IOU" or res_type=="original"),\
				"No such initialize s_ij method... Process Abort!!!"
	 
	def create_mask(self,mat_dim):
		
		out = []
		out = th.zeros([mat_dim,mat_dim - 1]).long()
		pool = [x for x in range(mat_dim)]
		tmp_pool = []
		for i in range(mat_dim):
			tmp_pool = pool.copy()
			tmp_pool.remove(int(i))
			for j in range(mat_dim - 1):
				out[i][j] = tmp_pool[j]
				
		return self._gpu_cpu(out)
	
	def _gpu_cpu(self, input):
		
		if th.is_tensor(input):
			if self.gpu_state:
				input = input.float()
				return input.cuda()
		return input.float()


