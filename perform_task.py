# -*- coding=utf-8 -*-
import sys, os, time
import subprocess, argparse

inter_stride_pool = [358,]
bs_v = [64 for u in range(int(len(inter_stride_pool)))]
epochs_pool = [15, ]
p_pool = [0.15, 0.2, 0.25, 0.3, 0.35]
tao_pool = [0.3, 0.35, 0.4, 0.45, 0.5]
hash_bit_pool = [16, 32, 48, 64, 128]
w2v_pool = {'gg':'_googlenews_word2vec.pkl',
            'ft':'_fasttext_word2vec.pkl',
            'gl':'_glove_word2vec.pkl'}
MFB_pool = [False, True]


def par_option():
	parser = argparse.ArgumentParser(description='Performation all')
	parser.add_argument('-H',"--HASH_TASK", action='store_true')
	parser.add_argument('-p',"--perform_now", action='store_true')
	parser.add_argument("--datasetname", type=str, default='voc')
	parser.add_argument("--start_subscript", type=int, default=0)
	parser.add_argument("--end_subscript", type=int, default=30)
	
	return parser

class Performation(object):
	def __init__(self, state={}):
		super(Performation, self).__init__()
		self.state = state
		for k, v in self.state.items():
			print('{0} = {1}'.format(k, v))
		
	def _state(self,name):
		if name in self.state.keys():
			return self.state[str(name)]
		return None
		
	def cat_base_cmd(self):
		if self.state['datasetname'] == 'voc':
			return str("python demo_" + str(self.state['datasetname']) + "2007_gcn.py data/" + str(
				self.state['datasetname']) + "/ --image-size 448 --batch-size ")
		else:
			return str(
				"python demo_" + str(self.state['datasetname']) + "_gcn.py data/" + str(self.state['datasetname']) +
				"/ --image-size 448 --batch-size ")
	
	def cat_cmd(self, base_cmd, destination_path, bs_value, inter_value, log_name,
	            hash_bit_value, pooling_stride_value, w2v_value, epoch_value=50,
	            is_MFB=True, p_value=0.15):
		exec_cmd = ''
		if self.state['HASH_TASK']:
			if is_MFB:
				exec_cmd = base_cmd + str(bs_value) + \
				           " --HASH_TASK --IS_USE_MFB --NORMED --IS_USE_IOU -t -v" + \
				           " --linear_intermediate " + str(inter_value) + \
				           " --pooling_stride " + str(pooling_stride_value) + \
				           " --epochs " + str(epoch_value) + \
				           " --threshold_p " + str(p_value) + \
				           " --HASH_BIT " + str(hash_bit_value) + \
				           " --word2vec_file " + str(w2v_value) + \
				           "  > "  + destination_path + log_name
			else:
				exec_cmd = base_cmd + str(bs_value) + \
				           " --HASH_TASK --NORMED --IS_USE_IOU -t -v" + \
				           " --linear_intermediate " + str(inter_value) + \
				           " --pooling_stride " + str(pooling_stride_value) + \
				           " --epochs " + str(epoch_value) + \
				           " --threshold_p " + str(p_value) + \
				           " --HASH_BIT " + str(hash_bit_value) + \
				           " --word2vec_file " + str(w2v_value) + \
				           "  > " + destination_path + log_name
		else:
			if is_MFB:
				exec_cmd = base_cmd + str(bs_value) + \
				           " --linear_intermediate " + str(inter_value) + \
				           " --pooling_stride " + str(pooling_stride_value) + \
				           " --epochs " + str(epoch_value) + \
				           " --threshold_p " + str(p_value) + \
				           " --word2vec_file " + str(w2v_value) + \
				           " --IS_USE_MFB --NORMED --IS_USE_IOU -t  > " + destination_path + log_name
			else:
				exec_cmd = base_cmd + str(bs_value) + \
				           " --linear_intermediate " + str(inter_value) + \
				           " --pooling_stride " + str(pooling_stride_value) + \
				           " --epochs " + str(epoch_value) + \
				           " --threshold_p " + str(p_value) + \
				           " --word2vec_file " + str(w2v_value) + \
				           " --NORMED --IS_USE_IOU -t  > " + destination_path + log_name
		return exec_cmd
	
	
	def generate_cmdpool(self,):
		exec_cmd_pool = []
		basedir = os.path.abspath(os.path.dirname(__file__))
		
		if os.path.exists(basedir + '/' + str(self.state['datasetname']) + '_log_files/') == False:
			os.makedirs(basedir + '/' + str(self.state['datasetname']) + '_log_files/')
		
		desdir = os.path.join(basedir, str(self.state['datasetname']) + '_log_files/')
		
		base_cmd = self.cat_base_cmd()
		
		if self.state['HASH_TASK']:
			for h in range(len(MFB_pool)):
				for k in w2v_pool.keys():
					for i in range(int(len(inter_stride_pool))):
						for j in range(int(len(hash_bit_pool))):
							if MFB_pool[h]:
								log_name = "hashlog@" + str(self.state['datasetname']) + \
								           str(time.strftime('%Y%m%d%H%M', time.localtime(time.time()))) + \
								           "_it" + str(inter_stride_pool[i]) + \
								           "_hb" + str(hash_bit_pool[j]) + \
								           '_MFB'+ str(1) +'_'+ str(k)
								exec_cmd = self.cat_cmd(base_cmd, desdir, bs_v[i], inter_stride_pool[i], log_name,
								                   hash_bit_pool[j], inter_stride_pool[i],
								                   'data/' + str(self.state['datasetname']) + '/'
								                        + str(self.state['datasetname']) + w2v_pool[k],
								                   epochs_pool[0], is_MFB=True)
							else:
								log_name = "hashlog@" + str(self.state['datasetname']) + \
								           str(time.strftime('%Y%m%d%H%M', time.localtime(time.time()))) + \
								           "_it" + str(inter_stride_pool[i]) + \
								           "_hb" + str(hash_bit_pool[j]) + \
								           '_MFB' + str(0) + '_' + str(k)
								exec_cmd = self.cat_cmd(base_cmd, desdir, bs_v[i], inter_stride_pool[i], log_name,
								                   hash_bit_pool[j], inter_stride_pool[i],
								                   'data/'+str(self.state['datasetname'])+'/'
								                        +str(self.state['datasetname'])+w2v_pool[k],
								                   epochs_pool[0], is_MFB=False)
							exec_cmd_pool.append(exec_cmd)
		else:
			for h in range(len(MFB_pool)):
				for k in w2v_pool.keys():
					for i in range(int(len(inter_stride_pool))):
						if MFB_pool[h]:
							log_name = "MLlog@" + str(self.state['datasetname']) + \
							           str(time.strftime('%Y%m%d%H%M', time.localtime(time.time()))) + \
							           "_it" + str(inter_stride_pool[i]) + \
							           '_MFB'+ str(1) +'_'+ str(k)
							exec_cmd = self.cat_cmd(base_cmd, desdir, bs_v[i], inter_stride_pool[i], log_name,
							                   0,  inter_stride_pool[i],
							                   'data/' + str(self.state['datasetname']) + '/' +
							                        str(self.state['datasetname']) + w2v_pool[k],
							                   epochs_pool[0], is_MFB=True)
						else:
							log_name = "MLlog@" + str(self.state['datasetname']) + \
							           str(time.strftime('%Y%m%d%H%M', time.localtime(time.time()))) + \
							           "_it" + str(inter_stride_pool[i]) + \
							           '_MFB' + str(0) + '_' + str(k)
							exec_cmd = self.cat_cmd(base_cmd, desdir, bs_v[i], inter_stride_pool[i], log_name,
							                 0, inter_stride_pool[i],
							                   'data/'+str(self.state['datasetname'])+'/'+
							                        str(self.state['datasetname'])+w2v_pool[k],
							                   epochs_pool[0], is_MFB=False)
						exec_cmd_pool.append(exec_cmd)
		
		if self._state('exec_cmd_pool') is None:
			self.state['exec_cmd_pool'] = exec_cmd_pool
	
	def perform(self):
		cmd_pool_len = len(self.state['exec_cmd_pool'])
		i = 0
		if self.state['end_subscript'] > cmd_pool_len:
			self.state['end_subscript'] = cmd_pool_len
		for i in range(self.state['start_subscript'], self.state['end_subscript']):
			print("\n**********[ The {seq}-th experiment/still {value} more experiments/total {total} ]**********".
			      format(seq = int(i), value=int(cmd_pool_len - i), total=cmd_pool_len))
			cmd = str(self.state['exec_cmd_pool'][i])
			print('({0}):'.format(i),cmd)
			print()
			if self.state['perform_now']:
				os.system(str(cmd))
		
def main():
	parser = par_option()
	local_args = parser.parse_args()
	state = {'datasetname':local_args.datasetname, 'start_subscript':local_args.start_subscript,
	         'end_subscript':local_args.end_subscript, 'HASH_TASK':local_args.HASH_TASK,
	         'perform_now':local_args.perform_now}
	perform_obj = Performation(state)
	perform_obj.generate_cmdpool()
	perform_obj.perform()
	
if __name__ == "__main__":
	main()
