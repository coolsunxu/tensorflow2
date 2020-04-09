

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class ModelRegistrar(keras.Model):
	def __init__(self, model_dir):
		super(ModelRegistrar, self).__init__()
		self.model_dict = dict()
		self.model_dir = model_dir


	def call(self):
		# 仅仅用来存储参数
		raise NotImplementedError('Although ModelRegistrar is a nn.Module, it is only to store parameters.')


	def get_model(self, name, model_if_absent=None):
		# 4 cases: name in self.model_dict and model_if_absent is None         (OK)
		#          name in self.model_dict and model_if_absent is not None     (OK)
		#          name not in self.model_dict and model_if_absent is not None (OK)
		#          name not in self.model_dict and model_if_absent is None     (NOT OK)

		if name in self.model_dict:
			return self.model_dict[name]

		elif model_if_absent is not None:
			self.model_dict[name] = model_if_absent # 注册 
			return self.model_dict[name]

		else:
			raise ValueError(f'{name} was never initialized in this Registrar!')


	def print_model_names(self):
		print(self.model_dict.keys())


	def save_models(self):
		
		print('')
		print('Saving to ' + self.model_dir)
		self.save_weights(self.model_dir)
		print('Saved!')
		print('')


	def load_models(self):
		self.model_dict.clear()
		print('')
		print('Loading from ' + self.model_dir)
		self.load_weights(self.model_dir)
		print('Loaded!')
		print('')

