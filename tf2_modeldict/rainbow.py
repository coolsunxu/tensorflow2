
import tensorflow as tf
# import matplotlib.pyplot as plt
import random
from pathlib import Path

"""
class RAINBOW(object):
	def __init__(self, data_root_orig, batch_size):
		self.data_root_orig = data_root_orig
		self.batch_size = batch_size
		self.number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a','b',
						'c','d','e','f','g','h','i','j','k','l','m','n','o','p',
						'q','r','s','t','u','v','w','x','y','z']
		self.load()
		
	def load(self):
		self.data_root = Path(self.data_root_orig)
		self.all_image_paths = list(self.data_root.glob('*'))
		self.all_image_paths = [str(path) for path in self.all_image_paths]
		
		self.image_count = int(len(self.all_image_paths)/self.batch_size)
		
	def shuffle_samples(self):
		random.shuffle(self.all_image_paths)
	
	# 处理图像
	def preprocess_image(self,image):
		image = tf.io.decode_image(image, channels=1,dtype=tf.dtypes.float32)
		#image /= 255.0  # normalize to [0,1] range
		return image

	def load_and_preprocess_image(self,path):
		image = tf.io.read_file(path)
		return self.preprocess_image(image)

	def get_next_batch(self,index):
		inputs = []
		labels = []
		for path in self.all_image_paths[index*self.batch_size:(index+1)*self.batch_size]:
			
			inputs.append(self.load_and_preprocess_image(path))
			labels.append(self.number.index(Path(path).name[-5]))
			
		return tf.stack(inputs,axis=0),tf.one_hot(labels,depth=len(self.number))
"""
########################################################################################

class RAINBOW(object):
	def __init__(self, data_root_orig, batch_size):
		self.data_root_orig = data_root_orig
		self.batch_size = batch_size

		self.load()
		
	def load(self):
		self.data_root = Path(self.data_root_orig)
		self.all_image_paths = list(self.data_root.glob('*/*/*'))
		self.all_image_paths = [str(path) for path in self.all_image_paths]
		
		self.image_count = int(len(self.all_image_paths)/self.batch_size)
		
	def shuffle_samples(self):
		random.shuffle(self.all_image_paths)
	
	# 处理图像
	def preprocess_image(self,image):
		image = tf.io.decode_image(image, channels=3,dtype=tf.dtypes.float32)
		
		#image /= 255.0  # normalize to [0,1] range
		return image

	def load_and_preprocess_image(self,path):
		image = tf.io.read_file(path)
		return self.preprocess_image(image)

	def get_next_batch(self,index):
		inputs = []
		labels = []
		for path in self.all_image_paths[index*self.batch_size:(index+1)*self.batch_size]:
			
			inputs.append(self.load_and_preprocess_image(path))
			labels.append(int(Path(path).name.split('_')[1]))
			
		return tf.stack(inputs,axis=0),tf.one_hot(labels,depth=10)

# https://tensorflow.google.cn/tutorials/load_data/images
