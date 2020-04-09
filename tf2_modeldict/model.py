


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

class SUNSHINE(object):
	def __init__(self, model_registrar, number):
					 
		# 一些基本信息的设置
		self.model_registrar = model_registrar
		self.number = number
		
		self.node_modules = dict() # 字典
		self.criterion = keras.losses.CategoricalCrossentropy()
		self.accuracy = keras.metrics.CategoricalAccuracy()
		
		

	def add_submodule(self, name, model_if_absent):
		# 注册模型
		self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)


	def clear_submodules(self):
		self.node_modules.clear() # 清除字典


	def create_graphical_model(self):
		self.clear_submodules()
		self.add_submodule('conv1', model_if_absent=layers.Conv2D(16, 3, 1, 'same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
		self.add_submodule('pool', model_if_absent=layers.AveragePooling2D(2, 2))
		self.add_submodule('conv2', model_if_absent=layers.Conv2D(32, 3, 1, 'same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
		self.add_submodule('conv3', model_if_absent=layers.Conv2D(64, 3, 1, 'same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
		
		self.add_submodule('flatten1', model_if_absent=layers.Flatten())
		
		self.add_submodule('fc1', model_if_absent=layers.Dense(1024,activation='relu', kernel_regularizer=regularizers.l2(0.001)))
		self.add_submodule('fc2', model_if_absent=layers.Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.001)))
		self.add_submodule('fc3', model_if_absent=layers.Dense(self.number,activation='softmax', kernel_regularizer=regularizers.l2(0.001)))
		
		self.add_submodule('dropout1', model_if_absent=layers.Dropout(0.5))
		self.add_submodule('dropout2', model_if_absent=layers.Dropout(0.5))
		
		
	def encoder(self, x, training):
		
		x = self.node_modules['pool'](self.node_modules['conv1'](x))
		x = self.node_modules['pool'](self.node_modules['conv2'](x))
		x = self.node_modules['pool'](self.node_modules['conv3'](x))
		
		x = self.node_modules['flatten1'](x)
		
		x = self.node_modules['dropout1'](self.node_modules['fc1'](x),training=training)
		x = self.node_modules['dropout2'](self.node_modules['fc2'](x),training=training)
		x = self.node_modules['fc3'](x)

		return x

	def train_loss(self, inputs, labels, training=True):
		
		outputs = self.encoder(inputs,training)
		loss = self.criterion(labels, outputs)
		regularization_loss=tf.math.add_n(self.model_registrar.losses)
		loss = loss + regularization_loss
		_ = self.accuracy.update_state(labels, outputs)
		accuracy = self.accuracy.result().numpy()
		return loss,accuracy
