
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 编码模块
class Encoder(keras.Model):
	def __init__(self,filter_units,ksize,latent_size):
		super(Encoder,self).__init__()
		
		self.filter_units = filter_units # [6,6,6]
		self.ksize = ksize # 卷积核大小
		self.latent_size = latent_size # 潜在变量大小
		
		self.blocks = keras.Sequential() # 卷积模块
		
		# 卷积池化激活模块
		for i in range(len(self.filter_units)):
			# 添加卷积
			self.blocks.add(layers.Conv2D(self.filter_units[i],self.ksize,padding="same"))
			# 添加池化
			self.blocks.add(layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'))
			# 添加激活
			self.blocks.add(layers.ReLU())
			
		# 均值和标准差 生成函数
		self.fcmu = layers.Dense(self.latent_size) # 均值
		self.fcsigmas = layers.Dense(self.latent_size) # 标准差
		
	def call(self,x): # x [bs,96,56,1]
		bs = x.shape[0] # bs
		x = self.blocks(x) # [bs,12,7,6]
		x = tf.reshape(x,[bs,-1]) # [bs,-1]
		mu = self.fcmu(x)
		sigmas = self.fcsigmas(x)
		return mu,sigmas

# 解码模块
class Decoder(keras.Model):
	def __init__(self,filter_units,ksize,strides):
		super(Decoder,self).__init__()
		
		self.filter_units = filter_units # [6,6,6]
		self.ksize = ksize # 卷积核大小
		self.strides = strides # 步长
		
		self.blocks = keras.Sequential() # 卷积模块
		
		# 卷积池化激活模块
		for i in range(len(self.filter_units)):
			# 添加卷积
			self.blocks.add(layers.Conv2D(self.filter_units[i],self.ksize,padding="same"))
			# 添加反卷积 上采样
			self.blocks.add(layers.Conv2DTranspose(self.filter_units[i],self.ksize,strides=self.strides,padding="same"))
			# 添加激活
			self.blocks.add(layers.ReLU())
			
		self.fc = layers.Dense(12*7*self.filter_units[0])
		
	def call(self,x): # x [bs,64+36]
		bs = x.shape[0] # bs
		x = self.fc(x) 
		x = tf.reshape(x,[bs,12,7,self.filter_units[0]])
		x = self.blocks(x)
		
		return x


# CVAE
class CVAE(keras.Model):
	def __init__(self,filter_units,ksize,strides,latent_size,depth):
		super(CVAE,self).__init__()
		
		self.filter_units = filter_units # [6,6,6]
		self.ksize = ksize # 卷积核大小
		self.strides = strides # 步长
		self.latent_size = latent_size
		self.depth = depth # 深度 36
		
		# 编码器
		self.encoder = Encoder(self.filter_units[0:-1],self.ksize,self.latent_size)
		# 解码器
		self.decoder = Decoder(self.filter_units[1:],self.ksize,self.strides)
		
	def call(self,y,mode,x=None): # x [bs,96,56,1]
		bs = len(y) # bs
		
		e = tf.random.normal([bs,self.latent_size]) # 正态分布 (0,1)
		y = tf.one_hot(y, self.depth,dtype=tf.float32) # 真实分类标签
		if(mode=="training"): # 训练
			mu,sigmas = self.encoder(x)
			e = mu+e*sigmas
		x = tf.concat([e,y],1) # [bs,depth+latent_size]
		x = self.decoder(x) # 解码
		
		if(mode=="training"):
			return x,mu,sigmas
		else : 
			return x

"""
x = tf.random.normal([4,100,56,1])
y = [2,1,0,7]
cvae = CVAE([6,6,6,1],3,2,64,36)
x,mu,sigmas = cvae(y,"training",x)
print(x.shape,mu.shape,sigmas.shape)
#x = cvae(y,"testing")
#print(x.shape)
"""
