

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers

class TemporalAttention(keras.Model):
	def __init__(self,fin,fout=1):
		super(TemporalAttention,self).__init__()
		self.fin = fin # 输入维度
		self.fout = fout # 输出维度 这里为1 求得是分数
		
		self.initializer = initializers.GlorotUniform() # 初始化分布
		# 自定义可学习参数
		self.w = tf.Variable(self.initializer(shape=[self.fin, self.fout], dtype=tf.float32))
		
	def call(self,h): # h:[bs,seq,fin]
		x = h # [bs,seq,fin]
		alpha = h @ self.w # [bs,seq,1] fout==1
		alpha = tf.nn.softmax(tf.tanh(alpha),1) # [bs,seq,1]
		x = tf.einsum('ijk,ijm->ikm', alpha, x) # [bs,1,fin]
		return tf.squeeze(x,[1]) # [bs,fin]
		
a = tf.random.normal([42,8,64])
model = TemporalAttention(64,1)
z = model(a)
print(z.shape)

