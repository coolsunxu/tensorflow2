

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,initializers


class BatchMultiHeadGraphAttention(keras.Model): # 多头图注意力模型
	def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
		super(BatchMultiHeadGraphAttention, self).__init__()
		self.n_head = n_head # 头大小
		self.f_in = f_in # 输入大小
		self.f_out = f_out # 输出大小
		self.attn_dropout = attn_dropout # dropout
		self.add_self_loop = True # 为防止没有邻居结点出现的情况
		self.initializer = initializers.GlorotUniform() # 初始化分布
		self.w = tf.Variable(self.initializer(shape=[self.n_head, self.f_in, self.f_out], dtype=tf.float32)) # 自定义参数 权重
		self.adj = []
		self.fc = tf.Variable(self.initializer(shape=[self.n_head, 2*self.f_out, 1], dtype=tf.float32)) # 自定义参数 att
		self.leaky_relu = layers.LeakyReLU(alpha=0.2) # 激活函数
		self.softmax = layers.Softmax(axis=-1) # 归一层
		self.dropout = layers.Dropout(rate=self.attn_dropout) # Dropout 层
		if bias:
			self.bias = tf.Variable(tf.zeros(self.f_out)) # 自定义参数 偏置

	def remove_self_loops(self,edge_index): # 移除自环
		row, col = edge_index
		mask = tf.where(row != col) # 返回的是序号 不相等
		edge_index = tf.transpose(tf.gather(tf.transpose(edge_index),tf.squeeze(mask)))
		return edge_index
	
	def add_self_loops(self, edge_index, num_nodes): # 添加自环
		loop_index = tf.range(0, num_nodes, dtype=tf.int64)
		loop_index = tf.tile(tf.expand_dims(loop_index,0),[2, 1])
		edge_index = tf.concat([edge_index, loop_index], 1)
		return edge_index
		
	def call(self, h, edge_index): 
		bs = h.shape[0] # [bs,fin]
		if self.add_self_loop: # 是否添加自环 
			self.remove_self_loops(edge_index)
			self.add_self_loops(edge_index, bs)
			
		h_prime = tf.matmul(h, self.w) # [head,bs,fout]
		
		for i in range(h_prime.shape[1]): # for each node
			neighbors = tf.gather(edge_index[1,:],tf.squeeze(tf.where(edge_index[0,:]==i)),0) # neighbors
			if self.n_head == 1:
				shape = tf.cast(tf.constant([bs]),dtype = tf.int64)
			else :
				shape = tf.cast(tf.constant([bs,self.n_head]),dtype = tf.int64)
			n_neighbors = neighbors.shape[0] # number of this node's neighbors
			curr_node = tf.tile(tf.expand_dims(h_prime[:,i,:],1),[1, n_neighbors, 1]) # [head,cbs,fout] 注意tf.repeat是按照某一个维度切分 分别复制的
			neighbors_node = tf.gather(h_prime,neighbors,axis=1) # [head,cbs,fout]
			total_node = tf.concat((curr_node,neighbors_node),2) # [head,cbs,fout*2]
			
			#att_node = self.leaky_relu(tf.matmul(total_node,self.fc))
			att_node = self.leaky_relu(total_node@self.fc)
			att_node = self.softmax(tf.reshape(att_node,[self.n_head,n_neighbors])) # [head,cbs]
			att_node = self.dropout(att_node)
			att_node = tf.transpose(att_node,[1,0]) # 方便使用tf.scatter_nd函数
			scatter = tf.scatter_nd(tf.expand_dims(neighbors,1), tf.squeeze(att_node), shape) 
			self.adj.append(tf.transpose(scatter))
		output = tf.matmul(tf.stack(self.adj,1),h_prime)  # [head,bs,f_out]
		output = tf.reduce_mean(output,0) # [bs,fout]
		
		if self.bias is not None:
			return output + self.bias
		else:
			return output
			
# 生成领接矩阵
def Get_Adj(bs):
	import itertools
	a = [[],[]]
	list_indices = range(bs) # 行人列表
	for i, j in itertools.permutations(list_indices, 2): # 行人-行人 两两全排列
		a[0].append(i)
		a[1].append(j)
	return tf.convert_to_tensor(a,dtype=tf.int64)
	
import time
heads = 12
bs = 512
fin = 256
fout = 128
a = Get_Adj(bs)
h = tf.random.normal([bs,fin])
model = BatchMultiHeadGraphAttention(n_head=heads, f_in=fin, f_out=fout, attn_dropout=0.5)
start_time = time.time()  #开始时间
out = model(h,a)
end_time = time.time()   #结束时间
print("time:%d"  % (end_time-start_time))
print(out.shape)

"""
with tf.GradientTape(persistent=True) as tape: 
	 out = model(h,a)
	 loss = tf.reduce_sum(out)
	 print(out.shape)
print(tape.gradient(loss, model.trainable_variables))
"""

