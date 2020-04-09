
import  tensorflow as tf
from tensorflow.keras import layers

# 训练阶段
with tf.GradientTape() as tape:
	
	out = network(x, training=True) # 类似 pytorch 的 model.train()

# 在测试阶段，需要设置 training=False， 避免 BN 层采用错误的行为：
for x,y in db_test: # 遍历测试集
	
	out = network(x, training=False) # 类似 pytorch 的 model.eval()

# 上述同样适用于 dropout 

