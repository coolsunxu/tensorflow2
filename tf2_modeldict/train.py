
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
import matplotlib.pyplot as plt
import time

from model_registrar import ModelRegistrar
from rainbow import RAINBOW
from model import SUNSHINE


rain = RAINBOW('./CIFAR10',8)
number = 10
model_dir = './checkpoints/weights.ckpt'
model_registrar = ModelRegistrar(model_dir) # 注册模型 完成对模型的一些基本操作

if Path(model_dir+'.index').exists():
	model_registrar.load_models()
else :
	pass

model_1 = SUNSHINE(model_registrar,number)
model_1.create_graphical_model()

model_2 = SUNSHINE(model_registrar,number)
model_2.create_graphical_model()

model_registrar.print_model_names()

optimizer = keras.optimizers.RMSprop() 

# 开始训练
plt.ion()
fig,axes=plt.subplots(2)
ax1=axes[0]
ax2=axes[1]

def train_epoch(epoch):
	for step in range(rain.image_count-1):
		
		with tf.GradientTape() as tape:
			inputs, labels = rain.get_next_batch(step)
			inputs_, labels_ = rain.get_next_batch(step+1)
			
			loss1,accuracy1 = model_1.train_loss(inputs, labels, training=True)
			loss2,accuracy2 = model_2.train_loss(inputs_, labels_, training=True)
			loss = (loss1 + loss2) / 2
			accuracy = (accuracy1 + accuracy2) / 2
		
		grads = tape.gradient(loss, model_registrar.trainable_variables)
		optimizer.apply_gradients(zip(grads, model_registrar.trainable_variables))

		if (step+1) % 1 == 0:
			print(epoch, step, 'loss:', loss.numpy(), 'accuracy:', accuracy)
		
		if ((epoch+1)*step) % 100 == 0:
			model_registrar.save_models()
			
		# 实时绘图
		ax1.plot((epoch+1)*step,loss.numpy(),'.b')
		ax2.plot((epoch+1)*step,accuracy,'.r')
		plt.draw()
		plt.pause(0.001)
		
		
def train():
	
	for epoch in range(30):
		
		rain.shuffle_samples()
		train_epoch(epoch)
		
	print('Finished Training')
	
train()
