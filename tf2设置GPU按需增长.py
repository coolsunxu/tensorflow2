
import tensorflow as tf

def Set_GPU_Memory_Growth():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			# 设置 GPU 显存占用为按需分配
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			# 异常处理
			print(e)
	else :
		print('No GPU')

# 放在建立模型实例之前
Set_GPU_Memory_Growth()
