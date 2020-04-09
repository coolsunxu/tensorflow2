
import tensorflow as tf
from tensorflow.keras import applications

"""
keras.applications 中共有以下模型
DenseNet121(...)
DenseNet169(...)
DenseNet201(...)
InceptionResNetV2(...)
InceptionV3(...)
MobileNet(...)
MobileNetV2(...)
NASNetLarge(...)
NASNetMobile(...)
ResNet101(...)
ResNet101V2(...)
ResNet152(...)
ResNet152V2(...)
ResNet50(...)
ResNet50V2(...)
VGG16(...)
VGG19(...)
Xception(...)
下面仅以NASNetLarge为例
"""

# 第一种情况自定义输入大小 这时不能加载imagenet预训练参数  
IMG_SHAPE = (224,224,3)                                          
base_model = applications.NASNetLarge(input_shape=IMG_SHAPE, include_top=False, 
					weights=None, pooling='avg')

base_model.trainable = True # 参数可训练

# 第二种情况采用API自定义的输入大小，这时可以加载imagenet预训练参数                                           
base_model = applications.NASNetLarge(input_shape=None, include_top=False, 
					weights='imagenet', pooling='avg')
					
base_model.trainable = False # 参数可训练 这个看你自己的想法

# 第三种情况 不对网络进行修改
base_model = applications.NASNetLarge(input_shape=None, include_top=True, 
					weights='imagenet', pooling=None)
					
base_model.trainable = True # 参数可训练 这个看你自己的想法

# 看一下模型的结构
base_model.summary()

# 大多数 我们用的还是第二种情况

"""
https://tensorflow.google.cn/api_docs/python/tf/keras/applications
https://www.tensorflow.org/tutorials/images/transfer_learning
https://keras.io/applications/
"""
