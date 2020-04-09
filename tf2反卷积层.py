

import tensorflow as tf
from tensorflow.keras import layers


x = tf.random.normal([12,64,64,1])
conv1 = layers.Conv2DTranspose(5,kernel_size=4,strides=3,padding='same')

y = conv1(x)

print(y.shape)

# valid : o = (i-1)*s+k
# same  : o = i*s
