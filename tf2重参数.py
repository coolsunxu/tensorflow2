

import tensorflow as tf
import tensorflow_probability as tfp


# 生成 OneHotCategorical
def dist_from_h(h, N, K, z_logit_clip, mode):
	logits_separated = tf.reshape(h, [-1, N, K])
	logits_separated_mean_zero = logits_separated - tf.reduce_mean(logits_separated, axis=-1, keepdims=True)
	if z_logit_clip is not None and mode == 'train':
		c = z_logit_clip
		logits = tf.clip_by_value(logits_separated_mean_zero, -c, c)
	else:
		logits = logits_separated_mean_zero
	
	if logits.shape[0] == 1:
		logits = tf.squeeze(logits, 0)
	
	return tfp.distributions.OneHotCategorical(logits=logits)


def sample_q(k, p_dist, temp, z_dim, mode):
	if mode == 'train':
		z_dist = tfp.distributions.RelaxedOneHotCategorical(temp, logits=p_dist.logits)
		print(z_dist.reparameterization_type)
		z_NK = z_dist.sample((k, ))
	elif mode == 'eval':
		z_NK = p_dist.sample((k, ))
	return tf.reshape(z_NK, (k, -1, z_dim))


# 计算 KL 散度
def kl_q_p(p_dist, q_dist, kl_min):
	kl_separated = tfp.distributions.kl_divergence(p_dist, q_dist)
	
	if len(kl_separated.shape) < 2:
		kl_separated = tf.expand_dims(kl_separated, 0)
		
	kl_minibatch = tf.reduce_mean(kl_separated, axis=0, keepdims=True)
	
	if kl_min > 0:
		kl_lower_bounded = tf.maximum(kl_minibatch, kl_min)
		kl = tf.reduce_sum(kl_lower_bounded)
	else:
		kl = tf.reduce_sum(kl_minibatch)
	
	return kl


N = 2
K = 5
k = 3
z_logit_clip = 1
z_dim =10 
temp = 0.1
kl_min = 0.07
mode = 'train'

# p(x) 是目标分布，q(x)是去匹配的分布
logits_p = tf.random.normal([120])
logits_q = tf.random.normal([120])
dist_p = dist_from_h(logits_p, N, K, z_logit_clip, mode)
print(dist_p.reparameterization_type)
dist_q = dist_from_h(logits_q, N, K, z_logit_clip, mode)

print(dist_p)
print(dist_q)

# 计算 KL 散度
kl = kl_q_p(dist_p, dist_p, kl_min)

print(kl)

# 采样
sample  = sample_q(k, dist_p, temp, z_dim, mode)
print(sample)
