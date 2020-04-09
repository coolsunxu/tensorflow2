
import tensorflow as tf  
import os

from graph_nets import utils_np
from graph_nets import utils_tf

import sonnet as snt
import graph_nets as gn

from curtain import Code

batch_size = 16
img_height = 100
img_width = 56
learning_rate = 1e-4
max_iteration = 1000000

stride = 8

checkpoint_root = "./checkpoints"
checkpoint_name = "model"
save_prefix = os.path.join(checkpoint_root, checkpoint_name)

start_step = 0
code = Code("../tf2_modeldict/image",batch_size,stride,img_width,img_height)


OUTPUT_EDGE_SIZE = 256
OUTPUT_NODE_SIZE = 256
OUTPUT_GLOBAL_SIZE = 36

node = snt.Sequential([
    snt.Linear(1024),
    tf.nn.relu,
    snt.Linear(OUTPUT_NODE_SIZE)
])

edge = snt.Sequential([
    snt.Linear(1024),
    tf.nn.relu,
    snt.Linear(OUTPUT_EDGE_SIZE)
])

global_s = snt.Sequential([
    snt.Linear(256),
    tf.nn.relu,
    snt.Linear(512),
    tf.nn.relu,
    snt.Linear(OUTPUT_GLOBAL_SIZE)
])

graph_network = gn.modules.GraphNetwork(
    edge_model_fn=lambda: node,
    node_model_fn=lambda: edge,
    global_model_fn=lambda: global_s)
  


checkpoint = tf.train.Checkpoint(module=graph_network)

latest = tf.train.latest_checkpoint(checkpoint_root)
if latest is not None:
	checkpoint.restore(latest)

loss_object = tf.keras.losses.CategoricalCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

for echo in range(max_iteration):
	code.mess_up_order()
	
	for i in range(code.total_number):
		with tf.GradientTape() as gen_tape:
			Input_data, Output_data = code.next_batch(i)
			graph_dicts = utils_np.graphs_tuple_to_data_dicts(Input_data)
			
			graphs_tuple_tf = utils_tf.data_dicts_to_graphs_tuple(graph_dicts)
			output_data = graph_network(graphs_tuple_tf).globals
			loss = tf.nn.softmax_cross_entropy_with_logits(logits=output_data, labels=Output_data)
			#loss = loss_object(Output_data,output_data)
		
		gradients_of_generator = gen_tape.gradient(loss, graph_network.trainable_variables)

		generator_optimizer.apply_gradients(zip(gradients_of_generator, graph_network.trainable_variables))

		print('Echo %d,Iter %d: train_loss is: %.5f train_accuracy is: %.5f'%(echo+1, i+1, tf.reduce_mean(loss),train_accuracy(Output_data,output_data)))
		
			
		if i and i % 10 == 0:
			checkpoint.save(save_prefix)

    




