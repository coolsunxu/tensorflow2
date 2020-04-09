
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
optimizer = tf.keras.optimizers.Adam(1e-4)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

############# tf_function ###################
def update_step(inputs_tr, targets_tr):
	with tf.GradientTape() as tape:
		outputs_tr = graph_network(inputs_tr).globals
		# Loss.
		#loss_tr = loss_object(targets_tr,outputs_tr)
		loss_tr = tf.nn.softmax_cross_entropy_with_logits(logits=outputs_tr, labels=targets_tr)
	gradients = tape.gradient(loss_tr, graph_network.trainable_variables)
	optimizer.apply_gradients(zip(gradients, graph_network.trainable_variables))
	return outputs_tr, loss_tr

def specs_from_tensor(tensor_sample,description_fn=tf.TensorSpec):
	
	shape = list(tensor_sample.shape)
	dtype = tensor_sample.dtype

	return description_fn(shape=shape, dtype=dtype) 

# Get some example data that resembles the tensors that will be fed
# into update_step():

Input_data, example_target_data = code.next_batch(0)
graph_dicts = utils_np.graphs_tuple_to_data_dicts(Input_data)	
example_input_data = utils_tf.data_dicts_to_graphs_tuple(graph_dicts)

# Get the input signature for that function by obtaining the specs
input_signature = [
  utils_tf.specs_from_graphs_tuple(example_input_data),
  specs_from_tensor(example_target_data)
]

# Compile the update function using the input signature for speedy code.
compiled_update_step = tf.function(update_step, input_signature=input_signature)
############# tf_function ###################

for echo in range(max_iteration):
	code.mess_up_order()
	
	for i in range(code.total_number):
		Input_data, Output_data = code.next_batch(i)
		graph_dicts = utils_np.graphs_tuple_to_data_dicts(Input_data)
		graphs_tuple_tf = utils_tf.data_dicts_to_graphs_tuple(graph_dicts)
		
		outputs_tr, loss = compiled_update_step(graphs_tuple_tf, Output_data)
		
		print('Echo %d,Iter %d: train_loss is: %.5f train_accuracy is: %.5f'%(echo+1, i+1, tf.reduce_mean(loss),train_accuracy(Output_data,outputs_tr)))
	
		
		if i and i % 10 == 0:
			checkpoint.save(save_prefix)

	




