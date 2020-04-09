
	
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf
import graph_nets as gn

globals_0 = [1., 2., 3.]

# Node features for graph 0.
nodes_0 = [[10., 20., 30.],  # Node 0
           [11., 21., 31.],  # Node 1
           [12., 22., 32.],  # Node 2
           [13., 23., 33.],  # Node 3
           [14., 24., 34.]]  # Node 4

# Edge features for graph 0.
edges_0 = [[100., 200.],  # Edge 0
           [101., 201.],  # Edge 1
           [102., 202.],  # Edge 2
           [103., 203.],  # Edge 3
           [104., 204.],  # Edge 4
           [105., 205.]]  # Edge 5
print(type(edges_0))
# The sender and receiver nodes associated with each edge for graph 0.
senders_0 = [0,  # Index of the sender node for edge 0
             1,  # Index of the sender node for edge 1
             1,  # Index of the sender node for edge 2
             2,  # Index of the sender node for edge 3
             2,  # Index of the sender node for edge 4
             3]  # Index of the sender node for edge 5
receivers_0 = [1,  # Index of the receiver node for edge 0
               2,  # Index of the receiver node for edge 1
               3,  # Index of the receiver node for edge 2
               0,  # Index of the receiver node for edge 3
               3,  # Index of the receiver node for edge 4
               4]  # Index of the receiver node for edge 5

# Global features for graph 1.
globals_1 = [1001., 1002., 1003.]

# Node features for graph 1.
nodes_1 = [[1010., 1020., 1030.],  # Node 0
           [1011., 1021., 1031.]]  # Node 1

# Edge features for graph 1.
edges_1 = [[1100., 1200.],  # Edge 0
           [1101., 1201.],  # Edge 1
           [1102., 1202.],  # Edge 2
           [1103., 1203.]]  # Edge 3

# The sender and receiver nodes associated with each edge for graph 1.
senders_1 = [0,  # Index of the sender node for edge 0
             0,  # Index of the sender node for edge 1
             1,  # Index of the sender node for edge 2
             1]  # Index of the sender node for edge 3
receivers_1 = [0,  # Index of the receiver node for edge 0
               1,  # Index of the receiver node for edge 1
               0,  # Index of the receiver node for edge 2
               0]  # Index of the receiver node for edge 3

data_dict_0 = {
    "globals": globals_0,
    "nodes": nodes_0,
    "edges": edges_0,
    "senders": senders_0,
    "receivers": receivers_0
}

data_dict_1 = {
    "globals": globals_1,
    "nodes": nodes_1,
    "edges": edges_1,
    "senders": senders_1,
    "receivers": receivers_1
}

data_dict_list = [data_dict_0, data_dict_1]
graphs_tuple = utils_np.data_dicts_to_graphs_tuple(data_dict_list)

graphs_nx = utils_np.graphs_tuple_to_networkxs(graphs_tuple)
_, axs = plt.subplots(ncols=2, figsize=(6, 3))
for iax, (graph_nx, ax) in enumerate(zip(graphs_nx, axs)):
	nx.draw(graph_nx, ax=ax)
	ax.set_title("Graph {}".format(iax))
#plt.show()

def plot_graphs_tuple_np(graphs_tuple):
  networkx_graphs = utils_np.graphs_tuple_to_networkxs(graphs_tuple)
  num_graphs = len(networkx_graphs)
  _, axes = plt.subplots(1, num_graphs, figsize=(5*num_graphs, 5))
  if num_graphs == 1:
    axes = axes,
  for graph, ax in zip(networkx_graphs, axes):
    plot_graph_networkx(graph, ax)

def plot_graph_networkx(graph, ax, pos=None):
  node_labels = {node: "{:.3g}".format(data["features"][0])
                 for node, data in graph.nodes(data=True)
                 if data["features"] is not None}
  edge_labels = {(sender, receiver): "{:.3g}".format(data["features"][0])
                 for sender, receiver, data in graph.edges(data=True)
                 if data["features"] is not None}
  global_label = ("{:.3g}".format(graph.graph["features"][0])
                  if graph.graph["features"] is not None else None)

  if pos is None:
    pos = nx.spring_layout(graph)
  nx.draw_networkx(graph, pos, ax=ax, labels=node_labels)

  if edge_labels:
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax)

  if global_label:
    plt.text(0.05, 0.95, global_label, transform=ax.transAxes)

  ax.yaxis.set_visible(False)
  ax.xaxis.set_visible(False)
  return pos


    
def print_graphs_tuple(graphs_tuple):
	print("Shapes of `GraphsTuple`'s fields:")
	print(graphs_tuple.map(lambda x: x if x is None else x.shape, fields=graphs.ALL_FIELDS))
	print("\nData contained in `GraphsTuple`'s fields:")
	print("globals:\n{}".format(graphs_tuple.globals))
	print("nodes:\n{}".format(graphs_tuple.nodes))
	print("edges:\n{}".format(graphs_tuple.edges))
	print("senders:\n{}".format(graphs_tuple.senders))
	print("receivers:\n{}".format(graphs_tuple.receivers))
	print("n_node:\n{}".format(graphs_tuple.n_node))
	print("n_edge:\n{}".format(graphs_tuple.n_edge))

print_graphs_tuple(graphs_tuple)

recovered_data_dict_list = utils_np.graphs_tuple_to_data_dicts(graphs_tuple)

#print(recovered_data_dict_list)

tf.reset_default_graph()

graph_dicts = data_dict_list
OUTPUT_EDGE_SIZE = 10
OUTPUT_NODE_SIZE = 11
OUTPUT_GLOBAL_SIZE = 12
graph_network = modules.GraphNetwork(
    edge_model_fn=lambda: snt.Linear(output_size=OUTPUT_EDGE_SIZE),
    node_model_fn=lambda: snt.Linear(output_size=OUTPUT_NODE_SIZE),
    global_model_fn=lambda: snt.Linear(output_size=OUTPUT_GLOBAL_SIZE))


input_graphs = utils_tf.data_dicts_to_graphs_tuple(graph_dicts)

print(len(graph_dicts))
print(graph_dicts)
print(input_graphs)
output_graphs = graph_network(input_graphs)
print(output_graphs.globals)

#print_graphs_tuple(output_graphs)


def zeros_graph(sample_graph, edge_size, node_size, global_size):
  zeros_graphs = sample_graph.replace(nodes=None, edges=None, globals=None)
  zeros_graphs = utils_tf.set_zero_edge_features(zeros_graphs, edge_size)
  zeros_graphs = utils_tf.set_zero_node_features(zeros_graphs, node_size)
  zeros_graphs = utils_tf.set_zero_global_features(zeros_graphs, global_size)
  return zeros_graphs

tf.reset_default_graph()

mlp = snt.Sequential([
    snt.Linear(1024),
    tf.nn.relu,
    snt.Linear(OUTPUT_EDGE_SIZE),
])

graph_network = modules.GraphNetwork(
    edge_model_fn=lambda: mlp,#snt.Linear(output_size=OUTPUT_EDGE_SIZE),
    node_model_fn=lambda: snt.Linear(output_size=OUTPUT_NODE_SIZE),
    global_model_fn=lambda: snt.Linear(output_size=OUTPUT_GLOBAL_SIZE))

input_graphs = utils_tf.data_dicts_to_graphs_tuple(graph_dicts)
initial_state = zeros_graph(
    input_graphs, OUTPUT_EDGE_SIZE, OUTPUT_NODE_SIZE, OUTPUT_GLOBAL_SIZE)

num_recurrent_passes = 3

current_state = initial_state
for unused_pass in range(num_recurrent_passes):
  input_and_state_graphs = utils_tf.concat(
      [input_graphs, current_state], axis=1)
  current_state = graph_network(input_and_state_graphs)
output_graphs = current_state

"""
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    graphs_tuple_np = sess.run(output_graphs)
    
plot_graphs_tuple_np(graphs_tuple_np)

plt.show()

print(graphs_tuple_np)

recovered_data_dict_list = utils_np.graphs_tuple_to_data_dicts(graphs_tuple_np)
print(len(recovered_data_dict_list))

print(recovered_data_dict_list[1]['edges'])

"""


print("Output edges size: {}".format(output_graphs.edges.shape[-1]))  # Equal to OUTPUT_EDGE_SIZE
print("Output nodes size: {}".format(output_graphs.nodes.shape[-1]))  # Equal to OUTPUT_NODE_SIZE
print("Output globals size: {}".format(output_graphs.globals.shape[-1]))  # Equal to OUTPUT_GLOBAL_SIZE









