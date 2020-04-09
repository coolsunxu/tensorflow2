

from graph_nets import utils_np
from pathlib import Path

import networkx as nx
import numpy as np
import random
import cv2

class Code:
	def __init__(self,path_kind,batch_size,stride,img_width,img_height):
		self.alpha = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a','b',
					  'c','d','e','f','g','h','i','j','k','l','m','n','o','p',
					  'q','r','s','t','u','v','w','x','y','z']
					  
		self.data_root = Path(path_kind)
		self.batch_size = batch_size
		self.img_height = img_height
		self.img_width = img_width
		self.stride = stride
		
		self.block_height = int(img_height / stride)
		self.block_width = int(img_width / stride)
		
		self.load()
		
	def load(self):
		
		self.second_image_paths = list(self.data_root.glob('*'))
		self.second_image_paths=[str(path) for path in self.second_image_paths]
		self.total_number = len(self.second_image_paths) // self.batch_size
	
	
	def mess_up_order(self):
		random.shuffle(self.second_image_paths)
	
	
	def deal_image(self,image):
		graph_nx = nx.OrderedMultiDiGraph()

		# Globals.
		graph_nx.graph["features"] = np.random.randn(36)

		# Nodes.
		for i in range(self.block_height):
			for j in range(self.block_width):
				graph_nx.add_node(i*self.block_width+j, features=image[i*self.stride:(i+1)*self.stride,j*self.stride:(j+1)*self.stride].flatten())
				
				
				
		# Edges.
		for i in range(self.block_height):
			for j in range(self.block_width):
				if i-1>=0 and j>=0:
					graph_nx.add_edge(i*self.block_width+j, (i-1)*self.block_width+j, features=np.random.randn(10))
					graph_nx.add_edge((i-1)*self.block_width+j,i*self.block_width+j, features=np.random.randn(10))
					
				if i+1>=0 and j>=0 and i+1<=self.block_height-1:
					graph_nx.add_edge(i*self.block_width+j, (i+1)*self.block_width+j, features=np.random.randn(10))
					graph_nx.add_edge((i+1)*self.block_width+j,i*self.block_width+j, features=np.random.randn(10))
					
				if i>=0 and j-1>=0:
					graph_nx.add_edge(i*self.block_width+j, i*self.block_width+j-1, features=np.random.randn(10))
					graph_nx.add_edge(i*self.block_width+j-1,i*self.block_width+j, features=np.random.randn(10))
					
				if i>=0 and j+1>=0 and j+1<=self.block_width-1:
					graph_nx.add_edge(i*self.block_width+j, i*self.block_width+j+1, features=np.random.randn(10))
					graph_nx.add_edge(i*self.block_width+j+1,i*self.block_width+j, features=np.random.randn(10))
		
		return graph_nx
		
	def next_batch(self,index):
		graph_dicts = []
		labels = []
		for k,path in enumerate(self.second_image_paths[self.batch_size*index:self.batch_size*(index+1)]) : 
			
			temp_str = path.split('\\')[-1]
			begin=temp_str.find('_')
			end=temp_str.find('.')
			label = self.alpha.index(temp_str[begin+2:end])
			
			img = cv2.imread(path)
			img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			
			img = img / 255.0;
			
			graph_dicts.append(self.deal_image(img))
			labels.append(label)
			
		return utils_np.networkxs_to_graphs_tuple(graph_dicts),np.eye(len(self.alpha))[labels]
