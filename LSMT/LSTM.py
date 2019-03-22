# encoding: UTF-8
import tensorflow as tf
D = 1 
num_unrollings = 50 
batch_size = 500 
num_nodes = [200,200,150] 
n_layers = len(num_nodes) 
dropout = 0.2 

tf.reset_default_graph()