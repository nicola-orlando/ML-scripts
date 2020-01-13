# This is a simple script to build a ML model based on Decisionn Trees to be used for the competition 
# https://www.kaggle.com/c/dont-overfit-ii/data. It is largely based on this other script 
# https://github.com/nicola-orlando/tensorflow/blob/master/simple_tutorials/low_statistics_classification.py
# Class in TensorFlow https://www.tensorflow.org/api_docs/python/tf/estimator/BoostedTreesClassifier 

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import uproot
import tensorflow as tf
tf.enable_eager_execution()

from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

import pandas as pd

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(123)

# My data after downloading will be here /afs/cern.ch/user/o/orlando/.keras/datasets/
# For now hard coded grabbing 
# Data header: id,target,features[0-299]
print("Load the dataset ...\n")
train_file_path = "/afs/cern.ch/user/o/orlando/.keras/datasets/dont-overfit-ii/train.csv"
dftrain = pd.read_csv(train_file_path)
# Dropping from the dataset the uninteresting index
dftrain = dftrain.drop('id', 1)

# Look at the dataset structure
print("Looking at the input data ...\n")
print(dftrain.head())
print(dftrain.describe())

# Pickup the target column 
y_train = dftrain.pop('target')

# Input layer, need to define feature_column 
fc = tf.feature_column

# Fill up the inputs 
feature_columns = []
for index in range(0,300) :
    #print("Appending feature column number", str(index))
    feature_columns.append(fc.numeric_column(str(index),dtype=tf.float32))

example = dftrain.head(1)
print("Features enetering in the training ...\n")
print(example)

# Build the input layer 
fc.input_layer(dict(example), feature_columns).numpy()
# Use entire batch since this is such a small dataset.                                                                                                      
NUM_EXAMPLES = len(y_train)

# Create the correlation matrix and check if for trivial patters to simplify the problem
#corr_matrix = dftrain.corr()
#print (corr_matrix)

# Build the input function
def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    # For training, cycle thru dataset as many times as need (n_epochs=None).                                                                              #
    dataset = dataset.repeat(n_epochs)
    # In memory training doesn't use batching.                                                                                                             #
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn

# Training and evaluation input functions.                                                                                                                  
print("Defining the training function ...\n")
train_input_fn = make_input_fn(dftrain, y_train)
#eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)
print("end of function definition")
print(train_input_fn)

n_batches = 5
print("Training the model ...\n")

# Properties to be set 
# __________________________________

in_n_batches_per_layer=1
in_n_trees=100
in_max_depth=6
in_learning_rate=0.1
in_l1_regularization=0.0
in_l2_regularization=0.0
in_tree_complexity=0.0
in_min_node_weight=0.0
in_center_bias=False
in_pruning_mode='none'
in_quantile_sketch_epsilon=0.01

est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                          n_batches_per_layer = in_n_batches_per_layer,    
                                          n_trees = in_n_trees,                
                                          max_depth = in_max_depth,              
                                          learning_rate = in_learning_rate,          
                                          l1_regularization = in_l1_regularization,      
                                          l2_regularization = in_l2_regularization,     
                                          tree_complexity = in_tree_complexity,        
                                          min_node_weight = in_min_node_weight,        
                                          center_bias = in_center_bias,            
                                          pruning_mode = in_pruning_mode,           
                                          quantile_sketch_epsilon = in_quantile_sketch_epsilon)

# The model will stop training once the specified number of trees is built, not 
# based on the number of steps.
# Cavieat: max_steps can give run time errors, attempting a few times before it actually works 
est.train(input_fn=train_input_fn,max_steps=10)
