# This is a simple script to build a ML model based on Decisionn Trees to be used for the competition 
# https://www.kaggle.com/c/dont-overfit-ii/data. It is largely based on this other script 
# https://github.com/nicola-orlando/tensorflow/blob/master/simple_tutorials/low_statistics_classification.py
# Class in TensorFlow https://www.tensorflow.org/api_docs/python/tf/estimator/BoostedTreesClassifier 
# [1] https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
# [2] https://www.tensorflow.org/datasets/splits

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import uproot
import tensorflow as tf
tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(123)

from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from scipy.stats import uniform

import pandas as pd

# HP scan
from sklearn.model_selection import RandomizedSearchCV

# Logistic regression model 
from sklearn.linear_model import LogisticRegression

# Remove verbose warnings 
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

#metric = 'accuracy'

# Function used to store the log files of each run, taken from Ref [1] 
#with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
#  hp.hparams_config(
#    hparams=[hp_scan_n_batches_per_layer, hp_scan_n_trees, hp_scan_max_depth, hp_scan_learning_rate, hp_scan_l1_regularization, 
#             hp_scan_l2_regularization, hp_scan_tree_complexity, hp_scan_min_node_weight, hp_scan_center_bias, hp_scan_pruning_mode, 
#             hp_scan_quantile_sketch_epsilon],
#    metrics=[hp.Metric(metric, display_name='Accuracy')],
#  )

# PART 1 design the dataset handling for 10-fold cross training
# Now it's time to run, before then, split the dataset accroding to the cross training granularity we want to use 
# See Ref [2]

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

# Build the input function, used for DecisionTreesClassifier 
def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    # For training, cycle through dataset as many times as need (n_epochs=None).                                                                           
    dataset = dataset.repeat(n_epochs)
    # In memory training doesn't use batching.                                                                                                             
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn

# Now we need to split the data, and call the runner for training/evaluating the model from the splitting look  
import numpy as np
from sklearn.model_selection import RepeatedKFold
X = dftrain.to_numpy()


# PART 2 design the functions used to handle the HP scan and run functionalites
# ____________________________________________

# Train a logistic regression model performing an HP scan, print the best found HP set and return a model having the best found HP setting 
def train_validate_model_logreg(X,y):
  print("Training a Logistic Regression model")
  logistic = LogisticRegression(solver='saga', tol=0.01, max_iter=200,random_state=0)
  #Some HP combinations not necessarily available
  hp_setting = dict(tol=uniform(loc=0, scale=3),
                    penalty=['l2', 'l1'],
                    C=uniform(loc=0, scale=4),
                    fit_intercept=[True,False],
                    solver=['liblinear', 'saga'],
                    max_iter=uniform(loc=50, scale=200) )
  clf = RandomizedSearchCV(logistic, hp_setting, random_state=0)
  search = clf.fit(X,y)
  ypredicts=search.predict(X)
  print('Best hp setting')
  print(search.best_params_)
  logistic_return = LogisticRegression()
  logistic_return.set_params(**search.best_params_)

  print('Printing predictions')
  print(ypredicts)
  print('Printing random search results')
  print(search.cv_results_)
  #print('Output model')
  #print(logistic_return)
  return logistic_return

def train_validate_model_svm(X,y):
    print("\nTraining a SVM model ...")
    svm = LinearSVC(random_state=0, tol=1e-5)
    # HP setting 
    hp_setting = dict(penalty=['l2', 'l1'],
                      loss=['hinge', 'squared_hinge'],
                      dual=[True, False],
                      tol=uniform(loc=0, scale=3),
                      C=uniform(loc=0, scale=4),
                      fit_intercept=[True, False],
                      intercept_scaling=uniform(loc=0, scale=3),
                      max_iter=uniform(loc=50, scale=200))

random_state = 12883823
# See documentation here https://scikit-learn.org/stable/modules/cross_validation.html
# Here for the predict function https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict

fold_count=0
repeated_k_fold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=random_state)
for train_index, test_index in repeated_k_fold.split(X):
  fold_count+=1
  print("Processing fold count ",fold_count)

  # Input layer, need to define feature_column 
  fc = tf.feature_column
  feature_columns = []
  # Loop over all features
  for index in range(0,300) :
    feature_columns.append(fc.numeric_column(str(index),dtype=tf.float32))

  # Converting to dataframe
  dataframe_from_numpy_train = pd.DataFrame(dftrain,    # values
                                            index=train_index,    # 1st column as index
                                            columns=list(dftrain.columns.values))
  
  # Pickup the target column 
  y_train = dataframe_from_numpy_train.pop('target')
  
  logistic_best_model = train_validate_model_logreg(dataframe_from_numpy_train,y_train)
  print('Printing now best model pars as obtained from the function')
  print(logistic_best_model.get_params)
  
  predictions = logistic_best_model.predict(dataframe_from_numpy_train)

  #score = model.score(x_test, y_test)
  #print(score)
  df['predictions'] = predictions





  example = dataframe_from_numpy_train.head(1)

  # Build the input layer 
  fc.input_layer(dict(example), feature_columns).numpy()
  # Use entire batch since this is such a small dataset.                                                                                                   
  NUM_EXAMPLES = len(y_train)

  # Training and evaluation input functions.                                                                                                               
  #print("Defining the training function ...\n")
  train_input_fn = make_input_fn(dataframe_from_numpy_train, y_train)

  



  

# Function used to train and test the BDT model with a given HP setting
# The goal is to return accuracy and the hyperparameters setting associated to that
#def train_validate_model_bdt():
#  model_hp=dict(
#  )

  # Define the model
#  est = tf.estimator.BoostedTreesClassifier(feature_columns,
#                                            n_batches_per_layer = hparams[hp_scan_n_batches_per_layer],    
#                                            n_trees = hparams[hp_scan_n_trees])                




#  est = tf.estimator.BoostedTreesClassifier(feature_columns,
#                                            n_batches_per_layer = hparams[hp_scan_n_batches_per_layer],    
#                                            n_trees = hparams[hp_scan_n_trees],                
#                                            max_depth = hparams[hp_scan_max_depth],          
#                                            learning_rate = hparams[hp_scan_learning_rate],                
#                                            l1_regularization = hparams[hp_scan_l1_regularization],                
#                                            l2_regularization = hparams[hp_scan_l2_regularization],                
#                                            tree_complexity = hparams[hp_scan_tree_complexity],                
#                                            min_node_weight = hparams[hp_scan_min_node_weight],                
#                                            center_bias = hparams[hp_scan_center_bias],                
#                                            pruning_mode = hparams[hp_scan_pruning_mode],                
#                                            quantile_sketch_epsilon = hparams[hp_scan_quantile_sketch_epsilon])                

  # Train, on the given input function  
  # The model will stop training once the specified number of trees is built, not 
  # based on the number of steps.
  # Cavieat: max_steps can give run time errors, attempting a few times before it actually works   
#  est.train(input_fn=train_input_fn,max_steps=10)
  # Test on the given test function 
#  results_bdt = est.evaluate(eval_input_fn)
#  return results_bdt['accuracy']

# Function to run a given job and store the relvant info in a log file
#def run(run_dir, hparams):
#  with tf.summary.create_file_writer(run_dir).as_default():
#    hp.hparams(hparams)  # Record the values used in this trial
#    accuracy = train_validate_model(hparams)
#    tf.summary.scalar(metric, accuracy, step=1)



#train_validate_model_logreg(hparams):

#train_validate_model_linear(hparams):
