# This is a simple script to build a ML model based on Decisionn Trees to be used for the competition 
# https://www.kaggle.com/c/dont-overfit-ii/data. It is largely based on this other script 
# https://github.com/nicola-orlando/tensorflow/blob/master/simple_tutorials/low_statistics_classification.py
# Class in TensorFlow https://www.tensorflow.org/api_docs/python/tf/estimator/BoostedTreesClassifier 
# [1] https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
# [2] https://www.tensorflow.org/datasets/splits
# [3] https://stackoverflow.com/questions/40729162/merging-results-from-model-predict-with-original-pandas-dataframe

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
 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import linear_model

from sklearn.model_selection import RepeatedKFold

# Remove verbose warnings 
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

# Configuration of the job, to be parsed from command line eventually
number_of_folds=10

# PART 1 design the dataset handling for 10-fold cross training
# Now it's time to run, before then, split the dataset accroding to the cross training granularity we want to use 
# See Ref [2]

# My data after downloading will be here /afs/cern.ch/user/o/orlando/.keras/datasets/
# For now hard coded grabbing 
# Data header: id,target,features[0-299]
print("Load the dataset ...")
train_file_path = "/afs/cern.ch/user/o/orlando/.keras/datasets/dont-overfit-ii/train.csv"
dftrain = pd.read_csv(train_file_path)
# Dropping from the dataset the uninteresting index
dftrain = dftrain.drop('id', 1)

# Look at the dataset structure
print("Looking at the input data ...")
print(dftrain.head())
print(dftrain.describe())

# PART 2 design the functions used to handle the HP scan and run functionalites

# Train a logistic regression model performing an HP scan, print the best found HP set and return a model having the best found HP setting 
def train_validate_model_logreg(X_train,y_train,X_test,y_test,full_dataframe,print_verbose=False):
  print("Training a Logistic Regression model")
  logistic = LogisticRegression(solver='saga', tol=0.01, max_iter=200,random_state=0)
  #Some HP combinations not necessarily available
  hp_setting = dict(tol=uniform(loc=0, scale=3),
                    penalty=['l2', 'l1'],
                    C=uniform(loc=0, scale=4),
                    fit_intercept=[True,False],
                    solver=['liblinear', 'saga'],
                    max_iter=uniform(loc=50, scale=200) )
  clf = RandomizedSearchCV(logistic,hp_setting,random_state=0)
  search = clf.fit(X_train,y_train)
  scores_train_prediction=search.predict(X_train)
  if print_verbose :
    print('Best hp setting')
    print(search.best_params_)
    print('Printing predictions')
    print(scores_train_predictions)
    print('Printing random search results')
    print(search.cv_results_)
  score_predictions = search.predict(X_test)
  dataframe_with_scores = pd.DataFrame(data = score_predictions, columns = ['score_predictions_logistic'], index = X_test.index.copy())
  output_dataframe = pd.merge(full_dataframe, dataframe_with_scores, how = 'left', left_index = True, right_index = True)
  return output_dataframe

def train_validate_model_svm(X_train,y_train,X_test,y_test,full_dataframe,print_verbose=False):
    print("Training a SVM model")
    svm = LinearSVC(random_state=0, tol=1e-5)
    # HP setting 
    hp_setting = dict(penalty=['l2'],
                      loss=['hinge', 'squared_hinge'],
                      dual=[True],
                      tol=uniform(loc=0, scale=3),
                      C=uniform(loc=0, scale=4),
                      fit_intercept=[True, False],
                      intercept_scaling=uniform(loc=0, scale=3),
                      max_iter=uniform(loc=50, scale=200))
    clf = RandomizedSearchCV(svm,hp_setting,random_state=0)
    search = clf.fit(X_train,y_train)
    scores_train_prediction=search.predict(X_train)
    if print_verbose :
      print('Best hp setting')
      print(search.best_params_)
      print('Printing predictions')
      print(scores_train_predictions)
      print('Printing random search results')
      print(search.cv_results_)
    score_predictions = search.predict(X_test)
    dataframe_with_scores = pd.DataFrame(data = score_predictions, columns = ['score_predictions_svm'], index = X_test.index.copy())
    output_dataframe = pd.merge(full_dataframe, dataframe_with_scores, how = 'left', left_index = True, right_index = True)
    return output_dataframe

def train_bayesian_rdge(X_train,y_train,X_test,y_test,full_dataframe,print_verbose=False):
  print("Training a Bayesian model")
  bayesian_model = linear_model.BayesianRidge()
  bayesian_model.fit(X_train,y_train)
  score_predictions = bayesian_model.predict(X_test)
  dataframe_with_scores = pd.DataFrame(data = score_predictions, columns = ['score_predictions_bayes'], index = X_test.index.copy())
  output_dataframe = pd.merge(full_dataframe, dataframe_with_scores, how = 'left', left_index = True, right_index = True)
  return output_dataframe

# Now we need to split the data, and call the runner for training/evaluating the model from the splitting look  
X = dftrain.to_numpy()

random_state = 12883823
# See documentation here https://scikit-learn.org/stable/modules/cross_validation.html
# Here for the predict function https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict

fold_count=0
dataframes_with_logreg_scores=[]
dataframes_with_svm_scores=[]
dataframes_with_bayes_scores=[]
repeated_k_fold = RepeatedKFold(n_splits=number_of_folds, n_repeats=1, random_state=random_state)
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
  # values, 1st column as index
  dataframe_from_numpy_train = pd.DataFrame(dftrain,index=train_index,columns=list(dftrain.columns.values))
  dataframe_from_numpy_test = pd.DataFrame(dftrain,index=test_index,columns=list(dftrain.columns.values))
  
  # Pickup the target column 
  y_train = dataframe_from_numpy_train.pop('target')
  y_test = dataframe_from_numpy_test.pop('target')
  
  dataframe_with_logreg_scores = train_validate_model_logreg(dataframe_from_numpy_train,y_train,dataframe_from_numpy_test,y_test,dftrain)
  dataframe_with_svm_scores = train_validate_model_svm(dataframe_from_numpy_train,y_train,dataframe_from_numpy_test,y_test,dftrain)
  dataframe_with_bayes_scores = train_bayesian_rdge(dataframe_from_numpy_train,y_train,dataframe_from_numpy_test,y_test,dftrain)

  dataframes_with_logreg_scores.append(dataframe_with_logreg_scores)
  dataframes_with_svm_scores.append(dataframe_with_svm_scores)
  dataframes_with_bayes_scores.append(dataframe_with_bayes_scores)


combined_dataframe_with_logreg_scores = dataframes_with_logreg_scores[0]
combined_dataframe_with_svm_scores = dataframes_with_svm_scores[0]
combined_dataframe_with_bayes_scores = dataframes_with_bayes_scores[0]

for element in range(0,number_of_folds-1):
    combined_dataframe_with_logreg_scores['score_predictions_logistic'] = combined_dataframe_with_logreg_scores['score_predictions_logistic'].combine_first(dataframes_with_logreg_scores[element+1]['score_predictions_logistic'])
    combined_dataframe_with_svm_scores['score_predictions_svm'] = combined_dataframe_with_svm_scores['score_predictions_svm'].combine_first(dataframes_with_svm_scores[element+1]['score_predictions_svm'])
    combined_dataframe_with_bayes_scores['score_predictions_bayes'] = combined_dataframe_with_bayes_scores['score_predictions_bayes'].combine_first(dataframes_with_bayes_scores[element+1]['score_predictions_bayes'])


print('Printing final dataframe')
print(combined_dataframe_with_logreg_scores.head())
print(combined_dataframe_with_svm_scores.head())
print(combined_dataframe_with_bayes_scores.head())

# Print out the dataframe for investigation
combined_dataframe_with_logreg_scores.to_csv('log_out_final_dataset.csv', index=False) 
combined_dataframe_with_svm_scores.to_csv('svm_out_final_dataset.csv', index=False) 
combined_dataframe_with_bayes_scores.to_csv('bayes_out_final_dataset.csv', index=False) 

matching_labels_logistic = combined_dataframe_with_logreg_scores[combined_dataframe_with_logreg_scores.target == combined_dataframe_with_logreg_scores.score_predictions_logistic]
matching_labels_svm = combined_dataframe_with_svm_scores[combined_dataframe_with_svm_scores.target == combined_dataframe_with_svm_scores.score_predictions_svm]

print('Dataset component with matching labels to predictions')
print(matching_labels_logistic.head(-1))
print(matching_labels_svm.head(-1))

dftrain['score_predictions_logistic'] = combined_dataframe_with_logreg_scores['score_predictions_logistic'] 
dftrain['score_predictions_svm'] = combined_dataframe_with_svm_scores['score_predictions_svm'] 
dftrain['score_predictions_bayes'] = combined_dataframe_with_bayes_scores['score_predictions_bayes'] 

print(dftrain.head(-1))

