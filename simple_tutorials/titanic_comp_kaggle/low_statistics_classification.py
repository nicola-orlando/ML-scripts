# This script is a mixture of two tensorflow tutorials https://www.tensorflow.org/beta/tutorials/load_data/from_pandas and https://www.tensorflow.org/tutorials/estimators/boosted_trees. 
# The main idea is to compare a feedforward NN performace to linear classifier for a problem which is relatively simple and has low 
# statistics for the training. You can inspect the dataset to convince yourself :) . 
# This script also illustrate the differences when in terms of data structures expected by a neural network from Keras and a classifier 
# from tensorflow which ineriths from Estimators. 

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

# Load data, note that this is then stored and picked from a #{HOME}/.keras/dataset                                                                         
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# Print data path to the dataset for inspection                                                                                                             
print("\nPrinting dataset path")
print(train_file_path)

# Convert from csv to pandas dataframe                                                                                                                      
df = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)

# To be used for the BDT and linear classifiers
y_train = df.pop('survived')
y_eval = df_test.pop('survived')

# PART 1. NEURAL NETWORK 
# _______________________________

# Convert categorical data objects into numbers                                                                                                            
categorical_data = ['sex','Class','deck','embark_town','alone']
for cat_data_to_cnv in categorical_data :
  print ("Handling now data category "+cat_data_to_cnv)
  df[cat_data_to_cnv] = pd.Categorical(df[cat_data_to_cnv])
  df_test[cat_data_to_cnv] = pd.Categorical(df_test[cat_data_to_cnv])

df['sex'] = df.sex.cat.codes
df['Class'] = df.Class.cat.codes
df['deck'] = df.deck.cat.codes
df['embark_town'] = df.embark_town.cat.codes
df['alone'] = df.alone.cat.codes

df_test['sex'] = df_test.sex.cat.codes
df_test['Class'] = df_test.Class.cat.codes
df_test['deck'] = df_test.deck.cat.codes
df_test['embark_town'] = df_test.embark_town.cat.codes
df_test['alone'] = df_test.alone.cat.codes

target = y_train
target_test = y_eval

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
dataset_test = tf.data.Dataset.from_tensor_slices((df_test.values, target_test.values))

# Shuffle and batch.                                                                                                                                        
# Shuffling is described in the QueueBase class https://www.tensorflow.org/api_docs/python/tf/queue/QueueBase, not necessarily common in all applications   
# Batch is standard, e.g. nice description here https://stackoverflow.com/questions/41175401/what-is-a-batch-in-tensorflow                                  
train_dataset = dataset.shuffle(len(df)).batch(1)
dataset_test = dataset_test.shuffle(len(df_test)).batch(1)

# Define a NN model                                                                                                                                      
model_nn = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_nn.fit(train_dataset.make_one_shot_iterator(), verbose=1, epochs=200, steps_per_epoch=1)
test_loss, test_accuracy = model_nn.evaluate(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)

# END OF PART 1. NEURAL NETWORK 
# _______________________________


# PART 2. LINEAR CLASSIFIERS
# _______________________________

# The BDT and other linear classifiers hinerit from Estimator class. This requires a feature colum at the initialiation and a dataset function

# Definition of the feature_columns
print("Building feature_columns")
fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'Class', 'deck', 
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
  
def one_hot_cat_column(feature_name, vocab):
  return fc.indicator_column(
      fc.categorical_column_with_vocabulary_list(feature_name,vocab))

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  # Need to one-hot encode categorical features.
  vocabulary = df[feature_name].unique()
  feature_columns.append(one_hot_cat_column(feature_name, vocabulary))
  
for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(fc.numeric_column(feature_name, dtype=tf.float32))

example = df.head(1)
class_fc = one_hot_cat_column('Class', ('First', 'Second', 'Third'))
print('Feature value: "{}"'.format(example['Class'].iloc[0]))
fc.input_layer(dict(example), feature_columns).numpy()

# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    # For training, cycle thru dataset as many times as need (n_epochs=None).    
    dataset = dataset.repeat(n_epochs)  
    # In memory training doesn't use batching.
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn

# Training and evaluation input functions.
train_input_fn = make_input_fn(df, y_train)
eval_input_fn = make_input_fn(df_test, y_eval, shuffle=False, n_epochs=1)

# Define the linear model 
linear_est = tf.estimator.LinearClassifier(feature_columns)
# Train the linear model 
linear_est.train(train_input_fn, max_steps=100)
# Evaluation for the linear model
results = linear_est.evaluate(eval_input_fn)

# Define the BDT
n_batches = 5
est = tf.estimator.BoostedTreesClassifier(feature_columns,n_batches_per_layer=n_batches)
# The model will stop training once the specified number of trees is built, not 
# based on the number of steps.
# Cavieat: max_steps can give run time errors, attempting a few times before it actually works 
est.train(train_input_fn, max_steps=10)

# Evaluation of a BDT model
results_bdt = est.evaluate(eval_input_fn)
print("\n\nComparing the performance of different classifiers")
print('Accuracy linear model: ', results['accuracy'])
print('Accuracy BDT model : ', results_bdt['accuracy'])
print('Dummy model accuracy: ', results_bdt['accuracy_baseline'])
print('Accuracy NN model : ', test_accuracy)


print("\n\nPrinting training model")
print(train_input_fn)

print("\n\nPrinting feature_columns")
print(feature_columns)
