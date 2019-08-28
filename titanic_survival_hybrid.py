# This oversimplified example is meant to be an illustration of 1) Collecting data from csv, 2) Manipulating/accessing it, 3) Convert categorical data
# 4) Train a simple model 5) Basic NN inspection and predictions. 

# This script aim at accomplishing the same task as in this tutorial https://www.tensorflow.org/beta/tutorials/load_data/csv while loading and 
# handling data by means of pandas dataframes as illustrated in this tutorial https://www.tensorflow.org/beta/tutorials/load_data/from_pandas. 
# Here you will see, how to load csv data and handle it, how to create a simple NN model

# Before you start please change 'class' into 'Class' in the title of one of the columns of your data (to be downloaded following the code below). 
# This is because 'class' is a keyword in python while in this dataset refers to the ticket class. 
# Also please checkout where your csv file is saved, should be in $HOME/.keras/dataset or something similar

# Data available and format
# survived                int64
# sex                    object
# age                   float64
# n_siblings_spouses      int64
# parch                   int64
# fare                  float64
# class                  object
# deck                   object
# embark_town            object
# alone                  object

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import uproot
import tensorflow as tf
tf.enable_eager_execution()

from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

import pandas as pd

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

print("\nPrining head of the file to see how it looks")
print(df.head())
print("\nPrining data types")
print(df.dtypes)

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

# Print the head of your dataset after conversion of strings into numbers
print("\nPrining tain sample header")
print(df.head())
print("\nPrining test sample header")
print(df_test.head())

target = df.pop('survived')
target_test = df_test.pop('survived')

print("\nPrinting now data associated to the target")
print(target.values)

print("\nPrinting now data associated to the target in the testing sample")
print(target_test.values)

print("\nPrinting features vectors")
print(df.values)

print("\nPrinting features vectors in the testing sample")
print(df_test.values)

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
dataset_test = tf.data.Dataset.from_tensor_slices((df_test.values, target_test.values))

# Print the actual dataset you will be using for training 
print("\nPrinting feature,target for the training sample")
for feature, target in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feature, target))

# Print the actual dataset you will be using for training 
print("\nPrinting feature,target for the testing sample")
for feature, target in dataset_test.take(5):
  print ('Features: {}, Target: {}'.format(feature, target))

# Shuffle and batch. 
# Shuffling is described in the QueueBase class https://www.tensorflow.org/api_docs/python/tf/queue/QueueBase, not necessarily common in all applications
# Batch is standard, e.g. nice description here https://stackoverflow.com/questions/41175401/what-is-a-batch-in-tensorflow
train_dataset = dataset.shuffle(len(df)).batch(5)
dataset_test = dataset_test.shuffle(len(df_test)).batch(5)

# Define the model 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_dataset.make_one_shot_iterator(), verbose=1, epochs=200, steps_per_epoch=1)

# Print model performance indicatiors 
test_loss, test_accuracy = model.evaluate(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)
print('\n\nTest Loss {}, Test Accuracy {}\n\n'.format(test_loss, test_accuracy))

predictions = model.predict(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)

print(predictions) 
print( list(dataset_test)[0][1][:10] ) 

for prediction, survived in zip(predictions[:10], list(dataset_test)[0][1][:10]):
  print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))
