# This oversimplified example is meant as a comparative example of neural networks trained with different regularisation schemes. 
# It starts as a cut and paste of https://github.com/nicola-orlando/tensorflow/blob/master/simple_tutorials/titanic_survival_hybrid.py
# Please see the header of the script above for more information. 
# Here I test different regularisation methods. If you try to run it you will see that the results don't follow the pattern expected from 
# the publication http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf because the NN hyperparameter setting is not optimised; rather it 
# shows some simple sintax that can be used to define the relevant NN architecture. 

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

from keras import regularizers
from keras.constraints import max_norm

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

target = df.pop('survived')
target_test = df_test.pop('survived')

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

# Define regularised model, L1 regularisation 
model_reg_l1 = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu',kernel_regularizer=regularizers.l1(0.01), activity_regularizer=regularizers.l1(0.01) ),
  tf.keras.layers.Dense(10, activation='relu',kernel_regularizer=regularizers.l1(0.01), activity_regularizer=regularizers.l1(0.01) ),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model_reg_l1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_reg_l1.fit(train_dataset.make_one_shot_iterator(), verbose=1, epochs=200, steps_per_epoch=1)

# Define regularised model, L2 regularisation 
model_reg_l2 = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu',kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01) ),
  tf.keras.layers.Dense(10, activation='relu',kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01) ),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model_reg_l2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_reg_l2.fit(train_dataset.make_one_shot_iterator(), verbose=1, epochs=200, steps_per_epoch=1)

# Define regularised model, max-norm
model_reg_mn = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', kernel_constraint=max_norm(2.) ),
  tf.keras.layers.Dense(10, activation='relu', kernel_constraint=max_norm(2.) ),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model_reg_mn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_reg_mn.fit(train_dataset.make_one_shot_iterator(), verbose=1, epochs=200, steps_per_epoch=1)

# Define regularised model, max-norm plus L2
model_reg_mnl2 = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', kernel_constraint=max_norm(2.), 
                        kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01) ),
  tf.keras.layers.Dense(10, activation='relu', kernel_constraint=max_norm(2.), 
                        kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01) ),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model_reg_mnl2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_reg_mnl2.fit(train_dataset.make_one_shot_iterator(), verbose=1, epochs=200, steps_per_epoch=1)

# Print model performance indicatiors 
test_loss, test_accuracy = model.evaluate(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)
test_loss_reg_l1, test_accuracy_reg_l1 = model_reg_l1.evaluate(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)
test_loss_reg_l2, test_accuracy_reg_l2 = model_reg_l2.evaluate(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)
test_loss_reg_mn, test_accuracy_reg_mn = model_reg_mn.evaluate(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)
test_loss_reg_mnl2, test_accuracy_reg_mnl2 = model_reg_mnl2.evaluate(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)

print('\n\nTest Loss {}, Test Accuracy {}\n\n'.format(test_loss, test_accuracy))
print('\n\nTest Loss {}, Test Accuracy {} (L1 regularised model)\n\n'.format(test_loss_reg_l1, test_accuracy_reg_l1))
print('\n\nTest Loss {}, Test Accuracy {} (L2 regularised model)\n\n'.format(test_loss_reg_l2, test_accuracy_reg_l2))
print('\n\nTest Loss {}, Test Accuracy {} (MN regularised model)\n\n'.format(test_loss_reg_mn, test_accuracy_reg_mn))
print('\n\nTest Loss {}, Test Accuracy {} (MNL2 regularised model)\n\n'.format(test_loss_reg_mnl2, test_accuracy_reg_mnl2))

predictions = model.predict(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)
predictions_reg_l1 = model_reg_l1.predict(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)
predictions_reg_l2 = model_reg_l2.predict(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)
predictions_reg_mn = model_reg_mn.predict(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)
predictions_reg_mnl2 = model_reg_mnl2.predict(dataset_test.make_one_shot_iterator(), verbose=1, steps=20)

print(predictions) 
print( list(dataset_test)[0][1][:10] ) 

for prediction, survived in zip(predictions[:10], list(dataset_test)[0][1][:10]):
  print("Predicted survival default NN model: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))

for prediction, survived in zip(predictions_reg_l1[:10], list(dataset_test)[0][1][:10]):
  print("Predicted survival L1 regularised model: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))

for prediction, survived in zip(predictions_reg_l2[:10], list(dataset_test)[0][1][:10]):
  print("Predicted survival L2 regularised model: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))

for prediction, survived in zip(predictions_reg_mn[:10], list(dataset_test)[0][1][:10]):
  print("Predicted survival max-norm model: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))

for prediction, survived in zip(predictions_reg_mnl2[:10], list(dataset_test)[0][1][:10]):
  print("Predicted survival max-norm+L2 regularisation: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))
