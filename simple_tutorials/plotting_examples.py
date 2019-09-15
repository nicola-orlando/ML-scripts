# Header of the file survived,sex,age,n_siblings_spouses,parch,fare,Class,deck,embark_town,alone
# Define one vector per column you may want to plot
# These are odered according to the file structure 
# There are many other nice tutorials on the web (See for example [1,2,3]). Here I just mixing them up and customising according to my poor taste :) .  
# [1] https://www.tensorflow.org/tutorials/estimators/boosted_trees 
# [2] https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0 
# [3] https://matplotlib.org/tutorials/intermediate/tight_layout_guide.html

from numpy import *

import tensorflow as tf
tf.enable_eager_execution()

import matplotlib.pyplot as plt
import csv

# My data after downloading will be here /afs/cern.ch/user/o/orlando/.keras/datasets/train.csv
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)

# Relevant features and target (survived) of the titanic dataset: https://www.tensorflow.org/beta/tutorials/load_data/csv
survived = []
sex = []
age = []
n_siblings_spouses = []
parch = []
fare = []
Class = []
deck = []
embark_town = []
alone = []

with open(train_file_path,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        survived.append(row[0])
        sex.append(row[1])
        age.append(row[2])
        n_siblings_spouses.append(row[3])
        parch.append(row[4])
        fare.append(row[5])
        Class.append(row[6])
        deck.append(row[7])
        embark_town.append(row[8])
        alone.append(row[9])

print("\nPrinting an example of data column\n")
print(survived)

# After you check that your data looks as expected remove the first element of each vector which corresponds infact to the title of the given feature 
survived.pop(0)
sex.pop(0)
age.pop(0)
n_siblings_spouses.pop(0)
parch.pop(0)
fare.pop(0)
Class.pop(0)
deck.pop(0)
embark_town.pop(0)
alone.pop(0)

print("\nPrinting an example of data column without the title\n")
print(survived)

# Convert data to int
def convert(input_vector):
    for i in range(len(input_vector)): 
        input_vector[i] = float(input_vector[i])
    return input_vector

# Converting all int-like data to floating point precision
survived=convert(survived)
age=convert(age)
n_siblings_spouses=convert(n_siblings_spouses)
parch=convert(parch)
fare=convert(fare)

# Convert categorical data
# Categorical data will be converted to an array of int based on input categories
def convert_cat_data(input_vector, category):
    for i in range(len(input_vector)):
        for cat in range(len(category)):
            if input_vector[i] == category[cat]:
                input_vector[i] = cat
    return input_vector

print("\nPrinting sex before conversion\n")
print(sex)

sex_map=['male', 'female']
sex=convert_cat_data(sex,sex_map)

Class_map=['First', 'Second', 'Third']
Class=convert_cat_data(Class, Class_map)

print("\nPrinting sex after conversion\n")
print(sex)

# Do some plots here, one dimension
# Rearrange the data for making all plots easily
inputs_of_features_to_plot=[sex,Class,survived,age,n_siblings_spouses,parch,fare]
title_of_features_to_plot=['Sex (0=Male, 1=Female)','Class (0=First, 1=Second, 2=Third)',
                           'Survived (0=No, 1=Yes)','Age [y]','#sibling and #spouses','Parch','Fare']
name_of_features_to_plot=['sex_plot.png','Class_plot.png',
                          'survived_plot.png','age_plot.png','n_siblings_spouses_plot.png','parch_plot.png','fare_plot.png']

index=0
for feature in inputs_of_features_to_plot: 
    plt.hist(feature, density=True)
    plt.ylabel('Density hisogram')
    plt.xlabel(title_of_features_to_plot[index])
    plt.savefig(name_of_features_to_plot[index])
    index=index+1
    plt.clf()    

# Now we want to do something more tricky 
# Try to split each array in two components, one corresponds to entries with survived==1 and another with survived==0
# This is because "survived" is the field we want to predict  
def splitted_feature(input_vector,survived_feature):
    vector_survived=[]
    vector_not_survived=[]
    for i in range(len(input_vector)): 

        if survived_feature[i]==0:
            vector_not_survived.append(input_vector[i])

        if survived_feature[i]==1:
            vector_survived.append(input_vector[i])

    splitted_vector=[vector_survived,vector_not_survived]
    return splitted_vector

# Split the arrays into survived / non-survived components    
sex_splitted=splitted_feature(sex,survived)
print("\nPrinting array with survived people\n")
print(sex_splitted[0])
print("\nPrinting array with not survived people\n")
print(sex_splitted[1])

Class_splitted=splitted_feature(Class,survived)
n_siblings_spouses=splitted_feature(n_siblings_spouses,survived)

legend=['Survived', 'Not survived']
colors = ['#E69F00', '#56B4E9']

plt.subplot(2, 1, 1)
plt.hist([sex_splitted[0], sex_splitted[1]], density=True, color = colors, label=legend)
plt.ylabel('Density hisogram')
plt.xlabel('Sex (0=Male, 1=Female)')
plt.legend()

plt.subplot(2, 1, 2)
plt.hist([Class_splitted[0], Class_splitted[1]], density=True, color = colors, label=legend)
plt.ylabel('Density hisogram')
plt.xlabel('Class (0=First, 1=Second, 2=Third)')
plt.legend()

# From this great tutorial https://matplotlib.org/tutorials/intermediate/tight_layout_guide.html
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.savefig("survival_sex_Class.png")
