# Collection of scripts and notes for the Kaggle competition https://www.kaggle.com/c/dont-overfit-ii/overview

To do list as 13/01/2020

* Add control plots of the input variables and correlation (currently not possible to use seaborn plotting for my installation). Can make good use of this tutorial https://www.tensorflow.org/tutorials/estimator/boosted_trees_model_understanding. 
* Parse via command line the relevant hyperparameters to optimise, define a proper default value (currently just using what is picked from TF constructor) 
* Add cross validation: here want to use cross training with a non strandard sample splitting and apply the k-NN algorithm for scanning the HP in the validation samples
* I would like to average the BDT with something as simple as possible, maybe a sequence of polinominal models. There are several options in TF, e.g. linear classifiers https://www.tensorflow.org/tutorials/estimator/linear maybe can make use of this https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis. 
* I would not just average everything in one go. I'd try to average models iteratively adding as a treshold the proximity in answer of the classifiers 