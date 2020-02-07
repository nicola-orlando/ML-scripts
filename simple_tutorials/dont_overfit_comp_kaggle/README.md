### Collection of scripts and notes for the Kaggle competition https://www.kaggle.com/c/dont-overfit-ii/overview

To do list as 13/01/2020

* Add control plots of the input variables and correlation (currently not possible to use seaborn plotting for my installation). Can make good use of this tutorial https://www.tensorflow.org/tutorials/estimator/boosted_trees_model_understanding. 
* Parse via command line the relevant hyperparameters to optimise, define a proper default value (currently just using what is picked from TF constructor). Pointers for hyperparameter optimisation 
    - https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/ (old now)
    - https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams in plain ternsorflow
    - https://www.curiousily.com/posts/hackers-guide-to-hyperparameter-tuning/ keras build-in functionalities 
* Add cross validation: here want to use cross training with a non standard sample splitting and apply the k-NN algorithm for 
scanning the HP in the validation samples
* I would like to average the BDT with something as simple as possible, maybe a sequence of polynomial models. There are several options in TF, e.g. linear classifiers https://www.tensorflow.org/tutorials/estimator/linear maybe can make use of this https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis. 
* I would not just average everything in one go. I'd try to average models iteratively adding as a threshold the proximity in answer of the classifiers 

**Additional notes on cross-validation implementation**

* Tried HPar, apparently facing this issue https://github.com/tensorflow/tensorboard/issues/2122 althoguh I should be using TensorFlow 2.0
* Might need to try https://scikit-learn.org/stable/modules/cross_validation.html, here the problem is that the splitted panda dataframe is casted into a NumPy array and can't use this for data manipulation as needed for the BDT later on, this might be a solution https://stackoverflow.com/questions/20763012/creating-a-pandas-dataframe-from-a-numpy-array-how-do-i-specify-the-index-colum 
* This tried to use pandaframes https://stackoverflow.com/questions/24999146/pandas-groupby-for-k-fold-cross-validation-with-aggregation/59785310#59785310

**Notes on HP setting for random scan**

* Some notes here https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/
* Other notes here https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

**Notes on don't overfit** 

* https://www.kaggle.com/rafjaa/dealing-with-very-small-datasets 
