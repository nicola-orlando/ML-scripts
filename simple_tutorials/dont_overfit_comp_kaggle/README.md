### Collection of scripts and notes for a Kaggle competition 
https://www.kaggle.com/c/dont-overfit-ii/overview

**Done as 23/03/2020**

* Implemented three classifiers: logistic regression, svm classifier, bayesian classifier 
* Implemented cross validation 
* Propagated the prediction to the original dataset, current accuracy 70% (target is 80%)

**To do list as 23/03/2020**

* Add control plots of the input variables and correlation (currently not possible to use seaborn plotting for my installation). Can make good use of this tutorial https://www.tensorflow.org/tutorials/estimator/boosted_trees_model_understanding. 
* I would like to average the BDT with something as simple as possible, maybe a sequence of polynomial models. There are several options in TF, e.g. linear classifiers https://www.tensorflow.org/tutorials/estimator/linear maybe can make use of this https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis. 
* I would not just average everything in one go. I'd try to average models iteratively adding as a threshold the proximity in answer of the classifiers 
* Add more simple classifiers, I noticed that the cases with misclassification occur frequently with logistic refression and svm giving same answer, maybe the data where the classification fails is a sort of fake data? Can try perhaps boosted trees to recover that 


**Notes on HP setting for random scan**

* Some notes here https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/
* Other notes here https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

**Notes on don't overfit** 

* https://www.kaggle.com/rafjaa/dealing-with-very-small-datasets (including features elimination model from sklearn.feature_selection import RFE)
* Dataset augmentation SMOTE (Synthetic Minority Oversampling TEchnique) 

