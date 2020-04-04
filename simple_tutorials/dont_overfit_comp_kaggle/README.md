### Collection of scripts and notes for a Kaggle competition 
https://www.kaggle.com/c/dont-overfit-ii/overview

**First solution dont_overfit_kaggle_comp.py** 

This is a first solution. The approach follows the steps highlighted below 
1) Multiple simple models are trained, including logistic regression, SVM with linear kernel, decision trees and more (see e.g. https://github.com/nicola-orlando/tensorflow/blob/d5b058fe4ae37385b1d7c6533729c0e0f90d2eaf/simple_tutorials/dont_overfit_comp_kaggle/dont_overfit_kaggle_comp.py#L208). 
2) The training procedure follows simple repeated k-fold cross validation. The hyper parameter setting of each model is optimised in each fold by means of validation datasets. The number of folds is 3. This is not optimised but chosen to ensure enough statistics of each classes in each fold (in the training dataset we have # class 0 >> # class 1). This is not perfect, better methods exist including stratified k-fold cross validation for example.  
3) The final model I choose are logistic regression, SVM with linear kernel, decision trees. This choice is based on observed performance in the training dataset and easy interpretability of the scores for unbalanced training sets (unlike what happens for other simple models such as BayesianRidge).

In total I have then 9 models trained, logistic regression, SVM with linear kernel, decision trees times the three folds. 
The final score for the output dataset is defined as majority vote of the scores: 
* The score is 1 if the sum of scores from the 9 models above is greater or equal to 5
* The score is 0 otherwise 

**Notes on don't overfit challeng** 

* Find this notebook very useful https://www.kaggle.com/rafjaa/dealing-with-very-small-datasets (including features elimination model from sklearn.feature_selection import RFE)
* Dataset augmentation SMOTE (Synthetic Minority Oversampling TEchnique): https://arxiv.org/abs/1106.1813  

