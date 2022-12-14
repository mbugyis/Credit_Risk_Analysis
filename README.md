# Credit_Risk_Analysis

Supervised Machine Learning on Credit dataset

## Overview
When determining wheter to approve a loan or not, a lot of factors are taken into place, and some can be overlooked. The purpose of this analysis is to use a machine learning model to evaluate the factors in order to make a proper judgement on approval. Several steps in the machine learning process were utilized to ensure accurate approval of the model.


## Results

### Naive Random Oversampling
![naive](https://github.com/mbugyis/Credit_Risk_Analysis/blob/main/images/naive_sampling.png)
1. Accuracy = 0.6249984891886339
2. Precision = low for high risk, high for low risk
3. Recall = high/low --> .60/.65

### SMOTE Oversampling
![smote](https://github.com/mbugyis/Credit_Risk_Analysis/blob/main/images/SMOTE_sampling.png)
1. Accuracy = 0.6512584051472337
2. Precision = low for high risk, high for low risk
3. Recall = high/low --> .64/.66

### Undersampling
![undersampling](https://github.com/mbugyis/Credit_Risk_Analysis/blob/main/images/undersampling.png)
1. Accuracy = 0.6512584051472337
2. Precision = low for high risk, high for low risk
3. Recall = high/low --> .59/.43

### Over-Under Sampling
![O/U Sampling](https://github.com/mbugyis/Credit_Risk_Analysis/blob/main/images/overunder_sampling.png)
1. Accuracy = 0.5103893461611291
2. Precision = low for high risk, high for low risk
3. Recall = high/low --> .70/.57

### Balanced RandomForest Classifier
![Balanced RF](https://github.com/mbugyis/Credit_Risk_Analysis/blob/main/images/balRF_sampling.png)
1. Accuracy = 0.7877672625306695
2. Precision = low for high risk, high for low risk
3. Recall = high/low --> .67/.91

### Easy Ensemble AdaBoost Classifier
![Easy AdaBoost](https://github.com/mbugyis/Credit_Risk_Analysis/blob/main/images/easyAda_sampling.png)
1. Accuracy = 0.925427358175101
2. Precision = low for high risk, high for low risk
3. Recall = high/low --> .91/.94

## Summary
In the end, the model that seemed to be the best, given the numbers of accuracy, recall, f1, etc., was the Easy Ensemble AdaBoost Classifier with a accuracy of about 93%. All others didn't come close to this number, nearly 15% behind this.

I think it would be worth an effort to try different approaches in spliting the data and trying a differnt n_estimators or random_state number to see if this has an impact on the accuracy of the model
