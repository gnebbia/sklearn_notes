# Machine Learning Model Tuning

## XGBoost  | LightGBM

* min_child_weight or lambda or alpha, good vaues may range from 0 to hundreds,
    so we should not hesitate to try a wide range of values
* eta, num_round
* max_depth (a reasonable default is 7) the higher the value, the more we overfit
* subsample or bagging_fraction, that is, the percentage to use as local validation, 
    the higher the more we solve overfitting, it's a kind of regularization
* colsample_bytree, colsample_bylevel or feature_fraction, that is, a subset of
    columns taken into account for the fitting

## Random Forest and Extra Trees

* number of estimators, the higher the better, start with 10, and see how much
    time it takes to fit data, if it doesn't take too much, then increase the
    number
* max_depth, controls the overfit, start with a value of 7, but also other
    reasonable values are 10, 20
* max_features
* min_samples_leaf

Criterion, gini or entropy, generally gini performs better

## Neural Netwoks

* Start with one layer or two with 32 or 64 units
* Use Adam/Adadelta/Adagrad as optimizers, they converge faster but sometimes
        lead to overfitting wrt SGD + momentum
* Big batch size leads to more overfitting
* Learning rate, generally start with a big one like 0.1 and then slowly
    decrease it

There is a rule of thumb as a connection between batch size and learning rate,
if we increase the batch size by a factor k, we should also increase the
learning rate by the same factor k.

For regularization, once people used L1 and L2 for weights, but nowadays the
most used technique is dropout regularization.
