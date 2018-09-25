#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 gnebbia <nebbionegiuseppe@gmail.com>
#
# Distributed under terms of the GPLv3 license.

from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Understand difference between these two:
## cross_val_score is a wrapper and uses KFold or StratifiedKFold
## strategies by default, the latter being used if the estimator derives from ClassifierMixin

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

""" 
The cross_validate function differs from cross_val_score in two ways -
It allows specifying multiple metrics for evaluation.
It returns a dict containing training scores, fit-times and score-times
in addition to the test score.
""" 


from sklearn.model_selection import cross_validate
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


import sklearn.datasets
ds = sklearn.datasets.load_iris()
# Data Preparation
iris = load_iris()
boston = load_boston()
type(iris)
print(iris.data)


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_s_curve
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_biclusters
from sklearn.datasets import make_checkerboard
from sklearn.datasets import make_regression


X = boston.data
y = boston.target



X = iris.data
y = iris.target

## split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=30)

X_train.shape
X_test.shape

y_train.shape
y_test.shape


# Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)


## interpret the model
print(linreg.intercept_)
print(linreg.coef_)

print(list(zip(boston.feature_names, linreg.coef_)))

y_pred = linreg.predict(X_test)

## evaluate
print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


## use cross val KFold method on linear regression

scores = cross_val_score(linreg, X, y, cv=10, scoring='neg_squared_error')
mse_scores = -scores
mse_scores

rmse_scores = np.sqrt(mse_scores)
print(rmse_scores)
print(rmse_scores.mean())


# KNN 

knn = KNeighborsClassifier(n_neighbors=5)


## fit the model with data
knn.fit(X_train, y_train)

## predict the response for new observations
y_pred = knn.predict(X_test)


## measure performance
### Accuracy
print(metrics.accuracy_score(y_test,y_pred))

### KFold Cross Validation
scores = cross_val_score(knn, iris.data, iris.target, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

### we can also obtain predictions with:
predicted = cross_val_predict(knn, iris.data, iris.target, cv=10)
print("Accuracy: %0.2f" %metrics.accuracy_score(iris.target,predicted))



## evaluate with different K, using train/test split
scores = []
k_range = range(1,20)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))


import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Testing Accuracy")


## evaluate with different K, using KFold split
scores = []
k_range = range(1,20)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    scores.append(cv_scores.mean())


import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Testing Accuracy")



# Logistic Regression
logreg = LogisticRegression()

## fit the model with data
logreg.fit(X_train, y_train)

## predict the response for new observations
y_pred = logreg.predict(X_test)

## measure performance
print(metrics.accuracy_score(y_test,y_pred))



## Efficient Parameter Tuning using GridSearchCV

from sklearn.grid_search import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=0)

param_grid = dict(n_neighbors=k_range)
print(param_grid)

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(knn, param_grid, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


print(clf.grid_scores_)
print(clf.best_params_)



## Another good option is related to RandomizedSearchCV



### How to proceed in ML
# split data
data = ...
train, validation, test = split(data)

# tune model hyperparameters
parameters = ...
for params in parameters:
    model = fit(train, params)
    skill = evaluate(model, validation)

# evaluate final model for comparison with other models
model = fit(train)
skill = evaluate(model, test)

"""
Basically the procedure is:

Let's say we want to compare three classifiers, a knn, a logistic regression and the SVM, i can 
train KNN on the training set on k=(2, 10) and then log reg with C=(0.001 to 1) and SVM with C=(0.01 to 10) 
and then i use the validation set to check performance of each of these models...  
at the end i pick the model with parameters which gave the best performance. 
At this point we want to have a score which is independent from
training and validation dataset and see the real performance on a test set.

* The final model could be fit on the aggregate of the training and validation datasets.
* The validation dataset may also play a role in other forms of model preparation, such as feature selection.


"""
"""
NOTE ABOUT KFOLD
A test set should still 
be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach,
called k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally
follow the same principles). The following procedure is followed for each of the k “folds”:
"""



# SANDBOX

X
from sklearn import cluster
from sklearn.decomposition import PCA
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X)
print(k_means.labels_)
len(X)
len(k_means.labels_)
 
pca = PCA(n_components=3).fit(X)

