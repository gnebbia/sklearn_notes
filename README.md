# Scikit Learn


## Datasets

Generally we can load a dataset using pandas, since dataframes are very handy
and easy to work with.

Anyway scikit-learn provides some example datasets or the ability to 
create some toy dataset to work with.

### Loading a Toy Dataset

Scikit Learn has a set of toy dataset which can be used for quick tests
or to experiment with scikit capabilities.
```python
import sklearn.datasets 

ds = sklearn.datasets.load_iris()              # classification
ds = sklearn.datasets.load_digits()            # classification
ds = sklearn.datasets.load_wine()              # classification
ds = sklearn.datasets.load_breast_cancer()     # classification
ds = sklearn.datasets.load_boston()            # regression
ds = sklearn.datasets.load_diabetes()          # regression
ds = sklearn.datasets.load_linnerud()          # multivariate regression
```

### Creating a Toy Dataset


```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_s_curve
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_regression

X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')

X, y = make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')

X, y = make_gaussian_quantiles(n_features=2, n_classes=3)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')

X, y = make_circles(n_samples=100, shuffle=True, noise=None, random_state=None, factor=0.8)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')

X, y = make_moons(n_samples=100, shuffle=True, noise=0.2, random_state=None)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')

X, y = make_s_curve(n_samples=10000, noise=0.0, random_state=42)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')

X, y = make_swiss_roll(n_samples=10000, noise=0.0, random_state=None)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')

X, y = make_regression(n_samples=100, n_features=10, n_informative=10, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)
plt.scatter(X[:, 0], X[:, 1], marker='o',  s=25, edgecolor='k')
```


## Normalizing Data



## General Machine Learning Workflow

The general machine learning workflow is reported below:

```python
# load data
data = load_data()

# split data in train/validation/test
train, validation, test = split(data)

# tune model hyperparameters
parameters = {C: [0.001, 0.01, 0.1, 1], lambda: [0.1, 0.3], ...}
for params in parameters:
    model = fit(train, params)
    skill_on_validation = evaluate(model, validation)

# pick the best model
final_model = model_with_max_skill_on_validation

# evaluate final model for comparison with other models
final_model = fit(train)
skill = evaluate(model, test)
```

An alternative workflow using a different strategy for validation called KFold
is reported below:

```python
# load data
data = load_data()

# split data only in train/test
train, test = split(data)

# tune model hyperparameters using cross validation, number of folds is generally 10
parameters = {C: [0.001, 0.01, 0.1, 1], lambda: [0.1, 0.3], ...}
for params in parameters:
    model = cv_fit(train, params)
    skill_on_validation = evaluate_kfold(model, cv=10)

# pick the best model
final_model = model_with_max_skill_on_validation

# evaluate final model for comparison with other models
final_model = fit(train)
skill = evaluate(model, test)
```

We generally prefer KFold when computation time is not a problem or dataset is
small, so this is not the preferred choice in common deep learning problems, but
if we can apply it, it is surely a good choice.

Notice that it is a good idea once we have selected our model to re-train using
the entire training + validation datasets.


## Models

Now we will show some of the common models used, this can be useful 
in order to copy paste, in some scripts, anyway the train/validation/test
strategy is applied to validate these models.
In order to see how to validate with KFold check later sections.

Remember that once the model is initialized, we can print at any time, its
parameters and its status by doing:
```python
print(model)
```

For example let's say we used a logistic regression, and we want to print its
parameters, we can do:
```python
logreg = LogisticRegression()

print(logreg)
```


### Linear Regression

```python
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ds = sklearn.datasets.load_boston()
X = boston.data
y = boston.target


# We use train/test/dev in this context for simplicity the split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=30)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state=30)


linreg = LinearRegression()


linreg.fit(X_train, y_train)

# interpret the model
print(linreg.intercept_)
print(linreg.coef_)


print(list(zip(boston.feature_names, linreg.coef_)))

y_pred = linreg.predict(X_validation)

# evaluate
print(metrics.mean_absolute_error(y_validation, y_pred))
print(metrics.mean_squared_error(y_validation, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_validation, y_pred)))

# We should now pick the best model and then train on the training/validation set
# and test on the testing set
```

### Support Vector Machines for Regression

TODO

### Random Forest Regressor

TODO

### KNN 

```python
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

ds = sklearn.datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=30)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state=30)

knn = KNeighborsClassifier(n_neighbors=5)


# fit the model with data
knn.fit(X_train, y_train)

# predict the response for new observations
y_pred = knn.predict(X_validation)

# print probabilities
y_pred_probs = knn.predict_proba(X_validation)


# measure performance: Accuracy
print(metrics.accuracy_score(y_validation,y_pred))

# KFold Cross Validation
scores = cross_val_score(knn, X_train + X_validation, y_train + y_validation, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

## we can also obtain predictions with:
predicted = cross_val_predict(knn, X_train + X_validation, y_train + y_validation, cv=10)
print("Accuracy: %0.2f" % metrics.accuracy_score(iris.target,predicted))



# evaluate with different K, using train/test split
scores = []
k_range = range(1,20)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_validation)
    scores.append(metrics.accuracy_score(y_validation, y_pred))


import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Testing Accuracy")


# evaluate with different K, using KFold split
scores = []
k_range = range(1,20)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train+X_validation, y_train+y_validation, cv=10, scoring='accuracy')
    scores.append(cv_scores.mean())


import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Testing Accuracy")
```

### Logistic Regression


```python

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
```


### SVM
```python
from sklearn.svm import SVC
## SVM
clf = SVC()
# or
clf = SVC(kernel="linear", C=0.025),
clf = SVC(gamma=2, C=1),
clf.fit(X, y)
y_pred = clf.predict(X_test)
```

To get fitted parameters of the model we can do:
```python
# get support vector
clf.support_vectors_
# get indices of support vectors
clf.support_ 
# get number of support vectors for each class
clf.n_support_ 
```

### Decision Trees

```python
### Decision Tree

from sklearn import tree
dc = tree.DecisionTreeClassifier(max_depth=5)
dc = dc.fit(X, Y)
y_pred = dc.predict(X_test)
```


#### Plot Decision Trees

```python
import graphviz 
dot_data = tree.export_graphviz(dc, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 

## or alternatively

dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
```

#### Decision Trees as Regressor

```python

dc = tree.DecisionTreeRegressor()
dc = dc.fit(X, y)
y_pred = dc.predict(X_test)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=5, n_estimators=10, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

We can inspect feature importance:
```python
print(rf.feature_importances_)
# Or plotting them
plt.plot(rf.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns.tolist(), rotation=90);

```


### Gaussian Process Classifier

```python
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

gpc = GaussianProcessClassifier(1.0 * RBF(1.0))
gpc.fit(X_train, y_train)
y_pred = gpc.predict(X_test)

```

### Adaptive Boost Classifier

```python
from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier()
abc.fit(X_train, y_train)
y_pred = abc.predict(X_test)
```

### Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

gb = GaussianNB()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

```

### Quadratic Discriminant Analysis

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred = qda.predict(X_test)
```

### XGBoost

This is a very useful algorithm, and is generally used to win data science
competitions, it has very good performances, we have to install it as a third
party module by doing:
```sh
pip install xgboost
```

Then we can use it as any other scikit-learn algorithm, hence:
```python
from xgboost import XGBClassifier

xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
y_pred = xgbc.predict(X_test)
```


## Model Tuning and Validation Techniques

Here we apply a grid search and cross validation on a KNN classifier.

```python
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

param_grid = dict(n_neighbors=k_range)
print(param_grid)

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(knn, param_grid, cv=10, scoring='%s_macro' % score)
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
```


## Clustering 

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris['feature_names'])
#print(X)
data = X[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']]

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    data["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
```



## Useful References
https://www.datascienceatthecommandline.com
