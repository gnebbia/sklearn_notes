# Multiclass and Multilabel Algorithms

Multiclass classification means a classification task with more than two
classes; e.g., classify a set of images of fruits which may be oranges, apples,
or pears. Multiclass classification makes the assumption that each sample is
assigned to one and only one label: a fruit can be either an apple or a pear but
not both at the same time.

Multilabel classification assigns to each sample a set of target labels. This
can be thought as predicting properties of a data-point that are not mutually
exclusive, such as topics that are relevant for a document. A text might be
about any of religion, politics, finance or education at the same time or none
of these.

Multioutput regression assigns each sample a set of target values. This can be
thought of as predicting several properties for each data-point, such as wind
direction and magnitude at a certain location.

Multioutput-multiclass classification and multi-task classification means that a
single estimator has to handle several joint classification tasks. This is both
a generalization of the multi-label classification task, which only considers
binary classification, as well as a generalization of the multi-class
classification task. The output format is a 2d numpy array or sparse matrix.

## Multi Class Classifiers


Although all classifiers support multiclass classification in most frameworks
it is possible to experiment with something sklearn provides us, which is:

* OneVsOneClassifier
* OneVsRestClassifier

These allow us to change how a specific classifier is working.
There are anyway classifiers which are inherently multiclass so these kind of
tuning/experimenting would not make sense.


### One vs Rest Classification

This strategy, also known as one-vs-all, is implemented in OneVsRestClassifier.
The strategy consists in fitting one classifier per class. For each classifier,
the class is fitted against all the other classes. In addition to its
computational efficiency (only n_classes classifiers are needed), one advantage
of this approach is its interpretability. Since each class is represented by one
and only one classifier, it is possible to gain knowledge about the class by
inspecting its corresponding classifier. This is the most commonly used strategy
and is a fair default choice.

We can perform a one vs rest classification with something like this:
```python
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

iris = datasets.load_iris()

X, y = iris.data, iris.target

clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)
y_preds = predict(X) 
```

### One vs One Classification


OneVsOneClassifier constructs one classifier per pair of classes. At prediction
time, the class which received the most votes is selected. In the event of a tie
(among two classes with an equal number of votes), it selects the class with the
highest aggregate classification confidence by summing over the pair-wise
classification confidence levels computed by the underlying binary classifiers.

Since it requires to fit n_classes * (n_classes - 1) / 2 classifiers, this
method is usually slower than one-vs-the-rest, due to its O(n_classes^2)
complexity. However, this method may be advantageous for algorithms such as
kernel algorithms which don’t scale well with n_samples. This is because each
individual learning problem only involves a small subset of the data whereas,
with one-vs-the-rest, the complete dataset is used n_classes times.

```python
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

iris = datasets.load_iris()

X, y = iris.data, iris.target

clf = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y)
y_preds = predict(X) 

```





