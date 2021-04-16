# Data Preprocessing

- Before scaling it is a good idea to remove outliers usually.
- In many problems we should also scale targets (outputs)

First, some definitions. "Rescaling" a vector means to add or subtract a
constant and then multiply or divide by a constant, as you would do to
change the units of measurement of the data, for example, to convert a
temperature from Celsius to Fahrenheit. 


## Normalization
"Normalizing" a vector most often means dividing by a norm of the vector,
for example, to make the Euclidean length of the vector equal to one. In the
NN literature, "normalizing" also often refers to rescaling by the minimum
and range of the vector, to make all the elements lie between 0 and 1. 

## Standardization
"Standardizing" a vector most often means subtracting a measure of location
and dividing by a measure of scale. For example, if the vector contains
random values with a Gaussian distribution, you might subtract the mean and
divide by the standard deviation, thereby obtaining a "standard normal"
random variable with mean 0 and standard deviation 1. 

Standardization is composed by two operations:
- Mean Subtraction
- Variance Scaling

In practice we often ignore the shape of the distribution and just transform the
data to center it by removing the mean value of each feature, then scale it by
dividing non-constant features by their standard deviation.

Motivation:
For instance, many elements used in the objective function of a learning
algorithm (such as the RBF kernel of Support Vector Machines or the l1 and l2
regularizers of linear models) assume that all features are centered around zero
and have variance in the same order. If a feature has a variance that is orders
of magnitude larger than others, it might dominate the objective function and
make the estimator unable to learn from other features correctly as expected.
```python
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)
scaler.transform(X_train)    
scaler.transform(X_test)    
```

## Scaling to a Range

An alternative standard standardization is scaling features to lie between
a given minimum and maximum value, often between zero and one, or so that 
the maximum absolute value of each feature is scaled to unit size. This 
can be achieved using MinMaxScaler or MaxAbsScaler, respectively.

The motivation to use this scaling include robustness to very small standard
deviations of features and preserving zero entries in sparse data.

Centering sparse data would destroy the sparseness structure in the data, and
thus rarely is a sensible thing to do. However, it can make sense to scale
sparse inputs, especially if features are on different scales.


### MaxAbsScaler [-1;+1]

MaxAbsScaler works in a very similar fashion, but scales in a way that the
training data lies within the range [-1, 1] by dividing through the largest
maximum value in each feature. It is meant for data that is already centered at
zero or sparse data.


```python
from sklearn import preprocessing

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train = max_abs_scaler.fit_transform(X_train)

# Apply the same operations on test data
X_test = max_abs_scaler.transform(X_test)
```



### MinMaxScaler [0;+1]

This approach of scaling to the range 0,1 is not adviced 
with sparse data.

```python
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

# Apply the same operations on test data
X_test = min_max_scaler.transform(X_test)
```



