# Features Engineering

## Introduction

* Machine learning models take as input features.
* A features is a numeric representation of an aspect of raw data.
* So features sit between model and data in machine learning.

Feature engineering is the act of extracting features from raw data and
transforming them into formats that are suitable for the machine learning model
(definition from the book "Features Engineering" by Alice Zheng.

Features are fundamentally important, the choice of better features can improve
machine learning systems quality.

## ML Pipeline

* Data are observations of real-world phenomena
* A mathematical model of data describes the relationships between different
    aspects of the data
* Mathematical formulas relate numeric quantities to each other. But raw data is
    often not numeric, e.g., pieces of texts, sounds, images and so on, so this
    is where features come in
* Feature Engineering is the process of formulating the most appropriate
    features given the data, the model and the task
* Also important is the number of features, which if it is too big, the model
    will be expensive to train, while if it is too small the model will be
    unable to perform its task

So the place of feature engineering in machine learning is just after raw data 
and it outputs features, and is generally focused on cleaning and transforming.

--> Raw Data --> Feature Engineering --> Features

## Numeric Data

* Do sanity checks, e.g., count data cannot be negative
* Remove outliers, either wiwth percentiles or other techniques
* Always normalize/standardize, this is particularly true for linear models or
    neural network models, there are three reasons for this: first, amount
    of regularization applied to a feature depends on the feature's scale.
    Second, optimization methods can perform differently depending on relative
    scale of features and for distance method e scale of features impacts the
    distance between samples. 
* If a feature has a strange distribution we can consider log transform or
    square root transform or more in general Box-Cox Transforms
* Winsorizing or winsorization is the transformation of statistics by limiting
    extreme values in the statistical data to reduce the effect of possibly
    spurious outliers. It is named after the engineer-turned-biostatistician
    Charles P. Winsor. The effect is the same as clipping in signal
    processing. 
* Interaction features, like sums can be useful for tree-based models


Notice that feature preprocessing can vary between tree based algorithms
and non-tree based algorithms, since tree based algorithms are not affected by
the scales of values.

### Numeric Count Data

With count data we can:
* Keep them as numbers
* Binarize, when we don't need the actual quantity, so we just need a 0 or 1
* Binning (for coarse granularity, this is done if we span different orders 
     of magnitude, this is useful for both supervised and unsupervised learning

#### Binning 
Binning can be done in generally two ways, each with some variations:
* Fixed Width Binning
* Quantile Binning

##### Fixed-width Binning
For what concerns binning there are different strategies, we can bin with a
fixed width and choose the spacing linearly or custom (e.g., ages to represent
phases of life). Or a better idea with data spanning multiple order of
magnitudes is to group by power of 10, so a bin will be 0-9, another 
10-99, then 100-999 and so on.  And to map from the count variable to the 
bin, we take the log10 of the count.
In general each bin contains a specific numeric range. The ranges can be custom
designed or automatically segmented and they can be linearly or exponentially
scaled.
To map from the count to the bin, we just simply divide by the width o the bin
and take the integer part.

##### Quantile Binning
Another approach is based on quantile binning, this is useful if there are large
gaps in the counts, and there could be many empty bins with no data using
fixed-width binning strategy.
So in this case we solve the problem, by adaptively positioning the bins based
on the quantiles of the distribution.





## Feature Selection

To train machine learning algorithms faster, and to reduce the complexity and
overfitting of the model, in addition to improving its accuracy, we can use
feature selection techniques.

In practice Feature selection techniques remove nonuseful features in order 
to reduce the complexity of the model.
Roughly speaking, feature selection techniques fall into three classes:

* Filtering: preprocess features in order to remove the ones which are unlikely
    to be useful. For example we could compute the correlation of the mutual
    information between each feature and the response variable, and then filter
    out the features who fall below a certain threshold, these techniques are
    cheap generally
* Wrapper Methods: these techniques are expensive but they allo us to try out
    subsets of features, so we do not consider single features but sets of
    features taken in combination. The wrapper method treats the model as a
    black box that provides a quality score for a certain subset of features
* Embedded Methods: these methods perform feature selection as part of the model
    training process. For example in decision trees, there is an intrinsic
    feature selection because it selects one feature on which to perform the
    splitting phase at each training step. Another example is the L1 regularizer
    which encourages models that use fewer features. Embedded filtering is
    cheaper but also less powerful than wrapper methods.

### Feature Selection: Filtering Techniques

In filter methods, each feature will be assigned a score, computed by different
statistical measures. In other words, these methods rank features by considering
the relationships between the features and the targets. Filter methods are
usually used in the pre-processing phase:

Examples are:

* Pearson's Correlation Index, scipy.stats.pearsonr(x, y) with scipy
* Linear Discriminant Analysis (LDA) which is used to find a linear combination
    of features which are able to separate classes
* ANOVA is similar to LDA, but operates using categorical features to check
    whether the means of the different classes are equal, it analyzes the
    differences between means
* To compare a categorical variable with a continuous variable we can perform 
  One-way ANOVA test: we calculate in-group variance and intra-group variance 
  and then compare them.
  I think the last two are the same thing
* Chi-Square statistical test, (measures independence between categorical variables by using
    a contingency table) is used to determine if a subset data matches a population,
    Chi-square is used to test if the relationship of a dependent variable is
    significant to an independent variable. Imagine you have a model trying to
    predict if a person will buy or not buy an item. If you have a variable for
    male (yes or no) and the distribution is 90 buy and 10 no buy the
    significance using chi-square would be higher or significant. This is over
    simplifying it but the important concept is that it is using the chi-square
    test to evaluate whether a predictor is significant to a target variable
    based on the expected vs observered value. You often have a threshold to
    select the top x amount based on the statistical result.
* Cramer's V (or Cramer's phi) is an index of correlation between two categorical
    variables based on the chi square statistical test, giving a value between 0
    and +1 (inclusive), Cramér's V varies from 0 (corresponding to no
    association between the variables) to 1 (complete association) and can reach
    1 only when the two variables are equal to each other. 
    In the case of a 2 × 2 contingency table Cramér's V is equal to the Phi
    coefficient. 
* Tschuprow similar to Cramer's V, since it is based on Chi-square
* Mean Square Contingency
* Maung, all of these coefficients can be viewed on the clustering book by Gan,
    Ma and Wu at page 102
* Variance of a feature, this is one of th simplest ways to perform feature
    selection. The trick here is to check how much is the variance of a 
    feature, for example features who have 0 variance are useless,
    we can for example filter out features who have a variance 
     an example, suppose that we have a dataset with boolean features, and we
     want to remove all features that are either one or zero (on or off) in more
     than 80% of the samples. Boolean features are Bernoulli random variables,
     and the variance of such variables is given by
     Var[X] = p(1 - p), in python we can filter that with:
```python
from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)
```

Disadvantages of feature selection is related to the fact that they do not take
into account interactions between features, apparently useless features can be
useful when grouped with others.
Advice: use light filtering as an efficient initial step if running time of our
learning algorithm is an issue.


### Feature Selection: Wrapper Methods

Wrapper Methods can be subdivided into:

* Forward Selection: uses searching as a technique for selecting the best
    features. It is an iterative method. In every iteration, we add more
    features to improve the model, until we no longer have any further
    improvements to make
* Backward Elimination:  is like the previous method but, this time, we start
    with all of the features, and we eliminate some in every iteration until the
    model stops improving
* Recursive Feature Elimination: recursive feature elimination as a greedy
    optimization algorithm. This technique is performed by creating models with
    different subsets and computing the best performing feature, scoring them
    according to an elimination ranking.

```python
from pandas import read_csv
from sklearn.feature_selection import RFEfrom
sklearn.linear_model import LogisticRegression 
load dataurl = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8] 
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
```

### Feature Selection: Embedded Methods

The main goal of feature selection's embedded method is learning which features
are the best in contributing to the accuracy of a specific machine learning model.
They have built-in penalization functions to reduce overfitting:

* Lasso Linear Regression, L1
* Ridge Regression, L2
* Tree Based Feature Selection

## Categorical Variables

A simple question can serve as litmus test for whether something should be a catego‐
rical variable: "Does it matter how different two values are? Or only that they are 
different?" A house price of $20,000 is twice higher than a house of $10,000. 
So, house price should be represented by a continuous numeric variable. 
While the industry of the company (oil, travel, tech, etc.), on the other hand, 
should probably be categorical.
The hour of the day is another tricky example, depending on the context we can represent
it as a numerical variable or as a categorical variable, but generally categorical variable
for this kind of information is a better choice.
Another example is user IDs, nowadays used in many contexts, this is another example of 
categorical variable, and can be very large. Hence, it becomes interesting how to encode 
these categorical information into numbers since machine learning at the end only 
understands numbers, so categorical encoding techniques tackle this problem and try
to give a meaningful representation to the categorical variables.

Note: It is tempting to try to assign a number to each category, but this is bad, since
categories are generally not orderable, while the resulting values would be orderable,
and e.g., if we encode eye colors "brown", "blue" and "black" with 0, 1 and 2, the algorithm
may understand that blue is actually closer to black than brown, which in general
may not make sense.

Let's look at some encoding techniques for categorical variables.

Here’s the list of Category Encoders functions with their descriptions and the
type of data they would be most appropriate to encode.

CLASSIC ENCODERS:
The first group of five classic encoders can be seen on a continuum of embedding
information in one column (Ordinal) up to k columns (OneHot). These are very
useful encodings for machine learning practitioners to understand.

* Ordinal
* One Hot
* Binary
* BaseN
* Hashing

CONTRAST ENCODERS:
The five contrast encoders all have multiple issues that I argue make them
unlikely to be useful for machine learning. They all output one column for each
column value. I would avoid them in most cases. Their stated intents are below.

* Helmert (reverse)
* Sum
* Backward Difference
* Polynomial

BAYESIAN ENCODERS:
The three Bayesian encoders use information from the dependent variable in their
encodings. They all output one column and can work well with high cardinality
data. Anyway it is import to encode exclusively on the training dataset.

* Target
* Leave One Out
* Weight Of Evidence

In detail:

For encoding a categorical variable with few levels we can use:
* Label Encoding, consists in converting classes to increasing integers, this is
    ok for categorical ordinal variables or with tree-based methods who work with 
    categories out of the box but using this encoding may help saving space
* One-Hot Encoding, introduces a linear combination, so the model lose interpretability, 
    since we can have multiple models as a solution, anyway it is a simple solution,
    and also having an additional combination allows us to have an extra category that we
    can use for missing data, this can create a lot of features if we have a lot of categories,
    in these cases a technique which may work is the application of a dimensionality reduction
    technique such as PCA after the encoding, to reduce the number of features
* Dummy Encoding, we solve the problem of linear combinations by dropping one of the 
    categories, so we have a unique model and it is interpretable, but lose the extra
    category that could be used for missing or unspecified data, dummy encoding
    also saves us from the linear dependency introduced with label encoding
* Effect Encoding (also sum or deviation encoding), similar to dummy encoding but encodes the reference category with a
    combinations filled with -1, this still gives us the model interpretability and
    uniqueness but with respect to dummy encoding produces a more dense vector
    which may be a disadvantage in terms of storage and computation time. There
    is a slight difference between effect/sum encoding and deviation encoding,
    which changes the meaning of the intercept in a regression context


For encoding a categorical variable with a medium number of levels we can use:
*  Binary Encoding, convert each integer to binary digits. Each binary digit gets one column.
    Some info loss but fewer dimensions. First the categories are encoded as
    ordinal, then those integers are converted into binary code, then the digits
    from that binary string are split into separate columns.  This encodes the
    data in fewer dimensions that one-hot, but with some distortion of the
    distances.
* Frequency Encoding, maps categories to their frequencies, this can have the
    downsides when we have categories with equal frequencies
* Mean Encoding, we encode a category with its mean with respect the target, so
    let's say that "category_1" is active 5 times when target variable is 1 and
    and 6 times when target variable is 0, then its mean encoded value will be
    5/11, of course we should encode only considering the training dataset,
    there are also regularization formulas to use when classes are unbalanced.


Generally in these cases the most common solutions adopted are dummy encoding 
and one-hot encoding.

For encoding a categorical variable which has many values we have different
solutions:
* Still use one of the previous techniques, and know that it will take a lot of
    space and the computations will be slower
* Compress categories:
    * Feature Hashing
    * Bin Counting


TODO:
* Helmert Encoding, The mean of the dependent variable for a level is
    compared to the mean of the dependent variable over all previous levels.
    Hence, the name ‘reverse’ being sometimes applied to differentiate from
    forward Helmert coding.
* Backward Difference Encoding, Backward Difference: the mean of the dependent
    variable for a level is compared with the mean of the dependent variable for
    the prior level. This type of coding may be useful for a nominal or an
    ordinal variable.
* Polynomial Encoding, The coefficients taken on by polynomial coding for k=4
    levels are the linear, quadratic, and cubic trends in the categorical
    variable. The categorical variable here is assumed to be represented by an
    underlying, equally spaced numeric variable. Therefore, this type of
    encoding is used only for ordered categorical variables with equal spacing.





ADVICES:
For nominal columns try OneHot, Hashing, LeaveOneOut, and Target encoding. Avoid
OneHot for high cardinality columns and decision tree-based algorithms.

For ordinal columns try Ordinal (Integer), Binary, OneHot, LeaveOneOut, and
Target. Helmert, Sum, BackwardDifference and Polynomial are less likely to be
helpful, but if you have time or theoretic reason you might want to try them.

For regression tasks, Target and LeaveOneOut probably won’t work well.
Of course the best way to encode categorical variables is to use models which do
not need any encoding and can deal naturally with categorical variables such as
tree based models.

For tree based models, label encoding and frequency encoding are helpful.
For non tree based models, one hot encoding is used frequently.


### Categorical Features Tricks

* We can generate new categorical features based on existing categorical
    features using combinations, so if feature cat1 has 2 values and feature
    cat2 has 3 values, we will have a third feature which will mix all the
    combinations, notice that the higher the possible values the more disk we
    are going to occupy, and we add complexity to the model, these are called
    **interaction features** and generally help non-tree based models
* We can add the frequency encoding for a certain categorical variable, this can
    help non-tree based models



### Feature Hashing


```python
from sklearn.feature_extraction import FeatureHasher
# number of classes we have to tune
m = 2000
h = FeatureHasher(n_features=m, input_type='string')
# we apply feature hashing to a feature representing the user_id
f = h.transform(ds['user_id'])
```


### Bin Counting

## Adding Other Features with Model Stacking

A trick can be, the one of performing clustering on data (we can also both use X and y),
and use the cluster as an additional feature of the dataset.

Using k-means to create new features for a classifier is an example of 
"model stacking", where the input to one model is the output of another model.

Another example of stacking could be for example the use of the output of a
decision tree-type model(gradient boosting tree or random forest) as input
to a classifier.

Key Intuition for Model Stacking
Use sophisticated base layers (often with expensive models) to gen‐
erate good (often nonlinear) features, combined with a simple and
fast top-layer model. This often strikes the right balance between
model accuracy and speed.


TODO: 
* Copy feature hashing code DONE
* Copy model stacking with k-means code
* Box Cox Transformation?
* Copy and UNDERSTAND BIN COUNTING


## Datetime and Coordinates Features

For datetime features we can considerate:
* Periodicity, like day of the week, or month or week identifier and so on
* Time since a specific event
* A difference between dates

For coordinates we can consider:
* Interesting places taken as reference, like "close to main church" and so on
* Centers of clusters of locations
* Aggregated statistics on locations, like statistical characteristics of neighborhoods


## Cleaning Data

* Check if data is shuffled, through visualization with mean and rolling mean,
    notice that the rolling mean should be around the mean
* Remove duplicated rows
* Remove duplicated rows with different labels
* Remove duplicated columns
* Remove columns with parallel classes, e.g., one is using A,B,C and the other
    one is using D,G,H respectively, and they are always the same

```python
for f in categorical_feats:
    traintest[f] = traintest[f].factorize()

traintest.T.drop_duplicates()
```

### Handling Missing Values in Machine Learning

We have to deal with missing values in our data in some way.
Missing values can come in different forms such as:

* NaN (Not a Number) or NA (Not Assigned)
* Empty Strings
* Outliers

We should review each feature/column to search for these kind of values.
One way to do this, it to search for evident outliers, like huge numbers or
negative numbers in a feature which cannot be negative or again a search for
empty strings.

Another way anyway is to visualize the distributions of features.
Visualizing distributions is particularly useful, since missing values can be
hidden from us, for example, somebody could have substituted these values with
the mean or median value for that feature, and this things can come out
sometimes by plotting the distribution.



### Techniques to Fill Not Assigned Values

The strategies used to assign missing values are:
* Use an outlier, like -1, -99, etc..., this option, gives the possibility to
    take missing value into a separate category, this is good for tree-based
    methods but not good for linear and neural network based models
* Use mean, median, this is good for linear and neural network based models, but
    is not beneficial for tree models
* Recontruct the value somehow, the benefits of this depend on the strategy used
    to reconstruct the value, we can use machine learning models to reconstruct
    these values, this can be tricky but useful

We can exploit missing values to create features, for example for each column
having missing values we can create an additional column "isnull" which says 1
if the value of the corresponding feature is missing.

The disadvantage of this is that, if we do this for each column we double the
number of features.

* Also remember to avoid filling NaNs before feature generation.
* Sometimes we should also consider to remove rows with missing values

So again, if we have a distribution which looks strange with a peak, we could
infer that a certain value was missing and was filled with mean/median, at this
point we can substitute this value with Nan, or -999 or remove these values or
add a column/feature which states that the value was missing.


## Features from Text

With text a preprocessing useful is many times useful and consists in:
* lowercasing words,
* stemming/lemmatization 
* stopwords removal

Once this is done we can extract features from text following two approaches:
* Bag of words, use tf-idf as post-processing, which is a way to normalize data
    in bag of words In this case it can be also useful to  also n-grams
    are helpful to take advantage of local context, also remember to appy apply tf-idf
    only at the end after having chosen the number of n-grams
* Word2Vec, also constructs vectors for each vector but in order to have
    context, it evaluates distance between word vectors, there are different
    implementations for this kind of algorithm, namely:
    * Word Level = Word2Vec, Glove, FastText, etc...
    * Sentence Level = Doc2Vec, etc...
    * There are Word2Vec pretrained models, based on different sources, e.g., wikipedia

A comparison between these two approaches are:
* Bag of Words:
    * Very Large Vectors
    * Meaning of each value in vector is known
* Word2Vec:
    * Relatively Small Vectors
    * Values in vector can be interpreted only in some cases
    * The words with similar meaning often have similar embeddings

These two approaches can give different results, but can also be used together.

A note on tf-idf:
* TF: normalizes sum of the row values to 1
* IDF: scales features inversely proportionally to a number of word occurrences
    over documents

## Feature from Images

* Use values of conv layers


## EDA and its impact on features


Always start with EDA, to build new features,
for example sometimes it is useful to understand whether there is a difference
between two features, and create a feature which is binary, 1 if there is a
difference and 0 if there is no difference.


## Visualize and Understand Features


### Explore individual features

* Histograms, look at different scales
* Describe features with statistics


#### Plot of index vs value

```python
# this allows us to visualize feature x with respect to label y
plt.scatter(range(len(x)), x, c=y)
```

### Explore Feature Relations



#### Explore Pairs of Features Relations
* Scatter Plot, Scatter Matrix
* Corrplot

#### Explore Groups of Features Relations
* Corrplot + Clustering
* Plot (index vs feature statistics)
* Plot Mean Value for each feature, and order mean values, to see how features
    can be grouped in terms of magnitudes of their values



## Mean Encoding or Likelihood Encoding



## Model Ensembling

TODO: Difference between two features does it make sense? and in decision trees?
TODO2: Deepen metrics
Bray Curtis Metric



## Advanced Feature Engineering


Adding a max and a min for a specific feature, for example let's say we have an
article id feature to which it corresponds a price, we can insert two new
features which will have the maximum price and the minimum price.

We can add standard deviation of a specific feature, or other features we
can understand from data.



We can create new features from decision trees to extract higher order
interactions.


## Ensemble Learning

Ensemble learning is a technique which consists in combining different machine
learning models to get a better prediction.

Common Ensemble methods are:

* Averaging (or blending)
* Weighted Averaging
* Conditional Averaging
* Bagging
* Boosting
* Stacking
* Stacknet


Here is a short description of Bagging, Boosting and Stacking three methods:

* Bagging (stands for Bootstrap Aggregating) is a way to decrease the variance
  of your prediction by generating additional data for training from your
  original dataset using combinations with repetitions to produce multisets of
  the same cardinality/size as your original data. By increasing the size of
  your training set you can't improve the model predictive force, but just
  decrease the variance, narrowly tuning the prediction to expected outcome.

* Boosting is a two-step approach, where one first uses subsets of the
  original data to produce a series of averagely performing models and
  then "boosts" their performance by combining them together using a
  particular cost function (=majority vote). Unlike bagging, in the
  classical boosting the subset creation is not random and depends upon
  the performance of the previous models: every new subsets contains the
  elements that were (likely to be) misclassified by previous models.

* Stacking is a similar to boosting: you also apply several models to
  your original data. The difference here is, however, that you don't
  have just an empirical formula for your weight function, rather you
  introduce a meta-level and use another model/approach to estimate
  the input together with outputs of every model to estimate the
  weights or, in other words, to determine what models perform well
  and what badly given these input data.



### Averaging

We just take the average in the prediction probabilities given by two or more
models.

So sum of model predictions divided by the number of models we are going to use.

### Weighted Averaging

In this case we give more importance to a specific model, so the final
prediction will be the weighted sum of model predictions divided by the number
of models used.

### Conditional Averaging

In the case of weighted averaging we may not know to which classifier to give
more weight, but for example we may spot regions in which certain classifiers
are better than others.

Let's say for example in a regression problem for house prices we have a model
which predicts very well houses with a surface smaller than 200 sq. meters, and
another model which predicts very well houses with a surface larger or equal
than 200 sq. meters, at this point we could use predictor 1 in case the house is
smaller than 200 sq. meters and predictor 2 in the other case.


### Bagging

The technique of bagging is related to averaging slightly different versions of
the same model to improve the predictive power.

Bagging is usued in Random Forest, where more decision trees are used for
predicitons.

Bagging can help use for two kinds of errors in modelling:
1. Errors due to Bias (underfitting)
2. Errors due to Variance (overfitting)

Bagging helps in avoiding overfitting, but how do we implement bagging?
We should focus on techniques which slightly change our models, like:
* Changing the Seed
* Row (Sub) sampling or Bootstrapping
* Shuffling
* Column (Sub) sampling
* Model-specific parameters (e.g., in a logistic regression, we could take 10
    slightly different regularization parameters)
* Control the number of models we are using, so how many logistic regressions
    are we using ? In this context the models are called **bags** so we control
    the number of bags, in general the more bags we use the better are the
    results
* Optionally (parallelism), we can parallelize bags



### Boosting

The technique of boosting consists in a form of weighted averaging of models
where each model is built sequentially via taking into account the past model
performance.

There are two main boosting algorithms:
* Weight Based
* Residual Based

#### Weight Based Boosting

In this case, we are going to track the predictions of our model, now this
probability predictions will have a certain margin of error, which will be:
|y - prediction|
so we consider the error as an absolute value, and then we create a new column
called "weight" where we say that this can just be the absolute error + 1.
There are different ways to compute this weight.

So what we are going to do next is to include this weight into our new model as
a feature and fit with this feature, this procedure can be repeated iteratively
until we reach the desired performances.
This is a technique used to maximize the focus from where the previous models
have done more wrong.

Parameters of this model are:
* Learning rate (or shrinkage or eta)
* Number of estimators

Notice that there are different variations to boosting, for example certain
variations, just take into account if the previous model correctly classified a
certain data point or not, so not everyone takes into account the margin of
error.

#### Residual Based Boosting

This is one of the most dominant techniques used in competitive data science
in the context of structured datasets.

This is very similar to the previous technique, the only difference is that we
use the error of predictions from the targets without considering the absolute
error.

Parameters related to residual boosting are:
* Learning rate (or shrinkage or eta)
* Number of estimators
* Row (sub) sampling
* Column (sub) sampling
* Input model - better be trees
* Sub boosting type:
    * Fully gradient based
    * Dart (very useful in classification problems)

Famous implementations of Residual Based Boosting are:
* Xgboost
* Lightgbm
* H20's GBM
* Catboost
* Sklearn's GBM


### Stacking

The technique of stacking consists in making predictions of a number of models
in a hold-out set and then using a different (meta) model to train on these
predictions.


In stacking we collect different predictions from different models and give them
to another model.

We can imagine as example, taking predictions from a logistic regression, an 
SVM and a random forest and feed them as features to a deep neural network.


This methodology was introduced by Wolpert in 1992, and it involves:
1. Training seveeral base learning algorithms (learners) on the training set
2. Make predictions with the base learners on the validation part
3. We use the predictions collected from the learners as the input in a new
   dataset which will represent the training set of a higher level learner

In this context the base learning algorithms are called "base learners" while
the higher level algorithm is called "meta learner".

Let's try an example in python with two base learners:
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split

training, valid, ytraining, yvalid = train_test_split(train, y, test_size=0.5)

model1 = RandomForestRegressor()
model2 = LinearRegression()

model1.fit(training,ytraining)
model2.fit(training,ytraining)

preds1 = model1.predict(valid)
preds2 = model2.predict(valid)

test_preds1 = model1.predict(test)
test_preds2 = model2.predict(test)

stacked_predictions = np.column_stack( (preds1, preds2) )
stacked_test_predictions = np.column_stack( (test_preds1, test_preds2) )

meta_model = LinearRegression()

meta_model.fit(stacked_predictions, y_valid)
final_predictions = meta_model.predict(stacked_test_predictions)
```

* Consider that there is a performance plateuing after a certain number of models
stacked.
* It is important to use diversity, such as:
    * Use of different algorithms
    * Different input features, such as categorical features encoded in
        different ways
* Generally meta learners are modest and not deep/very complex models, they are
    generally simple



### StackNet

This is a generalization of neural networks in a certain sense, where instead of being
limited to linear models using in neural networks.

With stacknets we can use any models, and stack them if they were a neural
network, anyway we do not use backpropagation since we cannot compute
differentiation of some of the models, but in this case we use model stacking
(seen previously).



## Tips for Ensemble Methods


A list of tips we can use with ensemble methods:

* Take advantage of Diversity Based on algorithms:
    * 2-3 gradient boosted trees (lightgb, xgboost, H2O, catboost)
    * 2-3 Neural Nets (keras, pytorch)
    * 1-2 ExtraTrees/Random Forest (sklearn)
    * 1-2 linear models as logistic regression/ridge regression, linear svm
        (sklearn)
    * 1-2 knn models (sklearn)
    * 1 Factorization machine (libfm)
    * 1 SVM with nonlinear kernel if size/memory allows (sklearn)
* Diversity based on input data:
    * Categorical features, one hot, label encoding, target encoding
    * Numerical features, outliers, binning, derivatives, percentiles 
    * Interactions: col1 operation col2, or groupby statements with categories
        of a categorical feature, or unsupervised algorithm to get new features

For meta learners, or higher layers in a stacknet, we should:
* Use simpler (or shallower) algorithms:
    * Gradient Boosted Trees with small depth (like 2 or 3)
    * Linear models with high resolution
    * Extra Trees
    * Shallow networks (as in 1 hidden layer)
    * knn with BrayCurtis distance
    * Brute forcing a search for best linear weights based on cross validation
* Feature engineering:
    * pairwise differences between meta features
    * row-wise statistics like average or standard deviations
    * standard feature selection techniques

As a rule of thumb, for every 7 models in the previous level we should add 1
meta model.
