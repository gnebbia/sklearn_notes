## Decision Trees

* Decision trees tend to overfit on data with a large number of features. Getting the right ratio of samples to number of features is important, 
    since a tree with few samples in high dimensional space is very likely to overfit.
* Consider performing dimensionality reduction (PCA, ICA, or Feature selection) beforehand to give your tree 
    a better chance of finding features that are discriminative.
* Visualise your tree as you are training by using the export function. Use `max_depth=3` as an initial tree 
    depth to get a feel for how the tree is fitting to your data, and then increase the depth.
* Remember that the number of samples required to populate the tree doubles for each additional level the 
    tree grows to. Use max_depth to control the size of the tree to prevent overfitting.
* Use `min_samples_split` or `min_samples_leaf` to control the number of samples at a leaf node. A very small number will usually 
    mean the tree will overfit, whereas a large number will prevent the tree from learning the data. 
    Try `min_samples_leaf=5` as an initial value. If the sample size varies greatly, a float number can be used as percentage in 
    these two parameters. The main difference between the two is that min_samples_leaf guarantees a minimum number of samples in a leaf, 
    while min_samples_split can create arbitrary small leaves, though `min_samples_split` is more common in the literature.
* Balance your dataset before training to prevent the tree from being biased toward the classes that are dominant. Class balancing can be 
    done by sampling an equal number of samples from each class, or preferably by normalizing the sum of 
    the sample weights (sample_weight) for each class to the same value. Also note that weight-based 
    pre-pruning criteria, such as min_weight_fraction_leaf, will then be less biased toward dominant classes 
    than criteria that are not aware of the sample weights, like min_samples_leaf.
* If the samples are weighted, it will be easier to optimize the tree structure using weight-based pre-pruning criterion such 
    as min_weight_fraction_leaf, which ensure that leaf nodes contain at least a fraction of the overall sum of the sample weights.
* If the input matrix X is very sparse, it is recommended to convert to sparse csc_matrix before calling fit and sparse 
    csr_matrix before calling predict. Training time can be orders of magnitude faster for a sparse matrix input compared 
    to a dense matrix when features have zero values in most of the samples.

## SVM

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

* If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.


### Pratical Tips


* Avoiding data copy: For SVC, SVR, NuSVC and NuSVR, if the data passed to certain methods is not C-ordered contiguous, and double precision, 
    it will be copied before calling the underlying C implementation. You can check whether a given numpy array is C-contiguous 
    by inspecting its flags attribute.
* Kernel cache size: For SVC, SVR, nuSVC and NuSVR, the size of the kernel cache has a strong impact on run times for larger problems. 
    If you have enough RAM available, it is recommended to set cache_size to a higher value than the default of 200(MB), such as 500(MB) or 1000(MB).
* Setting C: C is 1 by default and it’s a reasonable default choice. If you have a lot of noisy observations you should decrease it. 
    It corresponds to regularize more the estimation.
* Support Vector Machine algorithms are not scale invariant, so it is highly recommended to scale your data. For example, 
    scale each attribute on the input vector X to [0,1] or [-1,+1], or standardize it to have mean 0 and variance 1. 
    Note that the same scaling must be applied to the test vector to obtain meaningful results. 
    See section Preprocessing data for more details on scaling and normalization.
* Parameter nu in NuSVC/OneClassSVM/NuSVR approximates the fraction of training errors and support vectors.
* In SVC, if data for classification are unbalanced (e.g. many positive and few negative), 
    set class_weight='balanced' and/or try different penalty parameters C.
* The underlying LinearSVC implementation uses a random number generator to select features when fitting the model. 
    It is thus not uncommon, to have slightly different results for the same input data. If that happens, try with a smaller tol parameter.
* Using L1 penalization as provided by LinearSVC(loss='l2', penalty='l1', dual=False) yields a sparse solution, i.e. only a 
    subset of feature weights is different from zero and contribute to the decision function. Increasing C yields a more complex 
    model (more feature are selected). The C value that yields a “null” model (all weights equal to zero) can be calculated using l1_min_c.



## Linear Discriminant Analysis and Quadratic Discriminant Analysis

QDA is a good classification algorithm, with quadratic decision surfaces, while
LDA while also being a classification algorithm is not that good as QDA since it
can only compute linear decision boundaries.

LDA is also used as dimensionality reduction in a supervised setting, but may 
make sense only when the we have a lot of output classes, so in high multinomial
problems, since the dimensionality reduction applied will have as output 
a number of features smaller than the number of output outcomes which are
possible.

