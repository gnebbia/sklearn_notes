# Unsupervised Learning


## Dimensionality Reduction

### PCA

PCA can be useful in scenarios where we have high dimensional datasets
and want to reduce the size of the dataset by performing feature selection.
Indeed feature selection will help us learning faster and better in some cases.
This signal preserving/noise filtering property makes PCA a very useful feature
selection routine for example, rather than training a classifier on very
high-dimensional data,  you  might instead train the classifier on the lower
dimensional representation, which will automatically serve to filter 
out random noise in the inputs.

Lastly, it is best not to apply PCA to raw counts (word counts, music play counts,
movie viewing counts, etc.). 
The reason for this is that such counts often contain large outliers. 
(The probability is pretty high that there is a fan out there who watched
The Star Wards 201,582 times, which is can be considered an outlier with respect the 
rest of the counts.) since PCA looks for linear correlations within the features. 
Correlation and variance statistics are very sensitive to large outliers; a 
single large number could change the statistics a lot. So, it is a good 
idea to first trim the data of large values or apply a scaling transform like tf-idf 
or the log transform.

When seen as a method for eliminating linear correlation, PCA is related to the con‐
cept of whitening. Its cousin, ZCA, whitens the data in an interpretable way, but does
not reduce dimensionality.


```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ds = pd.read_csv("file.csv")

feature_names = list(ds)
scaler = StandardScaler().fit(ds)
X = pd.DataFrame(scaler.transform(ds), columns=feature_names) 


pca = PCA(n_components=2)
# Fit the model with X
pca.fit(X)

print(pca.explained_variance_)
print(pca.components_)

# Apply dimensionality reduction to X
X_reduced = pca.transform(X)
component_names = ["PC_" + str(i) for i in range(1, num_components+1)]
ds_pca = pd.DataFrame(X_reduced, columns=component_names)


# We can uncompress, this is for example used in noise reduction
# techniques
X_uncompressed = pca.inverse_transform(X_reduced)


## Plot the cumulative explained variance as function of number of components
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
```

We can also innstantiate PCA by specifying the amount of variance we want to be
explained instead of the number of components, let's see an example:

```python
# Instantiate PCA with 50% of variance explained, this
# is also useful in noise removal operations
pca = PCA(0.50).fit(X)
print(pca.n_components_)

# Noise removal
components = pca.transform(X)
filtered = pca.inverse_transform(components)
```

### PCA and Scaling

If the raw data is used principal component analysis will tend to give more
emphasis to those variables that have higher variances than to those variables
that have very low variances. In effect the results of the analysis will depend
on what units of measurement are used to measure each variable. That would imply
that a principal component analysis should only be used with the raw data if all
variables have the same units of measure. And even in this case, only if you
wish to give those variables which have higher variances more weight in the
analysis.

For example, the two variables of your data-set might be having same units of
measurement and may lie  loosely in the range of 0-255 or whatever. But if one
of the variable always has values in range 100-255 and the other one in between
1-10 for all the records , PCA will automatically give more weight to the first
variable. For this reason standardization is required, not only centering.

The other reason comes from the computational requirements, which might  not be
relevant here, if  most of your variables take  higher order values like
13456789  and on top of that you are using 10 million records of data for PCA ,
then its highly likely that your program will crash out of memory or will take
forever to complete.
Reference: Tibshirani book: Introduction to Statistical Learning,
paragraph 10.2.3 on page 380.

## Clustering Algorithms

Clustering is the grouping of objects together so that objects belonging in 
the same group (cluster) are more similar to each other than those in 
other groups (clusters). 

Also for what concerns unsupervised learning in general, it is a good idea to
apply standardization to data.


### K-Means

There are different implementations of the k-Means algorithm, the most famous
ones are:

* Lloyd/Forgy: which are basically the same, it's just that in the Forgy version
    the distribution of data is considered continuous and we must infer the
    distribution the data is following, but basically the algorithm is this one:
        * take num_clusters centers in some way:
            * randomly in R^d space
            * taking random data points from the dataset
            * using the farthest observations in the dataset
            * using kmeans++ (smarter choice)
        * compute the euclidean distances between centroids and data points and
            then assign points to clusters according to the minimum euclidean
            distance
        * repeat the last step until centroids don't change anymore or until a
            certain tolerance is reached in terms of sum of squared distances
            within clusters to the cluster centers
* MacQueen: it is identical to the Forgy version but cluster centers are updated
    also everytime there is a cluster change
* Hartigan-Wong: Similar to MacQueen in the sense that we update cluster centers
    more frequently (also at each cluster change) but instead of using the
    euclidean distance to perform cluster assignments we use the total
    within-cluster sum of squares. But centroids are still calculated as the
    mean of data points.


#### How does kmeans++ work?
The sklearn implementation by default uses the Lloyd version of the algorithm,
and uses kmeans++ as a technique to initialize centroids. Kmeans++ strategy can
be summarized in the following steps:

1. Choose one center uniformly at random from among the data points.
2. For each data point x, compute D(x), the distance between x and the nearest 
    center that has already been chosen.
3. Choose one new data point at random as a new center, using a weighted probability 
    distribution where a point x is chosen with probability proportional to D(x)^2.
4. Repeat Steps 2 and 3 until k centers have been chosen.
5. Now that the initial centers have been chosen, proceed using standard k-means clustering.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# We generate a dataset, discarding labels
ds, _ = make_blobs(n_samples=3000, centers=3, n_features=2, random_state=0)

# Standard Scaling of Data
X = StandardScaler().fit_transform(ds)

# Setting of K-Means algorithm with 3 clusters
nclust = 3 
model = KMeans(nclust)
model.fit(X)

# Assign cluster label to each points
labels = model.predict(X)

# Store cluster centroids
centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Plot clusters with different colors
plt.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')
plt.scatter(centroids_x, centroids_y, marker='X', s=200)
plt.show()
```


```python
# When plotting centroids we can do
plt.scatter(x,y, c=labels)
plt.scatter(centroids_x, centroids_y, marker='X', s=200)
```


### Agglomerative Clustering (or Hierarchical Clustering)

This kind of clustering does not need necessarily the user to specify the 
number of clusters.
Initially, each point is considered as a separate cluster, then it 
recursively clusters the points together depending upon the distance 
between them. The points are clustered in such a way that the distance 
between points within a cluster is minimum and distance between the 
cluster is maximum. Commonly used distance measures are Euclidean distance, 
Manhattan distance or Mahalanobis distance. Unlike k-means clustering, it is 
"bottom-up" approach.
The main idea behind Agglomerative clustering is that each node first starts 
in its own cluster, and then pairs of clusters recursively merge together 
in a way that minimally increases a given linkage distance.
The main advantage of Agglomerative clustering (and hierarchical clustering 
in general) is that you don’t need to specify the number of clusters. 
That of course, comes with a price: performance. But, in sklearn’s 
implementation, you can specify the number of clusters to assist the
algorithm’s performance.



```python
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(ds)
model = AgglomerativeClustering(affinity = 'euclidean', linkage = 'ward')
clust_labels1 = model.fit_predict(X)
```

Anyway we can also help it by providing the number of clusters in this way:
```python
nclust = 3
model = AgglomerativeClustering(n_clusters=nclust, affinity = 'euclidean', linkage = 'ward')
clust_labels = model.fit_predict(X)
```




### Spectral Clustering

The Spectral clustering technique applies clustering to a projection of 
the normalized Laplacian. When it comes to image clustering,
Spectral clustering works quite well.

```python
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(ds)
nclust = 3
model = SpectralClustering(n_clusters=nclust, affinity="precomputed", n_init=200))
clust_labels = model.fit_predict(X)
```


### Affinity Propagation

It does not require the number of cluster to be estimated and provided before 
starting the algorithm.  It makes no assumption regarding the internal 
structure of the data points.
Affinity propagation is a bit different. Unlike the previous algorithms, 
this one does not require the number of clusters to be determined before 
running the algorithm.
Affinity propagation performs really well on several computer vision and 
biology problems, such as clustering pictures of human faces and 
identifying regulated transcripts


```python
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(ds)
model = AffinityPropagation(damping = 0.5, max_iter = 250, affinity = 'euclidean')
model.fit(X)
clust_labels2 = model.predict(X)
cent = model.cluster_centers_
```


### Gaussian Mixture Modeling

It is probabilistic based clustering or kernel density estimation based clusterig.
The clusters are formed based on the Gaussian distribution of the centers. 


```python
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(ds)
model = GaussianMixture(n_components=nclust,init_params='kmeans')
model.fit(X)
clust_labels = model.predict(X)
```

### DBSCAN

It is a density-based clustering algorithm: given a set of points in some space,
it groups together points that are closely packed together (points with many
nearby neighbors), marking as outliers points that lie alone in low-density
regions (whose nearest neighbors are too far away). DBSCAN is one of the most
common clustering algorithms and also most cited in scientific literature.

The DBSCAN algorithm basically requires 2 parameters:
* eps:  the minimum distance between two points. It means that if the distance
    between two points is lower or equal to this value (eps), these points are
    considered neighbors. Generally it is preferred to have this parameter quite
    low
* minPoints: the minimum number of points to form a dense region. For example,
    if we set the minPoints parameter as 5, then we need at least 5 points to
    form a dense region. Generally a rule of thumb for this parameter is to have
    it can be derived from a number of dimensions (D) in the data set, as
    minPoints `>= D + 1`. Larger values are usually better for data sets with noise
    and will form more significant clusters. The minimum value for the minPoints
    must be 3, but the larger the data set, the larger the minPoints value that
    should be chosen.


```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(ds)
db = DBSCAN(eps=0.3, min_samples=10)
db.fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise, that is '-1' if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
```


## OPTICS

Finds core samples of high density and expands clusters from them. This example 
uses data that is generated so that the clusters have different densities.

The clustering is first used in its automatic settings, which is the 
sklearn.cluster.OPTICS algorithm, and then setting specific thresholds 
on the reachability, which corresponds to DBSCAN.
We can see that the different clusters of OPTICS can be recovered with 
different choices of thresholds in DBSCAN.

```python
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(ds)

clust = OPTICS(min_samples=9, rejection_ratio=0.5)
clust.fit(X)
labels = clust.labels_[clust.ordering_]
```


## Evaluation of Clustering Algorithm

Generally metrics which can be used to evaluate clustering techniques
can be subdivided into:

* Internal Evaluation (Intrinsic), it is useful in and of itself
    * helps understand the makeup of our data
* External Evaluation (Extrinsic), so we use clustering to help us solve another
    problem, the applications are:
    * used to represent for example data with cluster features
    * to train different classifiers for each sub-population
    * identify and eliminate outliers/corrupted points


### Internal Evaluation for Clustering

When a clustering result is evaluated based on the data that was clustered itself,
this is called internal evaluation. These methods usually assign the best 
score to the algorithm that produces clusters with high similarity within 
a cluster and low similarity between clusters. One drawback of using internal 
criteria in cluster evaluation is that high scores on an internal measure do 
not necessarily result in effective information retrieval applications.
Additionally, this evaluation is biased towards algorithms that use the same 
cluster model. For example, k-means clustering naturally optimizes object distances,
and a distance-based internal criterion will likely overrate the resulting clustering.

Therefore, the internal evaluation measures are best suited to get some insight 
into situations where one algorithm performs better than another, but this shall 
not imply that one algorithm produces more valid results than another.
Validity as measured by such an index depends on the claim that this kind of structure 
exists in the data set. An algorithm designed for some kind of models has no chance if 
the data set contains a radically different set of models, or if the evaluation 
measures a radically different criterion. For example, k-means clustering can 
only find convex clusters, and many evaluation indexes assume convex clusters.
On a data set with non-convex clusters neither the use of k-means, nor of an evaluation
criterion that assumes convexity, is sound. 


#### Elbow Method

This method looks at the percentage of variance explained as a function of the number
of clusters: One should choose a number of clusters so that adding another cluster
doesn't give much better modeling of the data. More precisely, if one plots the 
percentage of variance explained by the clusters against the number of clusters,
the first clusters will add much information (explain a lot of variance), but at
some point the marginal gain will drop, giving an angle in the graph. The number 
of clusters is chosen at this point, hence the "elbow criterion". This "elbow" 
cannot always be unambiguously identified. Percentage of variance explained is the 
ratio of the between-group variance to the total variance, also known as an F-test.
A slight variation of this method plots the curvature of the within group variance.

NOTE: These two methods are the same, they just compute things in different order it seems

##### Method 1: Average Within Cluster Sum of Squares

* This plot shows a decreasing elbow curve.
* This is a sort of in-cluster variance.

In this method As k increases, the sum of squared distance tends to zero. 
Imagine we set k to its maximum value n (where n is number of samples) each sample will 
form its own cluster meaning sum of squared distances equals zero.

Inertia is related to the well known concept of cost/loss function in a classification algorithm 
and is calculated as the sum of squared distance for each point to it's closest centroid, 
i.e., its assigned cluster. 

```python
import matplotlib.pyplot as plt

sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(K, sum_of_squared_distances, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(list(K))
plt.show()
```

##### Method 1.1: Slight methodological variation on previos method

* This plot shows an increasing elbow curve.

```python
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
```


##### Method 2: Percentage of Variance Explained

* This plot shows an increasing elbow curve.
* Percentage of variance explained is the ratio of the between-group variance to the total variance, also known as an F-test. 
* A slight variation of this method plots the curvature of the within group variance.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist, pdist

def elbow(df, n):
    kMeansVar = [KMeans(n_clusters=k).fit(df.values) for k in range(1, n)]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_euclid = [cdist(df.values, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(df.values)**2) / df.values.shape[0]
    bss = tss - wcss
    plt.plot(bss)
    plt.show()
```


#### Silhoutte Scores

```python
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(ds)
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
metrics.silhouette_score(X, labels, metric='euclidean')
```

* Advantages
    * The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters.
    * The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
* Drawbacks
    * The Silhouette Coefficient is generally higher for convex clusters than other concepts of clusters, such as density based clusters like those obtained through DBSCAN.

#### Calinski-Harabasz Index

The Calinski-Harabaz index also known as the Variance Ratio Criterion - can be used to evaluate the model, 
where a higher Calinski-Harabaz score relates to a model with better defined clusters.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(ds)

kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
metrics.calinski_harabaz_score(X, labels)  
```

* Advantages
    * The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    * The score is fast to compute
* Drawbacks
    * The Calinski-Harabaz index is generally higher for convex clusters than other concepts of clusters, such as density based clusters like those obtained through DBSCAN.


#### Davis Bouldin Index

The Davies-Bouldin index can be used to evaluate the model, where a lower Davies-Bouldin
index relates to a model with better separation between the clusters.


* Advantages
    * The computation of Davies-Bouldin is simpler than that of Silhouette scores
    * The index is computed only quantities and features inherent to the dataset.
* Drawbacks
    * The Davies-Boulding index is generally higher for convex clusters than other concepts 
        of clusters, such as density based clusters like those obtained from DBSCAN.
    * The usage of centroid distance limits the distance metric to Euclidean space.
    * A good value reported by this method does not imply the best information retrieval.



```python
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(ds)
kmeans = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans.labels_
davies_bouldin_score(X, labels)
```


#### Dunn Index

At the moment is not present in scikit learn but although computationally expensive it is 
one of the most useful indexes for internal evaluation 


### External Evaluation for Clustering

If we had access to the real labels of the clusters we could use these two metrics to
evaluate our clustering algorithms:

* Purity
* Mutual Information
* Jaccard Index
* Fowlkes–Mallows Index
* Adjusted Rand Score (ARS) 
* Normalized Mutual Information (NMI)
* Other metrics generally used by classification procedures


In external evaluation, clustering results are evaluated based on data that was not 
used for clustering, such as known class labels and external benchmarks. Such 
benchmarks consist of a set of pre-classified items, and these sets are often created 
by (expert) humans. Thus, the benchmark sets can be thought of as a gold standard 
for evaluation. These types of evaluation methods measure how close the clustering 
is to the predetermined benchmark classes. However, it has recently been discussed 
whether this is adequate for real data, or only on synthetic data sets with a factual 
ground truth, since classes can contain internal structure, the attributes present may 
not allow separation of clusters or the classes may contain anomalies. Additionally, 
from a knowledge discovery point of view, the reproduction of known knowledge may 
not necessarily be the intended result. In the special scenario of constrained 
clustering, where meta information (such as class labels) is used already in the 
clustering process, the hold-out of information for evaluation purposes is non-trivial.

A number of measures are adapted from variants used to evaluate classification tasks. 
In place of counting the number of times a class was correctly assigned to a single data 
point (known as true positives), such pair counting metrics assess whether each pair of 
data points that is truly in the same cluster is predicted to be in the same cluster


### Normalized Mutual Information

Mutual Information of two random variables is a measure of the mutual 
dependence between the two variables. Normalized Mutual Information is 
a normalization of the Mutual Information (MI) score to scale the results
between 0 (no mutual information) and 1 (perfect correlation). In other 
words, 0 means dissimilar and 1 means a perfect match.


### Adjusted Rand Score

Adjusted Rand Score on the other hand, computes a similarity measure 
between two clusters. ARS considers all pairs of samples and counts pairs
that are assigned in the same or different clusters in the predicted and 
true clusters.

If that's a little weird to think about, have in mind that, for now, 0
is the lowest similarity and 1 is the highest.



#### Zoo of metrics

```python
# External Evaluation
# An estimate of the number of clusters could be retrieved by using DBSCAN or
# other clustering algorithms
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))

# Internal Evalution
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
```


## Manifold Learning to Visualize High Dimensional Data

We have seen how principal component analysis can be used in the dimensionality
reduction task—reducing the number of features of a dataset while maintaining the
essential relationships between the points. While PCA is flexible, fast, and easily
interpretable, it does not perform so well when there are nonlinear relationships 
within the data.


To address this deficiency, we can turn to a class of methods known as 
manifold learning a class of unsupervised estimators that seeks to describe datasets as 
lowdimensional manifolds embedded in high-dimensional spaces. 

Manifold learning is an approach to non-linear dimensionality reduction.
Algorithms for this task are based on the idea that the dimensionality of 
many data sets is only artificially high.


Manifold Learning pros/cons:

* manifold learning is weaker with respect to missing data
* noise in data in manifold learning can drastically change the embedding
* there is not quantitative way to choose optimal number of neighbors in manifold learning
* in manifold learning the globally optimal number of dimensions is difficult to determine,
    in contrast PCA let us find the output dimension based on the explained variance
* in manifold learning, the meaning of the embedded dimensions is not always clear,
    while in PCA they have a clear meaning


We can use the techniques explained in this section to:

* Visualize data
* Visualize data to infer clusters
* Visualize clustering, once it is finished, 

Generally for this purpose common options are:
* PCA, performs poorly in visualization, but for not so highly dimensional data
    is a good choice, since it has the concept of explained variance and axes
    have a meaning
* MDS, tries to preserve the original data structure, but it is not robust
    agains nonlinear highly dimensional transformations
* LLE, is an improvement with respect to MDS for what concerns nonlinear transformations,
    but instead of preserving the distance between each pair of points in the dataset,
    it just preserves the distances between neighboring points
* Isomap, this generally leads to better result with respect to LLE
* t-SNE, which exaggerates clusters, works very well but it is much slower with
    respect to other methods


Generally it may be a good idea to use both techniques.
But just consider the question: do I want data to look more like clusters 
(t-SNE), or do I want to see data more scattered according to how similar
they are individually (MDS).

MDS will get you more of a grid with the points scattered through them,
if our clustering is good then you should have groups (by color) of 
points with possibly some more bare places between them


### t-SNE

The t-SNE algorithm comprises two main stages. First, t-SNE constructs a
probability distribution over pairs of high-dimensional objects in such 
a way that similar objects have a high probability of being picked, 
whilst dissimilar points have an extremely small probability of being picked.
Second, t-SNE defines a similar probability distribution over the points 
in the low-dimensional map, and it minimizes the Kullback–Leibler divergence
between the two distributions with respect to the locations of the points
in the map. Note that whilst the original algorithm uses the Euclidean 
distance between objects as the base of its similarity metric, this should 
be changed as appropriate. 

A feature of t-SNE is a tuneable parameter, "perplexity" which says (loosely)
how to balance attention between local and global aspects of your data. The parameter is,
in a sense, a guess about the number of close neighbors each point has. The perplexity 
value has a complex effect on the resulting pictures. The original paper says, "The performance of 
SNE is fairly robust to changes in the perplexity, and typical values are between 5 and 50."
But the story is more nuanced than that. Getting the most from t-SNE may mean analyzing multiple
plots with different perplexities.

* Perplexity value should be between 2 and 50 and also smaller than the number of points
* Larger datasets usually require a larger perplexity
* Plot with different numbers of iterations, until reaching a stable configuration, this means
    that increasing the number of iterations, the situation does not change
* There's no fixed number of steps that yields a stable result. Different data sets can 
    require different numbers of iterations to converge.
* As a result, it naturally expands dense clusters, and contracts sparse ones, evening out cluster sizes,
    so cluster size means nothing in t-SNE, Rather, density equalization happens by design and 
    is a predictable feature of t-SNE.
* The basic message is that distances between well-separated clusters in a t-SNE plot may mean nothing.
    we generally increase perplexity if we want to check for cluster distances
    but also with a perplexity of 50 sometimes we still see the same distances, other times we may be lucky 
    and increasing perplexity values may alllow us to check for distances

```python
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(ds)

# We apply some sort of clustering on data X
db = DBSCAN(eps=0.3, min_samples=5)
db.fit(X)

labels = db.labels_

# We apply dimensionality reduction
X_embedded = TSNE(n_components=2, n_iter=3000, learning_rate=200).fit_transform(X)

# We plot data
scatter(X_embedded[:,0], X_embedded[:,1], c=labels)
```

### Multi Dimensional Scaling (MDS)

MDS works well when data is not characterized by non linear transformations,
such as rotations, shiftings and so on.

```python
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(ds)

# We apply some sort of clustering on data X
db = DBSCAN(eps=0.3, min_samples=5)
db.fit(X)

labels = db.labels_

# We apply dimensionality reduction
X_embedded = MDS(n_components=2, max_iter=100, n_init=1).fit_transform(X)

# We plot data
scatter(X_embedded[:,0], X_embedded[:,1], c=labels)
```

### Local Linear Embedding (LLE)

```python
from sklearn.manifold import LocallyLinearEmbedding

# The 'modified' version of LLE performs generally better than the classical LLE
model = LocallyLinearEmbedding(n_neighbors = 100, n_components = 2, method = 'modified', eigen_solver='dense')

out = model.fit_transform(X)
```

### Isomap

```python
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(ds)

data = mnist.data[::30]
target = mnist.target[::30]

model = Isomap(n_components =2)
proj = model.fit_transform(faces .data)

plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap=plt.cm.get_cmap('jet', 10))
```


## Appendix A: Understanding the important features in clustering

In order to understand the important features during clustering, we can perform
the clustering and then treat it as a classification problem where the cluster
assignment will be the target, at this point we can use all the techniques that
are available for supervised learning feature selection to understand the
importance, such as random forest feature selection or mutual information and so on.
