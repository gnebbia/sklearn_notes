# Clustering and Dimensionality Reduction

Clustering is the grouping of objects together so that objects belonging in 
the same group (cluster) are more similar to each other than those in 
other groups (clusters). 

## Clustering Algorithms

### K-Means

```python
# number of clusters
nclust = 3 
model = KMeans(nclust)
model.fit(X)
clust_labels = model.predict(X)
cent = model.cluster_centers_
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
model = AffinityPropagation(damping = 0.5, max_iter = 250, affinity = 'euclidean')
model.fit(X)
clust_labels2 = model.predict(X)
cent = model.cluster_centers_
```


### Gaussian Modeling

It is probabilistic based clustering or kernel density estimation based clusterig.
The clusters are formed based on the Gaussian distribution of the centers. 


```python
model = GaussianMixture(n_components=nclust,init_params='kmeans')
model.fit(X)
clust_labels = model.predict(X)
```

### DBSCAN

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=10)
db.fit(X)
labels = db.labels_
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

clust = OPTICS(min_samples=9, rejection_ratio=0.5)
clust.fit(X)
labels = clust.labels_[clust.ordering_]
```


## Evaluation of Clustering Algorithm

Generally metrics which can be used to evaluate clustering techniques
can be subdivided into:

* Internal Evalutaion
* External Evaluation


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

#### Silhoutte Scores

```python
from sklearn import metrics
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
metrics.silhouette_score(X, labels, metric='euclidean')
```

* Advantages
    * The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters.
    * The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
* Drawbacks
    * The Silhouette Coefficient is generally higher for convex clusters than other concepts of clusters, such as density based clusters like those obtained through DBSCAN.

#### Calinski-Harabaz Index

The Calinski-Harabaz index also known as the Variance Ratio Criterion - can be used to evaluate the model, 
where a higher Calinski-Harabaz score relates to a model with better defined clusters.

```python
from sklearn.cluster import KMeans

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
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
# Internal Evalution
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

```

## Dimensionality Reduction to Visualize Data

We can use the following techniques to:
* Visualize data
* Visualize data to infer clusters
* Visualize clustering, once it is finished, 

Generally for this purpose common options are:
* t-SNE, which exaggerates clusters
* MDS, tries to preserve the original data structure


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

A feature of t-SNE is a tuneable parameter, “perplexity,” which says (loosely)
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

```python
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS

# We apply some sort of clustering on data X
db = DBSCAN(eps=0.3, min_samples=5)
db.fit(X)

labels = db.labels_

# We apply dimensionality reduction
X_embedded = MDS(n_components=2, max_iter=100, n_init=1).fit_transform(X)

# We plot data
scatter(X_embedded[:,0], X_embedded[:,1], c=labels)
```



