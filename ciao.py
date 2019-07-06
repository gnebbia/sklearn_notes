#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 gnebbia <nebbionegiuseppe@gmail.com>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

import pandas as pd
import numpy as np
import matplotlib.plotly as plt
from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering
from sklearn.mixture import GaussianMixture #For GMM clustering
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.clu


sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    sum_of_squared_distances.append(km.inertia_)

plot(list(K), sum_of_squared_distances)

X_embedded = TSNE(n_components=2, n_iter=5000).fit_transform(X)
X_embedded.shape

scatter(X_embedded[:,0], X_embedded[:,1], c=labels)

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]

plot(list(Nc), score)

iris = load_iris()
X = iris.data
c=y_data


# PCA
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

X= iris.data

def main():
    iris = load_iris()
    nclust = 3
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_


model = AgglomerativeClustering(n_clusters=nclust, affinity = 'euclidean', linkage = 'ward')
model = AgglomerativeClustering(affinity = 'euclidean', linkage = 'ward')
clust_labels1 = model.fit_predict(X)
model = AffinityPropagation(damping = 0.5, max_iter = 250, affinity = 'euclidean')
model.fit(X)
clust_labels2 = model.predict(X)
cent = model.cluster_centers_

model = GaussianMixture(n_components=nclust,init_params='kmeans')
model.fit(X)
clust_labels3 = model.predict(X)

from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=5)
db.fit(X)
labels = db.labels_

clust = OPTICS(min_samples=5, rejection_ratio=0.6)
clust.fit(X)
labels = clust.labels_[clust.ordering_]
    pass


if __name__ == '__main__':
    main()

from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS

# We apply some sort of clustering on data X
db = DBSCAN(eps=0.3, min_samples=5)
db.fit(X)

labels = db.labels_

# We apply dimensionality reduction
X_embedded = MDS(n_components=2, max_iter=100, n_init=1).fit_transform(X)
scatter(X_embedded[:,0], X_embedded[:,1], c=labels)
