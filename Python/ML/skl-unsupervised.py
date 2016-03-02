###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: April 26, 2015
# summary: scikit playground
# other resources:
# http://scikit-learn.org/stable/tutorial/
###############################################################################

###############################################################################
import sklearn as skl
from sklearn import datasets

import numpy as np
import pandas as pd
# to get ggplot-like style for plots
pd.set_option('display.mpl_style', 'default')
pd.set_option('display.line_width', 5000)
pd.set_option('display.max_columns', 60)

import matplotlib.pyplot as plt
plt.ion()  # turn on interactive plotting
plt.style.use('ggplot')  # make matplotlib appearance similar to ggplot

import functools
import itertools
import os, sys

# Get Python environment parameters
print 'Python version ' + sys.version
print 'Pandas version: ' + pd.__version__

from __future__ import print_function
###############################################################################

###############################################################################
### MODULE 4: Unsupervised Learning
# http://scikit-learn.org/stable/tutorial/
#   http://scikit-learn.org/stable/tutorial/statistical_inference/index.html
#       http://scikit-learn.org/stable/tutorial/statistical_inference/unsupervised_learning.html

###############################################################################
################################  CLUSTERING ##################################
###############################################################################

# It is VERY IMPORTANT to choose the proper clustering method for the data.
# For a great review of clustering methods, and how different methods could result in very different results based on the underlying structure of the data.
# http://scikit-learn.org/stable/modules/clustering.html

###############################################################################
############################# K-MEANS CLUSTERING ##############################
###############################################################################
### K-MEANS CLUSTERING
# http://scikit-learn.org/stable/modules/clustering.html#k-means
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

### PARAMETERS
# n_clusters: The number of clusters to form as well as the number of centroids to generate.
# n_init: Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
# init : {‘k-means++’, ‘random’ or an ndarray}
#   Method for initialization, defaults to ‘k-means++’:
#   (a) ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.
#   (b) ‘random’: choose k observations (rows) at random from data for the initial centroids.
#   (c) ndarray provided by the user to be used as initial cluster centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

### ATTRIBUTES:
# cluster_centers_ : coordinates of cluster centers
# labels_ : labels of each point
# inertia_ : sum of distances of samples to their closest cluster center.

### IMPLEMENTATION
# k-means algorithm is very fast (one of the fastest clustering algorithms available), but it falls in local minima. That’s why it can be useful to restart it several times.
# use the parameter n_init to run k-means multiple times with different seeds.

# If the number of points is too large (>10K), use MiniBatchKMeans() to do incremental updates of the cluster positions for faster convergence.

# Note that K-Means clustering currently only supports Euclidean distance. To use another distance (e.g., Manhattan), use other clustering methods such as SpectralClustering and AgglomerativeClustering().
###############################################################################


###############################################################################
### EXAMPLE OF K-MEANS CLUSTERING IN 1 DIMENSION
from sklearn import cluster, datasets
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X_iris)
print(k_means.labels_[::10])
print(y_iris[::10])
###############################################################################


###############################################################################
### VECTOR QUANTIZATION USING K-MEANS (IMAGE SIZE REDUCTION)
# The goal is to choose a small number of exemplars to compress the information
import scipy as sp

# Import the data (a nxn matrix of greyscale values that represent a picture)
try:
    lena = sp.lena()
except AttributeError:
    from scipy import misc
    lena = misc.lena()

# reshape the nxn matrix into a nx(1,n) shape; which is n vector of 1xn dimension. Each vector is one row in the image.
X = lena.reshape((-1, 1)) # We need an (n_sample, n_feature) array

# Fit the k-means model
k_means = cluster.KMeans(n_clusters=5, n_init=1)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_

# Construct the new vector by choosing the cluster centers values for each element based on the cluster label. This is similar to R's 'score' attribute.
lena_compressed = np.choose(labels, values)

# Reshape the vector into its original matrix shape.
lena_compressed.shape = lena.shape


## Plot the image
import matplotlib.cm as cm  # colormaps
plt.figure()
plt.imshow(lena, cmap=cm.gray)
plt.imshow(lena_compressed, cmap=cm.gray)
plt.draw()
plt.close()
###############################################################################


###############################################################################
### EXAMPLE OF CHOOSING INITIAL CLUSTER CENTERS FOR K-MEANS
# http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_stability_low_dim_dense.html#example-cluster-plot-kmeans-stability-low-dim-dense-py
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans


### USER INPUTS
random_state = np.random.RandomState(0)

# Number of run (with randomly generated dataset) for each strategy
# to be able to compute an estimate of the standard deviation
n_runs = 5

# k-means models can do several random inits to be able to trade
# CPU time for convergence robustness
n_init_range = np.array([1, 5, 10, 15, 20])


### Datasets generation parameters
n_samples_per_center = 100

# we will have a grid_size x grid_size set of cluster centers and measured points around them
grid_size = 3
n_clusters = grid_size ** 2

# Size of random noise (related to standard deviation of the Gaussian noise)
scale = 0.1


### FUNCTION TO CREATE RANDOM DATA POINTS FOR CLUSTERING
def make_data(random_state, n_samples_per_center, grid_size, scale):
    random_state = check_random_state(random_state)

    # Establish the cluster center points
    centers = np.array([[i, j]
                        for i in range(grid_size)
                        for j in range(grid_size)])
    n_clusters_true, n_features = centers.shape

    # Add noise to cluster centers to create final data points
    noise = random_state.normal(
        scale=scale, size=(n_samples_per_center, centers.shape[1]))

    X = np.concatenate([c + noise for c in centers])
    y = np.concatenate([[i] * n_samples_per_center
                        for i in range(n_clusters_true)])
    return shuffle(X, y, random_state=random_state)


### Part 1: Quantitative evaluation of various init methods
fig = plt.figure()
plots = []
legends = []

cases = [
    (KMeans, 'k-means++', {}),
    (KMeans, 'random', {}),
    (MiniBatchKMeans, 'k-means++', {'max_no_improvement': 3}),
    (MiniBatchKMeans, 'random', {'max_no_improvement': 3, 'init_size': 500}),
]

for factory, init, params in cases:
    print("Evaluation of %s with %s init" % (factory.__name__, init))
    inertia = np.empty((len(n_init_range), n_runs))

    for run_id in range(n_runs):
        X, y = make_data(run_id, n_samples_per_center, grid_size, scale)
        for i, n_init in enumerate(n_init_range):
            km = factory(n_clusters=n_clusters, init=init, random_state=run_id,
                         n_init=n_init, **params).fit(X)
            inertia[i, run_id] = km.inertia_
    p = plt.errorbar(n_init_range, inertia.mean(axis=1), inertia.std(axis=1))
    plots.append(p[0])
    legends.append("%s with %s init" % (factory.__name__, init))

plt.xlabel('n_init')
plt.ylabel('inertia')
plt.legend(plots, legends)
plt.title("Mean inertia for various k-means init across %d runs" % n_runs)
plt.draw()
plt.close('all')


### Part 2: Qualitative visual inspection of the convergence
X, y = make_data(random_state, n_samples_per_center, grid_size, scale)
km = MiniBatchKMeans(n_clusters=n_clusters, init='random', n_init=1,
                     random_state=random_state).fit(X)

fig = plt.figure()
for k in range(n_clusters):
    my_members = km.labels_ == k
    color = cm.spectral(float(k) / n_clusters, 1)
    plt.plot(X[my_members, 0], X[my_members, 1], 'o', marker='.', c=color)
    cluster_center = km.cluster_centers_[k]
    plt.plot(cluster_center[0], cluster_center[1], 'o',
             markerfacecolor=color, markeredgecolor='k', markersize=6)
    plt.title("Example cluster allocation with a single random init\n"
              "with MiniBatchKMeans")

plt.draw()
plt.close('all')
###############################################################################


###############################################################################
### MINI-BATCH K-MEANS
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
# Alternative online implementation that does incremental updates of the centers positions using mini-batches.
# For large scale learning (say n_samples > 10k) MiniBatchKMeans is probably much faster to than the default batch implementation.

###############################################################################


###############################################################################
################ HIERARCHICAL AGGLOMERATIVE CLUSTERING (Ward) #################
###############################################################################
# http://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
# Agglomerative: bottom-up clustering
# Divisive: top-down clustering

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from datetime_tut import datetime_tut as dt
import time

# Generate data
lena = sp.misc.lena()

# Downsample the image by a factor of 4
lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
X = np.reshape(lena, (-1, 1))

# Define the structure A of the data. Pixels connected to their neighbors.
# Specify which samples can be clustered together by giving a connectivity graph.
# Graphs in the scikit are represented by their adjacency matrix.
connectivity = grid_to_graph(*lena.shape)

# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()
n_clusters = 15  # number of regions
ward = AgglomerativeClustering(n_clusters=n_clusters,
        linkage='ward', connectivity=connectivity).fit(X)
label = np.reshape(ward.labels_, lena.shape)

print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", label.size)
print("Number of clusters: ", np.unique(label).size)

# Plot the results on an image
plt.figure(figsize=(5, 5))
plt.imshow(lena, cmap=plt.cm.gray)  # this is just for reference
for l in range(n_clusters):
    plt.contour(label == l, contours=1,
                colors=[plt.cm.spectral(l / float(n_clusters)), ])
plt.xticks(())
plt.yticks(())
plt.draw()
plt.close('all')
###############################################################################


###############################################################################
### AGGLOMERATIVE CLUSTERING ON 2D DATA
# http://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html#example-cluster-plot-digits-linkage-py
# Authors: Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2014

from time import time

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from sklearn import manifold, datasets

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
n_samples, n_features = X.shape

np.random.seed(0)

def nudge_images(X, y):
    # Having a larger dataset shows more clearly the behavior of the
    # methods, but we multiply the size of the dataset only by 2, as the
    # cost of the hierarchical clustering methods are strongly
    # super-linear in n_samples
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),
                                  .3 * np.random.normal(size=2),
                                  mode='constant',
                                  ).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y


X, y = nudge_images(X, y)


### Visualize the clustering
def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()


### 2D embedding of the digits dataset
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")

from sklearn.cluster import AgglomerativeClustering

for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    t0 = time()
    clustering.fit(X_red)
    print("%s : %.2fs" % (linkage, time() - t0))

    plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)


plt.draw()
###############################################################################


###############################################################################
########################### FEATURE AGGLOMERATION #############################
###############################################################################
# merge together similar features (clustering in the feature direction)
digits = datasets.load_digits()
images = digits.images
X = np.reshape(images, (len(images), -1))
connectivity = grid_to_graph(*images[0].shape)

agglo = cluster.FeatureAgglomeration(connectivity=connectivity, n_clusters=32)
agglo.fit(X)
X_reduced = agglo.transform(X)
X_approx = agglo.inverse_transform(X_reduced)
images_approx = np.reshape(X_approx, images.shape)

plt.figure()
for i in range(0,8):
    plt.subplot(2,4,i+1)
    plt.imshow(images_approx[i], cmap=plt.cm.gray)
    plt.draw()
plt.close()
###############################################################################


###############################################################################
#################### PRINCIPAL COMPONENT ANALYSIS (PCA) #######################
###############################################################################

###############################################################################
### PRINCIPAL COMPONENT ANALYSIS (PCA)
# http://scikit-learn.org/stable/modules/decomposition.html#pca
# Create a signal with only 2 useful dimensions
x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)
x3 = x1 + x2
X = np.c_[x1, x2, x3]

from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(X)
print(pca.explained_variance_)

# As we can see, only the 2 first components are useful
pca.n_components = 2
X_reduced = pca.fit_transform(X)
X_reduced.shape
###############################################################################


###############################################################################
################### INDEPENDENT COMPONENT ANALYSIS (PCA) ######################
###############################################################################

###############################################################################
### INDEPENDENT COMPONENT ANALYSIS
# http://scikit-learn.org/stable/modules/decomposition.html#ica
# Generate sample data
time = np.linspace(0, 10, 2000)
s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
S = np.c_[s1, s2]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise
S /= S.std(axis=0)  # Standardize data

# Mix data
A = np.array([[1, 1], [0.5, 2]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = decomposition.FastICA()
S_ = ica.fit_transform(X)  # Get the estimated sources
A_ = ica.mixing_.T

# Validate the estimates
np.allclose(X,  np.dot(S_, A_) + ica.mean_)  # Returns True if two arrays are element-wise equal within a tolerance.
estimates = (np.dot(S_, A_) + ica.mean_)

plt.figure()
plt.plot(s1)  # 1st underlying component
plt.plot(s2)  # 2nd underlying component
plt.plot(S)  # observed signal
plt.subplot(2,1,0)
plt.plot(S_)  # estimated signal
plt.subplot(2,1,1)
plt.plot(estimates)
plt.draw()
plt.close()
###############################################################################

