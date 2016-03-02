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

localPath = '/Users/amirkavousian/Documents/Py_Codes/Tutorials_Files/Stats/'
###############################################################################

###############################################################################
### STANDARDIZING FEATURES BEFORE CLASSIFICATION
# Standardization of a dataset is a common requirement for many machine learning estimators:
# they might behave badly if the individual feature do not more or less look like standard normally distributed data
# (e.g. Gaussian with 0 mean and unit variance).
# For instance many elements used in the objective function of a learning algorithm
# (such as the RBF kernel of Support Vector Machines or the L1 and L2 regularizers of linear models)
# assume that all features are centered around 0 and have variance in the same order.
# If a feature has a variance that is orders of magnitude larger that others,
# it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.


### METHOD: sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# Standardize features by removing the mean and scaling to unit variance
# Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set.
# Mean and standard deviation are then stored to be used on later data using the transform method.

# To use StandardScaler() function:
X = StandardScaler().fit_transform(X)
###############################################################################

###############################################################################
############################# K NEAREST NEIGHBOR ##############################
###############################################################################

##############################################################################
### CLASSIFICATION USING KNN

import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)

### Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
# Setup aside the 10 last permuted index as test set
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

### Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_neighbors=5, p=2, weights='uniform')
knn.predict(iris_X_test)
iris_y_test

### More on KNN:
# http://scikit-learn.org/stable/modules/neighbors.html#neighbors

# NOTE:For many estimators, including the SVMs, having datasets with unit standard deviation for each feature is important to get good prediction.
##############################################################################




###############################################################################
#################################### SVM ######################################
###############################################################################


###############################################################################
## Faces recognition example using eigenfaces and SVMs
from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
import PIL
from PIL import Image

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# Split into a training set and a test set using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)


# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# Train a SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# Quantitative evaluation of the model quality on the test set
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# Qualitative evaluation of the predictions using matplotlib
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.draw()
###############################################################################


###############################################################################
####################### RANDOM FOREST CLASSIFICATION ##########################
###############################################################################

##############################################################################
### RANDOM FOREST CLASSIFICATION
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# A random forest is a meta estimator that fits a number of decision tree classifiers on
# various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.
# n_estimators: The number of trees in the forest.
# max_features: The number of features to consider when looking for the best split.
# estimators_: The collection of fitted sub-estimators.
# classes_: The classes labels (single output problem), or a list of arrays of class labels (multi-output problem).


##############################################################################

##############################################################################
### COMPARISON OF DECISION TREES, RANDOM FOREST, EXTRA-TREES, ADA BOOST CLASSIFIER
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html
# RandomForests and ExtraTrees can be fitted in parallel on many cores as each tree is built independently of the others.
# AdaBoost’s samples are built sequentially and so do not use multiple cores.

# In this example, we compare four different classifiers. And we compare them over three different model setups using different feature sets to start with:
# (a) Using sepal width and the sepal length features only
# (b) using the petal length and sepal length only
# (c) using the petal width and the petal length only

import numpy as np
import matplotlib.pyplot as plt

from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier


### SET PARAMETERS
# Model parameters
n_classes = 3
n_estimators = 30

# Plot parameters
plot_colors = "ryb"
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
plot_idx = 1  # Initiate plot index (will increment this via the loop)

# Run parameters
RANDOM_SEED = 13  # fix the seed on each iteration


### LOAD DATA
iris = load_iris()


### SPECIFY MODELS
models = [DecisionTreeClassifier(max_depth=None),
          RandomForestClassifier(n_estimators=n_estimators),
          ExtraTreesClassifier(n_estimators=n_estimators),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             n_estimators=n_estimators)]


### FIT DIFFERENT MODELS TO DIFFERENT FEATURE SETS
# Iterate through three different feature sets
for pair in ([0, 1], [0, 2], [2, 3]):
    # Iterate through different classifiers
    for model in models:
        ## Prepare data for classifier
        # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        ## Train and get score
        # Note: the fit() method of classifiers is self-applied; i.e., it changes the classifier itself. So we should clone() it if we want to reuse the classifier using different setup later.
        clf = clone(model)
        clf = model.fit(X, y)

        scores = clf.score(X, y)

        ## Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]
        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(len(model.estimators_))
        print( model_details + " with features", pair, "has a score of", scores )

        ## Plot
        plt.subplot(3, 4, plot_idx)
        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a black outline
        xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),
                                             np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        for i, c in zip(xrange(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=c, label=iris.target_names[i],
                        cmap=cmap)

        plot_idx += 1  # move on to the next plot in sequence

plt.suptitle("Classifiers on feature subsets of the Iris dataset")
plt.axis("tight")

plt.draw()
plt.close()
##############################################################################

###############################################################################
############################### NAIVE BAYES ###################################
###############################################################################
# http://scikit-learn.org/stable/modules/naive_bayes.html
# Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features.

# Given a class variable y and a dependent feature vector x_1 through x_n, Bayes’ theorem states the following relationship:
# P(y | x1,x2,...,xn) = P(y) * P(x1,x2,...,xn) / P(x1,x2,...,xn)

# Using the naive independence assumption that observations are independent from each other;
# ie, given y, observing one does not change the probability of observing any others.
# P(xi | y, x1,x2,...,xi-1,xi+1,...,xn) = P(xi | y)

# for all i , this relationship is simplified to:
# P(y | x1,x2,...xn) = P(y) * multiplication(P(xi | y) / P(x1,x2,...,xn)

# Since P(x_1, ..., x_n) is constant given the input, we can use the following classification rule:
# P(y ] x1,x2,...xn)   [is proportional to]   P(y) * multiplication(P(xi | y)

# Which will result in the following estiamte for y (based on maximum likelihood)
# y_hat = argmax_y {P(y) * multiplication(P(xi | y)}

# and we can use Maximum A Posteriori (MAP) estimation to estimate P(y) and P(x_i | y); the former is then the relative frequency of class y in the training set.
# The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of P(x_i | y).
# Different flavors of naive Bayes classifiers include: Gaussian Naive Bayes, Multinomial Naive Bayes, Bernoulli Baive Bayes.

# In general, Naive Bayes classifiers work well is some classification situations such as document classification and spam filtering.
# They require a small amount of training data to estimate the necessary parameters.
# Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods.
# On the flip side, although naive Bayes is known as a decent classifier,
# it is known to be a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.

# More on the theory of naive Bayes:
# http://www.cs.unb.ca/profs/hzhang/publications/FLAIRS04ZhangH.pdf
##############################################################################


###############################################################################
### GAUSSIAN NAIVE BAYES
# http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
# In GNB, the likelihood of features, given the class, is assumed to be Gaussian.
# P(xi | y) = Normal(mu_y, sigma_y)
# The parameters mu_y and sigma_y are estimated using maximum likelihood.

### MODEL:
# sklearn.naive_bayes.GaussianNB()
# The GaussianNB() function does not accept any parameters.

#######################################
### GAUSSIAN NAIVE BAYES ON IRIS DATA
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))
#######################################

###############################################################################

###############################################################################
### MULTINOMIAL NAIVE BAYES
# Multinomial naive Bayes algorithm for multinomially distributed data.
# It is one of the two classic naive Bayes variants used in text classification
# (where the data are typically represented as word vector counts,
# although tf-idf vectors are also known to work well in practice).
# The distribution is parametrized by vectors
#   \theta_y = (\theta_{y1},...,\theta_{yn}) for each class y,
# where n is the number of features (in text classification, the size of the vocabulary) and
#   \theta_{yi} is the probability P(x_i | y) of feature i appearing in a sample belonging to class y.

# The parameters \theta_y is estimated by a smoothed version of maximum likelihood, i.e. relative frequency counting:
#   theta_hat_{yi} = (N_yi + alpha) / (N_y + alpha*n)
# where N_{yi} = \sum_{x \in T} x_i is the number of times feature i appears in a sample of class y in the training set T,
# and N_{y} = \sum_{i=1}^{|T|} N_{yi} is the total count of all features for class y.
# The smoothing priors alpha >= 0 accounts for features not present in the learning samples and prevents zero probabilities in further computations.
# Setting \alpha = 1 is called Laplace smoothing, while \alpha < 1 is called Lidstone smoothing.

### MODEL
# sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
# Parameters to work with in Multinomial NB:
#   alpha: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
#   fit_prior: Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
#   class_prior: Prior probabilities of the classes. If specified the priors are not adjusted according to the data.

##############################################################################

###############################################################################
### BERNOULLI NAIVE BAYES
# for data that is distributed according to multivariate Bernoulli distributions;
# i.e., there may be multiple features but each one is assumed to be a binary-valued (Bernoulli, boolean) variable.
# The decision rule for Bernoulli naive Bayes is based on
#   P(xi | y) = P(i|y)*xi + (1-P(i|y))*(1-xi)
# which differs from multinomial NB’s rule in that it explicitly penalizes
# the non-occurrence of a feature i that is an indicator for class y,
# where the multinomial variant would simply ignore a non-occurring feature.

# In the case of text classification, word occurrence vectors (rather than word count vectors) may be used to train and use this classifier.
# BernoulliNB might perform better on some datasets, especially those with shorter documents.
# It is advisable to evaluate both models, if time permits.


### MODEL
# sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
# We can set the threshold for binarizing (mapping to booleans) of sample features using the binarize parameter in .
# If None, input is presumed to already consist of binary vectors.

###############################################################################


###############################################################################
########################### ADABoost Classifier ###############################
###############################################################################
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
# An AdaBoost classifier is a meta-estimator that
# begins by fitting a classifier on the original dataset
# and then fits additional copies of the classifier on the same dataset
# but where the weights of incorrectly classified instances are adjusted
# such that subsequent classifiers focus more on difficult cases.

# AdaBoost is not a new classification method; it just improves an existing classifier iteratively by giving more weight to data points that have been mis-classified.

sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
# base_estimator : also called base classifier sklearn's default is DecisionTreeClassifier.
# n_estimators : The maximum number of estimators at which boosting is terminated.


#######################################
### EXAMPLE: TWO-CLASS ADA BOOST
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html

# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


### Construct dataset
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))


### Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(X, y)


### Plot
plot_colors = "br"
plot_step = 0.02
class_names = "AB"

plt.figure(figsize=(10, 5))

# Plot the decision boundaries
# Note how we predicted the class on a mesh grid of x,y values:
#   (a) Set the x and y grid points: np.arange(x_min, x_max, plot_step) and np.arange(y_min, y_max, plot_step)
#   (b) Create a mesh grid of x,y values: np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
#   (c) Flatten x and y mesh values, and zip them together: np.c_[xx.ravel(), yy.ravel()]
#   (d) Predict
#   (e) Reshape the predictions into the grid points: reshape(xx.shape)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

# Predict on the mesh
Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot a contour based on the classification results
plt.subplot(121)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.axis("tight")

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                label="Class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X)  # the numeric value that was produced by the algorithm and choose a class value
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y == i],
             bins=10,
             range=plot_range,
             facecolor=c,
             label='Class %s' % n,
             alpha=.5)
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Scores')

# plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.draw()
plt.close()
#######################################


#######################################
### DISCRETE VS REAL AdaBoost
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html
# The target Y is a non-linear function of 10 input features.

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier


n_estimators = 400
# A learning rate of 1. may not be optimal for both SAMME and SAMME.R
learning_rate = 1.

X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)

X_test, y_test = X[2000:], y[2000:]
X_train, y_train = X[:2000], y[:2000]


### Fit decision trees (base and deep)
# Decision tree of depth 1 (base estimator)
dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train, y_train)
dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

# Decision tree of depth 9 (for comparison to ada boost)
dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, y_train)
dt_err = 1.0 - dt.score(X_test, y_test)


ada_discrete = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME")
ada_discrete.fit(X_train, y_train)

ada_real = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME.R")
ada_real.fit(X_train, y_train)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-',
        label='Decision Stump Error')
ax.plot([1, n_estimators], [dt_err] * 2, 'k--',
        label='Decision Tree Error')


### Fit ada boost estimators and keep the prediction error at each iteration
# On the test set (discrete SAMME)
ada_discrete_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
    ada_discrete_err[i] = zero_one_loss(y_pred, y_test)

# On the training set (discrete SAMME)
ada_discrete_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
    ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)

# On the test set (real SAMME)
ada_real_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
    ada_real_err[i] = zero_one_loss(y_pred, y_test)

# On the training set (real SAMME)
ada_real_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
    ada_real_err_train[i] = zero_one_loss(y_pred, y_train)

ax.plot(np.arange(n_estimators) + 1, ada_discrete_err,
        label='Discrete AdaBoost Test Error',
        color='red')
ax.plot(np.arange(n_estimators) + 1, ada_discrete_err_train,
        label='Discrete AdaBoost Train Error',
        color='blue')
ax.plot(np.arange(n_estimators) + 1, ada_real_err,
        label='Real AdaBoost Test Error',
        color='orange')
ax.plot(np.arange(n_estimators) + 1, ada_real_err_train,
        label='Real AdaBoost Train Error',
        color='green')

ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')

leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)

plt.draw()
plt.close('all')
#######################################
###############################################################################


###############################################################################
######################## LINEAR DISCRIMINANT ANALYSIS #########################
###############################################################################
# http://scikit-learn.org/stable/modules/generated/sklearn.lda.LDA.html#sklearn.lda.LDA
# sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)

# A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.
# The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix.
# The fitted model can also be used to reduce the dimensionality of the input by projecting it to the most discriminative directions.

# (a) The default solver is ‘svd’. It can perform both classification and transform, and it does not rely on the calculation of the covariance matrix.
#     This can be an advantage in situations where the number of features is large. However, the ‘svd’ solver cannot be used with shrinkage.
# (b) The ‘lsqr’ solver is an efficient algorithm that only works for classification. It supports shrinkage.
# (c) The ‘eigen’ solver is based on the optimization of the between class scatter to within class scatter ratio.
#     It can be used for both classification and transform, and it supports shrinkage.
#     However, the ‘eigen’ solver needs to compute the covariance matrix, so it might not be suitable for situations with a high number of features.

#######################################
### LINEAR DISCRIMINANT ANALYSIS EXAMPLE
import numpy as np
from sklearn.lda import LDA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = LDA()
clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))
#######################################

#######################################
### EXAMPLE: NORMAL AND SHRINKAGE LINEAR DISCRIMINANT ANALYSIS FOR TEXT CLASSIFICATION
# http://scikit-learn.org/stable/auto_examples/classification/plot_lda.html

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.lda import LDA


n_train = 20  # samples for training
n_test = 200  # samples for testing
n_averages = 50  # how often to repeat classification
n_features_max = 75  # maximum number of features
step = 4  # step size for the calculation


def generate_data(n_samples, n_features):
    """Generate random blob-ish data with noisy features.

    This returns an array of input data with shape `(n_samples, n_features)`
    and an array of `n_samples` target labels.

    Only one feature contains discriminative information, the other features
    contain only noise.
    """
    X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])

    # add non-discriminative features
    if n_features > 1:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
    return X, y

acc_clf1, acc_clf2 = [], []
n_features_range = range(1, n_features_max + 1, step)
for n_features in n_features_range:
    score_clf1, score_clf2 = 0, 0
    for _ in range(n_averages):
        X, y = generate_data(n_train, n_features)

        clf1 = LDA(solver='lsqr', shrinkage='auto').fit(X, y)
        clf2 = LDA(solver='lsqr', shrinkage=None).fit(X, y)

        X, y = generate_data(n_test, n_features)
        score_clf1 += clf1.score(X, y)
        score_clf2 += clf2.score(X, y)

    acc_clf1.append(score_clf1 / n_averages)
    acc_clf2.append(score_clf2 / n_averages)

features_samples_ratio = np.array(n_features_range) / n_train

plt.figure()
plt.plot(features_samples_ratio, acc_clf1, linewidth=2,
         label="LDA with shrinkage", color='r')
plt.plot(features_samples_ratio, acc_clf2, linewidth=2,
         label="LDA", color='g')

plt.xlabel('n_features / n_samples')
plt.ylabel('Classification accuracy')

plt.legend(loc=1, prop={'size': 12})
plt.suptitle('LDA vs. shrinkage LDA (1 discriminative feature)')
plt.draw()
plt.close()
#######################################


#######################################
# COMPARISON OF LDA AND PCA 2D PROJECTION OF IRIS DATASET
# http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html

# The Iris dataset represents:
#    3 kind of Iris flowers (Setosa, Versicolour and Virginica) with
#    4 attributes: sepal length, sepal width, petal length and petal width.
# Principal Component Analysis (PCA) applied to this data identifies the
# combination of attributes (principal components, or directions in the feature space) that
# account for the most variance in the data.
# Linear Discriminant Analysis (LDA) tries to identify attributes that
# account for the most variance between classes.
# In particular, LDA, in contrast to PCA, is a supervised method, using known class labels.

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA of IRIS dataset')

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA of IRIS dataset')

plt.draw()
plt.close()
#######################################

###############################################################################

###############################################################################
###################### QUADRATIC DISCRIMINANT ANALYSIS ########################
###############################################################################

#######################################
### LINEAR AND QUADRATIC DISCRIMINANT ANALYSIS WITH CONFIDENCE ELLIPSOID
# http://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html

from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

from sklearn.lda import LDA
from sklearn.qda import QDA


### colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)



### generate datasets
def dataset_fixed_cov():
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


###
def dataset_cov():
    '''Generate 2 Gaussians samples with different covariance matrices'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -1.], [2.5, .7]]) * 2.
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


###
# plot functions
def plot_data(lda, X, y, y_pred, fig_index):
    splot = plt.subplot(2, 2, fig_index)
    if fig_index == 1:
        plt.title('Linear Discriminant Analysis')
        plt.ylabel('Data with fixed covariance')
    elif fig_index == 2:
        plt.title('Quadratic Discriminant Analysis')
    elif fig_index == 3:
        plt.ylabel('Data with varying covariances')

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()

    # class 0: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', color='red')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '.', color='#990000')  # dark red

    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', color='blue')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '.', color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             'o', color='black', markersize=10)
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             'o', color='black', markersize=10)

    return splot


def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    splot.set_xticks(())
    splot.set_yticks(())


def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
    plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')


def plot_qda_cov(qda, splot):
    plot_ellipse(splot, qda.means_[0], qda.covariances_[0], 'red')
    plot_ellipse(splot, qda.means_[1], qda.covariances_[1], 'blue')


###
for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
    # LDA
    lda = LDA(solver="svd", store_covariance=True)
    y_pred = lda.fit(X, y).predict(X)
    splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
    plot_lda_cov(lda, splot)
    plt.axis('tight')

    # QDA
    qda = QDA()
    y_pred = qda.fit(X, y, store_covariances=True).predict(X)
    splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
    plot_qda_cov(qda, splot)
    plt.axis('tight')
plt.suptitle('LDA vs QDA')
plt.draw()
plt.close('all')
#######################################

###############################################################################
################## PROBABILITY CALIBRATION FOR CLASSIFIERS ####################
###############################################################################
# http://scikit-learn.org/stable/modules/calibration.html
# When performing classification you often want not only to predict the class label, but also obtain a probability of the respective label.
# The calibration module allows you to better calibrate the probabilities of a given model, or to add support for probability prediction.

### THEORY
# When performing classification one often wants to predict not only the class label,
# but also the associated probability. This probability gives some kind of confidence on the prediction.

# Well calibrated classifiers are probabilistic classifiers for which
# the output of the predict_proba method can be directly interpreted as a confidence level.

# For instance, a well calibrated (binary) classifier should classify the samples
# such that among the samples to which it gave a predict_proba value close to 0.8, approximately 80% actually belong to the positive class.
# I.e., the probability attached to a classification would be close to Precision value of the classifier.


### STATISTICAL MODELS
# (a) LogisticRegression returns well calibrated predictions by default as it directly optimizes log-loss.
#     In contrast, the other methods return biased probabilities; with different biases per method:
# (b) GaussianNB tends to push probabilties to 0 or 1. This is mainly because it makes the assumption that features are conditionally independent given the class.
# (c) RandomForestClassifier shows the opposite behavior: the histograms show peaks at approximately 0.2 and 0.9 probability, while probabilities close to 0 or 1 are very rare.
#     To calibrate the RFC, we need a curve that helps push values closer to 0 and 1 to their respective edges.
#     As a result, the calibration curve shows a characteristic sigmoid shape, indicating that the classifier could trust its “intuition” more and return probabilties closer to 0 or 1 typically.
# (d) Linear Support Vector Classification (LinearSVC) shows an even more sigmoid curve as the RandomForestClassifier, which is typical for maximum-margin methods.


### METHODS AND MODULES
# Two approaches for performing calibration of probabilistic predictions are provided:
# (a) a parametric approach based on Platt’s sigmoid model; and,
# (b) a non-parametric approach based on isotonic regression

# Probability calibration should be done on new data not used for model fitting.
# We use CalibratedClassifierCV to calibrate the classification models. Depending on whether we have set aside a test set:
# (a) If a separate test set has been set aside,
#     the class CalibratedClassifierCV uses a cross-validation generator and
#     estimates for each split the model parameter on the train samples and
#     the calibration of the test samples. The probabilities predicted for the folds are then averaged.
# (b) Already fitted classifiers can be calibrated by CalibratedClassifierCV via the paramter cv=”prefit”.
#     In this case, the user has to take care manually that data for model fitting and calibration are disjoint.


### Calibration performance
# The calibration performance is evaluated with Brier score brier_score_loss, (the smaller the better).

# To visualize the calibration of a classification, we use a plot of 'Mean Prediction Value' vs 'Fraction of Positives'.
# The shape of the calibration curve gives insight on the accuracy and level of aggresivenss of the classification model.
# (a) Calibration curves that are linear, and close to y=x line are unbiased and accurate. Logistic Regression classifier normally shows this behavior.
# (b) Calibration curves with sigmoid shape (S-curve) correspond to an under-confident (non-aggressive) classification model.
#     In these cases, the class probability rarely approaches 0 or 1, and is usually around the middle 0.3-0.7.
#     Linear-SVC is a non-aggressive, under-confident classification model and shows this behavior.
# (c) A transposed-sigmoid (reverse S-shape, where the curve first aggressively goes upward and then slows down and again picks up at the end)
#     corresponds to aggressive, over-confident models such as Gaussian naive Bayes.
#     In this case, the classifier’s overconfidence is caused by the redundant features which violate the naive Bayes assumption of feature-independence.
#     In the case of over-confident models, calibration of the probabilities of Gaussian naive Bayes with isotonic regression can help.

# Sigmoid calibration normally cannot help with over-confident classifiers, since sigmoid calibrtion's parametric form assumes a sigmoid rather than a transposed-sigmoid curve.
# The non-parametric isotonic calibration model, however, makes no such strong assumptions and can deal with either shape, provided that there is sufficient calibration data.
# In general, sigmoid calibration is preferable if the calibration curve is sigmoid and when there is few calibration data; while
# isotonic calibration is preferable for non- sigmoid calibration curves and in situations where many additional data can be used for calibration.

# For an example of classifier calibration when having more than 2 classes, see:
# http://scikit-learn.org/stable/modules/calibration.html

###############################################################################
### PROBABILITY CALIBRATION CURVES
# http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#example-calibration-plot-calibration-curve-py

# Two calibration methods: isotonic calibration and sigmoid calibration
# The calibration performance is evaluated with Brier score, reported in the legend (the smaller the better).

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD Style.

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cross_validation import train_test_split


### Create dataset of classification task with many redundant and few informative features
X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=10,
                                    random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99,
                                                    random_state=42)


#######################################
def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    ###
    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.draw()
#######################################

### PLOT
# Plot calibration curve for Gaussian Naive Bayes
plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)
plt.close()

# Plot calibration cuve for Linear SVC
plot_calibration_curve(LinearSVC(), "SVC", 2)
plt.close()


###############################################################################
### COMPARISON OF CALIBRATION OF CLASSIFIERS
# http://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD Style.

import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve

X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=2)

train_samples = 100  # Samples used for training the models

X_train = X[:train_samples]
X_test = X[train_samples:]
y_train = y[:train_samples]
y_test = y[train_samples:]

# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)


#######################################
# Plot calibration plots

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.draw()
plt.close()
###############################################################################

###############################################################################
######################## COMPARISON OF CLASSIFIERS ############################
###############################################################################
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

# Create a random classification data set with 2 features
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.draw()

plt.savefig(localPath+'Results/ClassifiersComparison.png')
plt.close('all')


###############################################################################
### COMPARING THE DECISION SURFACE OF DIFFERENT CLASSIFIERS ON IRIS DATA SET
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html

# This plot compares the decision surfaces learned
# by a decision tree classifier (first column),
# by a random forest classifier (second column),
# by an extra- trees classifier (third column) and
# by an AdaBoost classifier (fourth column).

# In the first row, the classifiers are built using the sepal width and the sepal length features only,
# on the second row using the petal length and sepal length only,
# and on the third row using the petal width and the petal length only.

import numpy as np
import matplotlib.pyplot as plt

from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
n_estimators = 30
plot_colors = "ryb"
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

# Load data
iris = load_iris()

plot_idx = 1

models = [DecisionTreeClassifier(max_depth=None),
          RandomForestClassifier(n_estimators=n_estimators),
          ExtraTreesClassifier(n_estimators=n_estimators),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             n_estimators=n_estimators)]

for pair in ([0, 1], [0, 2], [2, 3]):
    for model in models:
        # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Train
        clf = clone(model)
        clf = model.fit(X, y)

        scores = clf.score(X, y)

        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]
        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(len(model.estimators_))
        print( model_details + " with features", pair, "has a score of", scores )

        plt.subplot(3, 4, plot_idx)
        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a black outline
        xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),
                                             np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        for i, c in zip(xrange(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=c, label=iris.target_names[i],
                        cmap=cmap)

        plot_idx += 1  # move on to the next plot in sequence

plt.suptitle("Classifiers on feature subsets of the Iris dataset")
plt.axis("tight")

plt.draw()
###############################################################################


