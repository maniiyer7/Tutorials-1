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
# http://scikit-learn.org/stable/data_transforms.html

# Transformers are general class of methods in sci-kit that deal with preprocessing and feature transformation.
# These transformers fit within the following major groups:
# (a) clean (see Preprocessing data),
# (b) reduce (see Unsupervised dimensionality reduction),
# (c) expand (see Kernel Approximation)
# (d) generate (see Feature extraction) feature representations.

# These methods are represented by classes with fit method,
# which learns model parameters (e.g. mean and standard deviation for normalization) from a training set,
# and a transform method which applies this transformation model to unseen data.
# fit_transform may be more convenient and efficient for modelling and transforming the training data simultaneously.

# For instance in the case of PCA, 'fit' and 'transform' methods take care of what R calls 'scores'.
# While R directly gives the transformed features as the score attribute of PCA, the scikit implementation of
# PCA only internally stores the 'loadings'. When we transform a data set, the PCA fit object applies those
# loadings to the data set to come up with the new dimensions.

###############################################################################

###############################################################################
############################# MODEL EVALUATION ################################
###############################################################################
# http://scikit-learn.org/stable/modules/model_evaluation.html

###############################################################################
### CROSS_VALIDATION
# http://scikit-learn.org/stable/modules/cross_validation.html

import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
iris.data.shape, iris.target.shape


### Estimating model accuracy using 40% hold-out as test set as demonstration of the technique
# (but this is not scalable, since leaving out 40% of training is not good practice)
# sample a training set while holding out 40% of the data for testing (evaluating) our classifier:
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

X_train.shape, y_train.shape
X_test.shape, y_test.shape

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)


### Using k-fold cross-validation
clf = svm.SVC(kernel='linear', C=1)

# The simplest way to use cross-validation is to call the cross_val_score helper function on the estimator and the dataset.
# fitting a model and computing the score 5 consecutive times (with different splits each time)
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)

# The mean score and the 95% confidence interval of the score estimate are hence given by:
scores
scores.mean()
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


### Changing the scoring method
# By default, the score computed at each CV iteration is the score method of the estimator. It is possible to change this by using the scoring parameter:
from sklearn import metrics
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_weighted')

scores


### Changing k-fold strategies
# When the cv argument is an integer, cross_val_score uses the KFold or StratifiedKFold strategies by default,
# the latter being used if the estimator derives from ClassifierMixin.
# It is also possible to use other cross validation strategies by passing a cross validation iterator instead, for instance:
n_samples = iris.data.shape[0]
cv = cross_validation.ShuffleSplit(n_samples, n_iter=3, test_size=0.3, random_state=0)

cross_validation.cross_val_score(clf, iris.data, iris.target, cv=cv)

# More on cross-validation: http://scikit-learn.org/stable/modules/cross_validation.html
###############################################################################


###############################################################################
### MANUAL CROSS-VALIDATION
from sklearn import datasets, svm
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
# Fit on training and test on test dataset. Test dataset is the last 100 observations.
svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])

# k-fold cross-validation using numpy array_split()
import numpy as np
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
    # We use 'list' to copy, in order to 'pop' later on
    X_train = list(X_folds)
    X_test  = X_train.pop(k)
    X_train = np.concatenate(X_train)
    # Since we did not partition the dataset randomly (instead we used array_split), the order of x_train and y_train is the same.
    y_train = list(y_folds)
    y_test  = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)
###############################################################################


###############################################################################
### CROSS-VALIDATION USING Cross-validation generators (sklearn built-in methods)
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold
from sklearn import cross_validation


k_fold = cross_validation.KFold(n=6, n_folds=3)  # n: number of observations; n_folds: number of folds
for train_indices, test_indices in k_fold:
    print('Train: %s | test: %s' % (train_indices, test_indices))

# Or, another way to write the code:
kfold = cross_validation.KFold(len(X_digits), n_folds=3)
[svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
    for train, test in kfold]

# The following code is not run properly on Interactive Python.
# cross_validation.cross_val_score(svc, X_digits, y_digits, cv=kfold, n_jobs=-1)


### cross-validation generators:
KFold(n, k)  # Split it K folds, train on K-1 and then test on left-out
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold
StratifiedKFold(y, k)  # It preserves the class ratios / label distribution within each fold.
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html#sklearn.cross_validation.StratifiedKFold
LeaveOneOut(n)  # Leave one observation out
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.LeaveOneOut.html#sklearn.cross_validation.LeaveOneOut
LeaveOneLabelOut(labels)  # Takes a label array to group observations
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.LeaveOneLabelOut.html#sklearn.cross_validation.LeaveOneLabelOut


#######################################
### Grid-search
# fits an estimator on a parameter grid and chooses the parameters to maximize the cross-validation score.
from sklearn import svm
svc = svm.SVC(C=1, kernel='linear')
from sklearn.grid_search import GridSearchCV
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

Cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),
                    n_jobs=-1)
# clf.fit(X_digits[:1000], y_digits[:1000])
# GridSearchCV(cv=None,...
clf.best_score_

clf.best_estimator_.C


# Prediction performance on test set is not as good as on train set
clf.score(X_digits[1000:], y_digits[1000:])


#######################################
### EXAMPLE: USE RANDOM FORESTS AND GRID SEARCH TO FIND THE BEST FEATURES
# http://stackoverflow.com/questions/23174964/how-to-gridsearch-over-transform-arguments-within-a-pipeline-in-scikit-learn
#######################################


#######################################
from sklearn import svm
svc = svm.SVC(C=1, kernel='linear')
from sklearn.grid_search import GridSearchCV
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

Cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),
                    n_jobs=-1)
# clf.fit(X_digits[:1000], y_digits[:1000])  # creates error
# clf.best_score_
# clf.best_estimator_.C
# Prediction performance on test set is not as good as on train set
# clf.score(X_digits[1000:], y_digits[1000:])

# Nested cross-validation
# Two cross-validation loops are performed in parallel:
# one by the GridSearchCV estimator to set gamma and
# the other one by cross_val_score to measure the prediction performance of the estimator.
cross_validation.cross_val_score(clf, X_digits, y_digits)
#######################################


#######################################
### Set the cross-validated estimators by different methods
from sklearn import linear_model, datasets
lasso = linear_model.LassoCV()
diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target
lasso.fit(X_diabetes, y_diabetes)
LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
    max_iter=1000, n_alphas=100, n_jobs=1, normalize=False, positive=False,
    precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
    verbose=False)
# The estimator chose automatically its lambda:
lasso.alpha_

###############################################################################

###############################################################################
################# (1) REMOVING FEATURES WITH LOW VARIANCE #####################
###############################################################################

###############################################################################
### MODULE 3: MODEL SELECTION
# http://scikit-learn.org/stable/tutorial/
#   http://scikit-learn.org/stable/tutorial/statistical_inference/index.html
#       http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html

# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html



###############################################################################
### VarianceThreshold
# Simply removes features whose variance is lower than a threshold
from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)
###############################################################################

###############################################################################
#################### (2) UNIVARIATE FEATURE SELECTION #########################
###############################################################################


###############################################################################
### SelectKBest
# Select features according to the k highest scores
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
from sklearn import feature_selection
from sklearn import cross_validation

# score_finc is a function taking two arrays X and y, and returning a pair of arrays (scores, pvalues). See:
# http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# K is the number of top features to select.

# Load data and split into training and test data sets
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.33, random_state=42)

# Build the model
kbest = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=2)

# Fit and apply it to training data set
X_sel = kbest.fit_transform(X_train, y_train)

# Apply it to test data set
X_test_sel = kbest.transform(X_test)


#######################################
### chi2 test statistic
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
# We can use KBest method with chi2 as the score algorithm.

# Chi-squared stats of non-negative features for classification tasks.
# Compute chi-squared stats between each non-negative feature and class.
# chi-square test measures dependence between stochastic variables.
# Therefore, chi2 test can help remove features that do not move with the target classes; i.e., are irrelevant for classification.
# If you use sparse data (i.e. data represented as sparse matrices), only chi2 will deal with the data without making it dense

# Example: using chi2 test to the samples to retrieve only the two best features as follows
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

iris = load_iris()
X, y = iris.data, iris.target
X.shape

X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new.shape
#######################################



#######################################
### f_classif & f_regression
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html

# ANOVA F-value between labe/feature for classification tasks.
# Compute the ANOVA F-value for the provided sample.
# Univariate linear regression tests.

# Quick linear model for testing the effect of a single regressor, sequentially for many regressors.
# This is done in 3 steps:
#   1. The regressor of interest and the data are orthogonalized wrt constant regressors.
#   2. The cross correlation between data and regressors is computed.
#   3. It is converted to an F score then to a p-value.
#######################################




###############################################################################
### f_regression
# F-value between label/feature for regression tasks.



###############################################################################
### SelectPercentile
# Select features based on percentile of the highest scores.




###############################################################################
### SelectFpr
# Select features based on a false positive rate test.


###############################################################################
### SelectFdr
# Select features based on an estimated false discovery rate.


###############################################################################
### SelectFwe
# Select features based on family-wise error rate.
# Select the p-values corresponding to Family-wise error rate.
# The function SelectFwe takes two arrays X and y, and returning a pair of arrays (scores, pvalues).


###############################################################################
### GenericUnivariateSelect
# Univariate feature selector with configurable mode.


###############################################################################
##################### (3) RECURSIVE FEATURE ELIMINATION #######################
###############################################################################


###############################################################################
########################## (4) L1 FEATURE SELECTION ###########################
###############################################################################
# http://scikit-learn.org/stable/modules/feature_selection.html#l1-based-feature-selection
# Linear models with L1-regularized terms have sparse features, meaning several coefficients will be zero.
# These models include the LASSO, SVM, and Logistic Regression.
# In sci-kit, these estimators provide a Transform() function that removes features whose coefficient was 0 from the feature space.

# With SVMs and logistic-regression, the parameter C controls the sparsity: the smaller C the fewer features selected.
# With Lasso, the higher the alpha parameter, the fewer features selected.
# There is no general rule to select an alpha parameter for recovery of non-zero coefficients.
# It can by set by cross-validation (LassoCV or LassoLarsCV), though this may lead to under-penalized models:
# including a small number of non-relevant variables is not detrimental to prediction score.
# BIC (LassoLarsIC) tends, on the opposite, to set high values of alpha.
# More on selecting alpha: http://dsp.rice.edu/sites/dsp.rice.edu/files/cs/baraniukCSlecture07.pdf


#######################################
# Example: L1-reguralized models such as LASSO, Logistic Regression, and Linear SVC will result in sparse features.
# These models have a transform() function that returns the list of feature subsets chosen by the model.
# Ie, a model such as Linear SVC can be used for feature selection.
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
X.shape
X_new = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(X, y)
X_new.shape
#######################################

###############################################################################
### RANDOMIZED SPARCE MODELS
# The limitation of L1-based sparse models is that faced with a group of very correlated features,
# they will select only one.
# To mitigate this problem, it is possible to use randomization techniques,
# reestimating the sparse model many times perturbing the design matrix or sub-sampling data and
# counting how many times a given regressor is selected.

# The functions RandomizedLasso() and RandomizedLogisticRegression() implement randomized L1 reguralization for regression and two-class classification settings.

#######################################
### RANDOMIZED LASSO
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RandomizedLasso.html
# Randomized Lasso works by resampling the train data and computing a Lasso on each resampling.
# In short, the features selected more often are good features.
# It is also known as stability selection.

# When the regressors are correlated (multi-collinearity), LASSO randomly selects a subset of features, which is not good.
# Randomized LASSO solves that problem.

# Parameter:
# alpha : float, ‘aic’, or ‘bic’, optional

from sklearn.linear_model import RandomizedLasso
randomized_lasso = RandomizedLasso()
#######################################

###############################################################################

###############################################################################
##################### (5) TREE-BASED FEATURE SELECTION #########################
###############################################################################

###############################################################################
# see the sklearn.tree module and forest of trees in the sklearn.ensemble module
### DECISION TREE CLASSIFIER FOR FEATURE SELECTION
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
X.shape

clf = ExtraTreesClassifier()
X_new = clf.fit(X, y).transform(X)
clf.feature_importances_

X_new.shape
#######################################

#######################################
### FOREST OF TREES FOR FEATURE SELECTION
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

### Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

### Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


### Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(10), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(10), indices)
plt.xlim([-1, 10])
plt.draw()
plt.close('all')
#######################################


#######################################
### ESTIMATING PIXEL IMPORTANCES USING A PARALLEL FOREST OF TREES
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances_faces.html#example-ensemble-plot-forest-importances-faces-py
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesClassifier

# Number of cores to use to perform parallel fitting of the forest model
n_jobs = 1

# Load the faces dataset
data = fetch_olivetti_faces()
X = data.images.reshape((len(data.images), -1))
y = data.target

mask = y < 5  # Limit to 5 classes (ie, the softest points)
X = X[mask]
y = y[mask]

# Build a forest and compute the pixel importances
print("Fitting ExtraTreesClassifier on faces data with %d cores..." % n_jobs)
t0 = time()
forest = ExtraTreesClassifier(n_estimators=1000,
                              max_features=128,
                              n_jobs=n_jobs,
                              random_state=0)

forest.fit(X, y)
print("done in %0.3fs" % (time() - t0))
importances = forest.feature_importances_
importances = importances.reshape(data.images[0].shape)

# Plot pixel importances
plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances with forests of trees")
plt.draw()
plt.close('all')
#######################################

# These are 400 images, each 64x64 pixles. Each pixel is represented by a single value from 1-10, which is the strength of the signal (exposure).
# Each of the 400 images show the same person, from different angles.
# Ie, each pixel is a feature. Each picture is an observation, and has a measured value across all features (all pixels).
# The goal is to find the most important features (pixles) that distinguish the image.

plt.matshow(data.images[4], cmap=plt.cm.hot)
plt.title("Pixel importances with forests of trees")
plt.draw()
plt.close('all')

###############################################################################
############### (6) FEATURE SELECTION AS PART OF A PIPELINE ###################
###############################################################################

clf = Pipeline([
  ('feature_selection', LinearSVC(penalty="l1")),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)




###############################################################################
################## CONCATENATING FEATURE SELECTION METHODS ####################
###############################################################################
# http://scikit-learn.org/stable/auto_examples/feature_stacker.html
# This example shows how to use FeatureUnion to combine features obtained by PCA and univariate selection.
# Combining features using this transformer has the benefit that it allows cross validation and grid searches over the whole process.
# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3 clause

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

iris = load_iris()

X, y = iris.data, iris.target

# This dataset is way to high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:
combined_features = FeatureUnion([("pca", pca),
                                  ("univ_select", selection)])

# Use combined features to transform dataset:
# Note: remember that 'fit' method learns the model reduction algo,
# and 'transform' applies it to trainig (or test) data to transform the features into new dimensions.
X_features = combined_features.fit(X, y).transform(X)

# Build the model estimator object
svm = SVC(kernel="linear")

# Do grid search over k, n_components and C:
pipeline = Pipeline([("features", combined_features),
                     ("svm", svm)])

# Generate a parameter grid for searching for the best parameter for the model.
# Pay attention to naming conventions.
param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])

# Do a cross-validation grid search
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print(grid_search.best_estimator_)
###############################################################################


###############################################################################
### Pipeline ANOVA and SVM
print(__doc__)

from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

# import some data to play with
X, y = samples_generator.make_classification(
    n_features=20, n_informative=3, n_redundant=0, n_classes=4,
    n_clusters_per_class=2)

# ANOVA SVM-C
# 1) anova filter, take 3 best ranked features
anova_filter = SelectKBest(f_regression, k=3)
# 2) svm
clf = svm.SVC(kernel='linear')

anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X, y)
anova_svm.predict(X)
###############################################################################




###############################################################################
### EXAMPLE: FEATURE SELECTION AND SVM
# http://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

### import some data to play with
# The iris dataset
iris = datasets.load_iris()

# Add some noisy features that are not correlated with target
E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))

# Add the noisy data to the informative features
X = np.hstack((iris.data, E))
y = iris.target

X_indices = np.arange(X.shape[-1])


### Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()


### Compare to the weights of an SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()


### First select features based on the percentile test above, and then fit the SVM
clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(X), y)

svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()


### PLOT
plt.figure(1)
plt.clf()
# Univariate feature selection score
plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='g')
# SVM weights without feature selection
plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight', color='r')
# SVM weights after feature selection
plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
        width=.2, label='SVM weights after selection', color='b')

plt.title("Comparing feature selection")
plt.xlabel('Feature number')
plt.yticks(())
plt.axis('tight')
plt.legend(loc='upper right')
plt.draw()
plt.close()
###############################################################################


###############################################################################
### EXAMPLE: SVM WITH FEATURE SELECTION
# http://scikit-learn.org/stable/auto_examples/svm/plot_svm_anova.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, feature_selection, cross_validation
from sklearn.pipeline import Pipeline


### PREP DATA
# Import some data to play with
digits = datasets.load_digits()
y = digits.target

# Throw away data, to be in the curse of dimension settings
y = y[:200]
X = digits.data[:200]
n_samples = len(y)
X = X.reshape((n_samples, -1))  # -1 means the second dimension will be calculated based on the other dimension. I.e., it is inferred.

# add 200 non-informative features
X = np.hstack((X, 2 * np.random.random((n_samples, 200))))


### Create a feature-selection transform and an instance of SVM that we
# combine together to have an full-blown estimator
# the method is 'SelectPercentile' and the metric is 'f_classif'
transform = feature_selection.SelectPercentile(feature_selection.f_classif)

clf = Pipeline([('anova', transform),
                ('svc', svm.SVC(C=1.0))])


### Plot the cross-validation score as a function of percentile of features
score_means = list()
score_stds = list()
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    # Compute cross-validation score using all CPUs
    this_scores = cross_validation.cross_val_score(clf, X, y, n_jobs=1)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())


### Plot an error-bar plot of cross validation score
plt.errorbar(percentiles, score_means, np.array(score_stds))

plt.title(
    'Performance of the SVM-Anova varying the percentile of features selected')
plt.xlabel('Percentile')
plt.ylabel('Prediction rate')

plt.axis('tight')
plt.draw()
plt.close()
###############################################################################


###############################################################################
### UNIVARIATE FEATURE SELECTION VS FEATURE AGGLOMERATION
# http://scikit-learn.org/stable/auto_examples/cluster/plot_feature_agglomeration_vs_univariate_selection.html#example-cluster-plot-feature-agglomeration-vs-univariate-selection-py

# The following script compares 2 dimensionality reduction strategies:
#   1. univariate feature selection with Anova
#   2. feature agglomeration with Ward hierarchical clustering

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, ndimage

from sklearn.feature_extraction.image import grid_to_graph
from sklearn import feature_selection
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.externals.joblib import Memory
from sklearn.cross_validation import KFold

#######################################
# Generate data
n_samples = 200
size = 40  # image size
roi_size = 15
snr = 5.
np.random.seed(0)
mask = np.ones([size, size], dtype=np.bool)

coef = np.zeros((size, size))
coef[0:roi_size, 0:roi_size] = -1.
coef[-roi_size:, -roi_size:] = 1.

# Form training data and add smooth it
X = np.random.randn(n_samples, size ** 2)
for x in X:  # smooth data
    x[:] = ndimage.gaussian_filter(x.reshape(size, size), sigma=1.0).ravel()
X -= X.mean(axis=0)
X /= X.std(axis=0)

# Form target value and add noise to it
y = np.dot(X, coef.ravel())
noise = np.random.randn(y.shape[0])
noise_coef = (linalg.norm(y, 2) / np.exp(snr / 20.)) / linalg.norm(noise, 2)
y += noise_coef * noise  # add noise

#######################################
### Compute the coefs of a Bayesian Ridge with GridSearch
# cross-validation generator for model selection
cv = KFold(len(y), 2)
ridge = BayesianRidge()

cachedir = tempfile.mkdtemp()
mem = Memory(cachedir=cachedir, verbose=1)


#######################################
### Ward agglomeration followed by BayesianRidge
# Create a Ward agglomeration
connectivity = grid_to_graph(n_x=size, n_y=size)
ward = FeatureAgglomeration(n_clusters=10, connectivity=connectivity,
                            memory=mem)

# Create a pipeline of Ward agglomeration followed by BayesianRidge
clf = Pipeline([('ward', ward),
                ('ridge', ridge)])


### Select the optimal number of parcels with grid search
clf = GridSearchCV(clf, {'ward__n_clusters': [10, 20, 30]}, n_jobs=1, cv=cv)
clf.fit(X, y)  # set the best parameters
coef_ = clf.best_estimator_.steps[-1][1].coef_
coef_ = clf.best_estimator_.steps[0][1].inverse_transform(coef_)
coef_agglomeration_ = coef_.reshape(size, size)


#######################################
### Anova univariate feature selection followed by BayesianRidge
f_regression = mem.cache(feature_selection.f_regression)  # caching function
anova = feature_selection.SelectPercentile(f_regression)
clf = Pipeline([('anova', anova),
                ('ridge', ridge)])

# Select the optimal percentage of features with grid search
clf = GridSearchCV(clf, {'anova__percentile': [5, 10, 20]}, cv=cv)
clf.fit(X, y)  # set the best parameters
coef_ = clf.best_estimator_.steps[-1][1].coef_
coef_ = clf.best_estimator_.steps[0][1].inverse_transform(coef_)
coef_selection_ = coef_.reshape(size, size)


#######################################
# Inverse the transformation to plot the results on an image
plt.close('all')
plt.figure(figsize=(7.3, 2.7))

plt.subplot(1, 3, 1)
plt.imshow(coef, interpolation="nearest", cmap=plt.cm.RdBu_r)
plt.title("True weights")

plt.subplot(1, 3, 2)
plt.imshow(coef_selection_, interpolation="nearest", cmap=plt.cm.RdBu_r)
plt.title("Feature Selection")

plt.subplot(1, 3, 3)
plt.imshow(coef_agglomeration_, interpolation="nearest", cmap=plt.cm.RdBu_r)
plt.title("Feature Agglomeration")

plt.subplots_adjust(0.04, 0.0, 0.98, 0.94, 0.16, 0.26)
plt.draw()
plt.close()

# Attempt to remove the temporary cachedir, but don't worry if it fails
shutil.rmtree(cachedir, ignore_errors=True)
###############################################################################


###############################################################################
### PIPELINE ANOVA SVM
# http://scikit-learn.org/stable/auto_examples/feature_selection/feature_selection_pipeline.html#example-feature-selection-feature-selection-pipeline-py
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

# import some data to play with
X, y = samples_generator.make_classification(
    n_features=20, n_informative=3, n_redundant=0, n_classes=4,
    n_clusters_per_class=2)

# ANOVA SVM-C
# 1) anova filter, take 3 best ranked features
anova_filter = SelectKBest(f_regression, k=3)
# 2) svm
clf = svm.SVC(kernel='linear')

anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X, y)
anova_svm.predict(X)
###############################################################################


###############################################################################
### FEATURE SELECTION FOR SMALL SPARSE MODELS
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_recovery.html#example-linear-model-plot-sparse-recovery-py
# When the true model is sparse, i.e. if a small fraction of the features are relevant,
# sparse linear models can outperform standard statistical models.
# The ability of L1-based approach to identify the relevant variables depends on
#   the sparsity of the ground truth,
#   the number of samples,
#   the number of features,
#   the conditioning of the design matrix on the signal subspace,
#   the amount of noise, and
#   the absolute value of the smallest non-zero coefficient.

# For a well-conditioned design matrix (small mutual incoherence)
# we are exactly in compressive sensing conditions (i.i.d Gaussian sensing matrix),
# and L1-recovery with the Lasso performs very well.
# For an ill-conditioned matrix (high mutual incoherence),
# regressors are very correlated, and the Lasso randomly selects one.
# However, randomized-Lasso can recover the ground truth well.

# There are two steps to the solution below:
# 1. we first vary the alpha parameter setting the sparsity of the estimated model and look at the stability scores of the randomized Lasso.
# 2. In a second time, we set alpha and compare the performance of different feature selection methods, using the area under curve (AUC) of the precision-recall.


# Author: Alexandre Gramfort and Gael Varoquaux
# License: BSD 3 clause

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

from sklearn.linear_model import (RandomizedLasso, lasso_stability_path, LassoLarsCV)
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.extmath import pinvh
from sklearn.utils import ConvergenceWarning


def mutual_incoherence(X_relevant, X_irelevant):
    """Mutual incoherence, as defined by formula (26a) of [Wainwright2006].
    """
    projector = np.dot(np.dot(X_irelevant.T, X_relevant),
                       pinvh(np.dot(X_relevant.T, X_relevant)))
    return np.max(np.abs(projector).sum(axis=1))


for conditioning in (1, 1e-4):
    ### Simulate regression data with a correlated design
    n_features = 501
    n_relevant_features = 3
    noise_level = .2
    coef_min = .2

    # The Donoho-Tanner phase transition is around n_samples=25: below we
    # will completely fail to recover in the well-conditioned case
    n_samples = 25
    block_size = n_relevant_features

    rng = np.random.RandomState(42)

    # The coefficients of our model
    coef = np.zeros(n_features)
    coef[:n_relevant_features] = coef_min + rng.rand(n_relevant_features)

    # The correlation of our design: variables correlated by blocs of 3
    corr = np.zeros((n_features, n_features))
    for i in range(0, n_features, block_size):
        corr[i:i + block_size, i:i + block_size] = 1 - conditioning
    corr.flat[::n_features + 1] = 1
    corr = linalg.cholesky(corr)

    # Our design
    X = rng.normal(size=(n_samples, n_features))
    X = np.dot(X, corr)

    # Keep [Wainwright2006] (26c) constant
    X[:n_relevant_features] /= np.abs(
        linalg.svdvals(X[:n_relevant_features])).max()
    X = StandardScaler().fit_transform(X.copy())

    # The output variable
    y = np.dot(X, coef)
    y /= np.std(y)

    # We scale the added noise as a function of the average correlation
    # between the design and the output variable
    y += noise_level * rng.normal(size=n_samples)
    mi = mutual_incoherence(X[:, :n_relevant_features],
                            X[:, n_relevant_features:])

    ###################################
    ### Plot stability selection path, using a high eps for early stopping
    # of the path, to save computation time
    alpha_grid, scores_path = lasso_stability_path(X, y, random_state=42,
                                                   eps=0.05)

    plt.figure()
    # We plot the path as a function of alpha/alpha_max to the power 1/3: the
    # power 1/3 scales the path less brutally than the log, and enables to
    # see the progression along the path
    hg = plt.plot(alpha_grid[1:] ** .333, scores_path[coef != 0].T[1:], 'r')
    hb = plt.plot(alpha_grid[1:] ** .333, scores_path[coef == 0].T[1:], 'k')
    ymin, ymax = plt.ylim()
    plt.xlabel(r'$(\alpha / \alpha_{max})^{1/3}$')
    plt.ylabel('Stability score: proportion of times selected')
    plt.title('Stability Scores Path - Mutual incoherence: %.1f' % mi)
    plt.axis('tight')
    plt.legend((hg[0], hb[0]), ('relevant features', 'irrelevant features'),
               loc='best')
    plt.draw()
    plt.close()


    ####################################
    ### Plot the estimated stability scores for a given alpha

    # Use 6-fold cross-validation rather than the default 3-fold: it leads to
    # a better choice of alpha:
    # Stop the user warnings outputs- they are not necessary for the example
    # as it is specifically set up to be challenging.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', ConvergenceWarning)
        lars_cv = LassoLarsCV(cv=6).fit(X, y)

    # Run the RandomizedLasso: we use a paths going down to .1*alpha_max
    # to avoid exploring the regime in which very noisy variables enter
    # the model
    alphas = np.linspace(lars_cv.alphas_[0], .1 * lars_cv.alphas_[0], 6)
    clf = RandomizedLasso(alpha=alphas, random_state=42).fit(X, y)
    trees = ExtraTreesRegressor(100).fit(X, y)
    # Compare with F-score
    F, _ = f_regression(X, y)

    plt.figure()
    for name, score in [('F-test', F),
                        ('Stability selection', clf.scores_),
                        ('Lasso coefs', np.abs(lars_cv.coef_)),
                        ('Trees', trees.feature_importances_),
                        ]:
        precision, recall, thresholds = precision_recall_curve(coef != 0,
                                                               score)
        plt.semilogy(np.maximum(score / np.max(score), 1e-4),
                     label="%s. AUC: %.3f" % (name, auc(recall, precision)))

    plt.plot(np.where(coef != 0)[0], [2e-4] * n_relevant_features, 'mo',
             label="Ground truth")
    plt.xlabel("Features")
    plt.ylabel("Score")
    # Plot only the 100 first coefficients
    plt.xlim(0, 100)
    plt.legend(loc='best')
    plt.title('Feature selection scores - Mutual incoherence: %.1f'
              % mi)

    plt.draw()

    plt.close()

###############################################################################

###############################################################################
### CHAINING A SVM AND LOGISTIC REGRESSION
# PCA does an unsupervised dimensionality reduction, while the logistic regression does the prediction.
# Here we use a GridSearchCV to set the dimensionality of the PCA.
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


### Create pipeline
logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca),
                       ('logistic', logistic)])


### Load data
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target


### PREDICT (FIT THE PIPELINE)
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))

estimator.fit(X_digits, y_digits)

# See the number of PCA dimensions chosens
estimator.best_estimator_.named_steps['pca'].n_components


#######################################
### PLOT RESULTS
# Plot PCA Spectrum (percent variance explained)
pca.fit(X_digits)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.draw()


plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.draw()
plt.close()
###############################################################################
