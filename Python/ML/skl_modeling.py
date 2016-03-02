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
pd.set_option('display.width', 5000)
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



##############################################################################
##################### REGRESSION (linear_model module) #######################
##############################################################################

###############################################################################
### MODULE 2: SUPERVISED LEARNING
# http://scikit-learn.org/stable/tutorial/
#   http://scikit-learn.org/stable/tutorial/statistical_inference/index.html
#       http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html

### linear_model module
# More examples: http://scikit-learn.org/stable/modules/linear_model.html
from sklearn import linear_model

# the linear_model module offers much more than just the OLS model.
# See the list of available models (including ridge regression, logistic regression, lasso, bayes)
dir(linear_model)

##############################################################################

##############################################################################
############################### OLS REGRESSION ###############################
##############################################################################

##############################################################################
### LINEAR REGRESSION (OLS)
# Load Diabetes dataset
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

# Fit the model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

# Estimated coefficients of the regression model
print(regr.coef_)

# The mean square error
np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)

# R2: Explained variance score
regr.score(diabetes_X_test, diabetes_y_test)
##############################################################################


##############################################################################
### MULTIVARIATE LINEAR REGRESSION (regression with multiple features)
## Load the diabetes dataset
diabetes = datasets.load_diabetes()

## Use only one feature
# First, expand the second dimension into a set of lists. I.e., instead of a 442x10 matrix, we now have a list of 442 lists, each with a 1x10 dimension.
diabetes_X = diabetes.data[:, np.newaxis]
# Then, choose the second element (corresponding to the second feature) from each records.
diabetes_X_temp = diabetes_X[:, :, 2]

# NOTE ON ABOVE CODE: The feature matrix in linearmodel estimators needs to be structured in a certain way. It needs to be a list of lists, when each sub-list is the collection of all features for each observation.
# So for a training set with N observations and k features, we need to have a list with the shape (N,k), which consists of a list, with N sub-lists where each sub-list contains k values (one per feature).
# The same order applies even when we only have one feature to work with. So, for a simple single-dimension training set (y~x model), we need a (N,1) shape, not a (N,) shape.
# If we had chosen diabetes_X = diabetes.data[:,2] below, we would have gotten a (N,) array; however, by adding one dimension to the dataset using np.newaxis, we then used [:, :, 2] which gave us a (N,1) array.


## Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-20]
diabetes_X_test = diabetes_X_temp[-20:]

## Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

## Create linear regression object
regr = linear_model.LinearRegression()

## Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

## The coefficients
print('Coefficients: \n', regr.coef_)

# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

## Plot outputs
plt.figure()
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.draw()
plt.close()
##############################################################################

##############################################################################
### MULTIVARIATE LINEAR REGRESSION: more than one regressor
# We need one list per observation. Each list element corresponds to one feature.
X = [[0., 0., 3], [1., 1., 3], [2., 2., 3], [3., 3., 3]]
Y = [0., 1., 2., 3.]
clf = linear_model.LinearRegression()
clf.fit(X, Y)
clf.predict ([[1, 0.,2]])
##############################################################################

##############################################################################
############################# RIDGE REGRESSION ###############################
##############################################################################

##############################################################################
### SHRINKAGE (RIDGE REGRESSION)
# If there are few data points per dimension, noise in the observations induces high variance.
# Solution: shrink the regression coefficients to zero.
# Ridge Regression does that by introducing a penalty term to the objective function.
# The penalty term is the L2 norm of the coefficients array. So, Ridge Regression minimizes the following objective function:
#     min { L2(error) + alpha*L2(coefficients) }
# The larger the ridge alpha parameter, the higher the bias and the lower the variance.
# The bias introduced by the ridge regression is called a Regularization.

# Ridge Regression theory:
# http://scott.fortmann-roe.com/docs/BiasVariance.html
# http://cbcl.mit.edu/projects/cbcl/publications/ps/MIT-CSAIL-TR-2007-025.pdf
# http://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf

# More on Ridge Regression:
# http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression

#######################################
### RIDGE REGRESSION VS OLS: COMPARE MODEL VARIANCE
# This example shows how Ridge Regression can reduce the model variance (while allowing some bias) in the face of noisy data.

# Generate the example data set
X = np.c_[ .5, 1].T
y = [.5, 1]
test = np.c_[ 0, 2].T


### Plot un-regularized regression (OLS)
regr = linear_model.LinearRegression()
plt.figure(figsize=[10,10])
ax1 = plt.subplot2grid((4,1), (0,0))  # generate the grid

np.random.seed(0)
for _ in range(6):
    this_X = .1 * np.random.normal(size=(2, 1)) + X
    regr.fit(this_X, y)
    ax1.plot(test, regr.predict(test))
    ax1.set_title('OLS Regression')
    ax1.scatter(this_X, y, s=20)
plt.draw()


### Plot regularized regression (Ridge)
# Choose alpha to minimize left out error
alphas = [0.1, 0.5, 1]

for i, alpha in enumerate(alphas):
    ax = plt.subplot2grid((4,1), (i+1,0))  # generate the grid
    ax.set_title('Ridge Regression with alpha='+str(alpha))
    ridge = linear_model.Ridge(alpha=alpha)
    np.random.seed(0)
    for _ in range(6):
        this_X = .1 * np.random.normal(size=(2, 1)) + X
        ridge.fit(this_X, y)
        ax.plot(test, ridge.predict(test))
        ax.scatter(this_X, y, s=20)

plt.draw()
plt.close('all')
#######################################

#######################################
### RIDGE REGRESSION: HOW TO CHOOSE PARAMETER ALPHA

### (option a) choose alpha based on its performance on a test set
# alpha is the regularization parameter. The larger the alpha, the higher the regularization.
alphas = np.logspace(-4, -1, 6)  # Numbers spaced evenly on log scale. Parameters are: (start, stop, num=50, endpoint=True, base=10.0, dtype=None)
ridge = linear_model.Ridge()

scores = [ridge.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train).
              score(diabetes_X_test, diabetes_y_test)
          for alpha in alphas]

# Plot the score against alpha values to find the alpha with highest score
plt.figure()
plt.scatter(alphas, scores)
plt.xlabel('Regularization alpha')
plt.ylabel('Model Score')
plt.draw()
plt.close('all')


### (option b) Use cross-validation to choose the optimal alpha value
clf = linear_model.RidgeCV(alphas=alphas)
clf.fit(diabetes_X_train, diabetes_y_train)
linear_model.RidgeCV(alphas=alphas, cv=None, fit_intercept=True, scoring=None, normalize=False)
clf.alpha_

# Note that in this case, CV (in option b) did not choose the same alpha that we observed having the highest score (option a), because
# in 'option a' we calculated the score using the test set above, while 'option b' CV calculates the score using the left-out portion of the trainig set (we did not even provide the test set to RidgeCV function)
# Even if we use score(diabetes_X_train, diabetes_y_train) in the above function to calculate scores based on training set, the results will not be the same, because
# CV is using a left-out subset of the training set, while score(diabetes_X_training, diabetes_y_training) uses the entire training set.
# So, we cannot compare the results of the above two model selection techniques.
# Ideally we would like to use the second method; ie, use RidgeCV() to fit the model. Then, we would use ridge.score(diabetes_X_test, diabetes_y_test) to calculate and report the model performance.
#######################################


#######################################
### EXAMPLE: THE EFFECT OF RIDGE PARAMETER ALPHA ON MODEL COEFFICIENTS
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html#example-linear-model-plot-ridge-path-py
# Author: Fabian Pedregosa -- <fabian.pedregosa@inria.fr>

# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)


### Compute paths
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_)


### Display results
ax = plt.gca()  # we need the current axis to be able to specify color cycle (see next line)
ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

ax.plot(alphas, coefs)

ax.set_xscale('log')
# ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis

plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')

plt.draw()
plt.close('all')
#######################################


#######################################
### FITTING POLYNOMIAL DATA USING RIDGE REGRESSION AND PSEUDO FEATURES (FEATURE AUGMENTATION)
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html

# We can fit use linear models to fit non-linear data.
# For example, if y has a polynomial relationship with x, we can introduce non-linear transformations of features (ie, augment the feature space)
# and then fit a linear model to the augmented feature space.

# The resulting feature space is called the Vandermonde matrix, which is n_samples x n_degree+1 and has the following form:
# [[1, x_1, x_1 ** 2, x_1 ** 3, ...],
# [1, x_2, x_2 ** 2, x_2 ** 3, ...], ...]

# Vandermonde matrix is a matrix of pseudo features (the points raised to some power).
# The matrix is akin to (but different from) the matrix induced by a polynomial kernel.
# Kernel methods extend this idea and can induce very high (even infinite) dimensional feature spaces.


#  Author: Mathieu Blondel
#         Jake Vanderplas
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ ground-truth function to be approximated by polynomial interpolation"""
    return x * np.sin(x)


# generate points used to plot
x_plot = np.linspace(0, 10, 100)

# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)  # the function that will be approcimated by polynimial interpolation

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

plt.plot(x_plot, f(x_plot), label="ground truth")
plt.scatter(x, y, label="training points")

for degree in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, label="degree %d" % degree)

plt.legend(loc='lower left')

plt.draw()
plt.close('all')

#######################################
### KERNEL RIDGE REGRESSION
# http://scikit-learn.org/stable/modules/kernel_ridge.html
# Kernle Ridge Regression learns a linear function in the space induced by the respective kernel and the data.
# The form of the model learned by KernelRidge is identical to support vector regression (SVR).
# However, different loss functions are used:
# KRR uses squared error loss while support vector regression uses \epsilon-insensitive loss, both combined with l2 regularization.
# In contrast to SVR, fitting KernelRidge can be done in closed-form and is typically faster for medium-sized datasets.
# On the other hand, the KRR learned model is non-sparse and thus slower than SVR, which learns a sparse model for \epsilon > 0, at prediction-time.
# In short, model fitting is faster with KRR, while prediction is faster with SVR because of sparsity of the model.
#######################################
##############################################################################

##############################################################################
################################# LASSO ######################################
##############################################################################

##############################################################################
### LASSO (least absolute shrinkage and selection operator)
# Unlike Ridge Regression, LASSO can set some coefficients to zero.
# Therefore, LASSO is useful in model selection.

# LASSO adds a penalty term with L1 norm of the coefficients vector.
# The objective function to minimize for LASSO is:
# min { (1/2n)*L2(error) + alpha*L1(coeffs) }
# where n is the sample size.

#######################################
# NOTE: the Lasso object in scikit-learn solves the lasso regression problem
# using a coordinate descent method, that is efficient on large datasets.
# scikit-learn also provides the LassoLars object using the LARS algorthm,
# which is very efficient for problems in which the weight vector estimated
# is very sparse (i.e. problems with very few observations).
# See Least Angle Regression and LARS:
# http://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression
#######################################

##############################################################################
### SELECTING PARAMETER ALPHA FOR LASSO
# If alpha is chosen too small, non-relevant variables enter the model.
# On the opposite, if alpha is selected too large, the Lasso is equivalent to stepwise regression,
# and thus brings no advantage over a univariate F-test.

#######################################
# (option a) USE THE MSE ON TEST SET TO FIND THE OPTIMAL REGULARIZATION CONSTANT (alpha)
alphas = np.logspace(-10, -2, n_alphas)

lasso = linear_model.Lasso()

scores = [lasso.set_params(alpha=alpha
            ).fit(diabetes_X_train, diabetes_y_train
            ).score(diabetes_X_test, diabetes_y_test)
                                for alpha in alphas]

best_alpha = alphas[scores.index(max(scores))]
lasso.alpha = best_alpha
lasso.fit(diabetes_X_train, diabetes_y_train)

print(regr.coef_)

# Plot
plt.figure(figsize=[8,6])
ax = plt.gca()
ax.set_xscale('log')
ax.plot(alphas, scores)
plt.draw()
plt.close('all')

# Note that the plot of the scores is a direct line (when we remove the log-scale from x axis and plot it in linear x space).
# I.e., the plot suggests a linear relationship between MSE (the score) and alpha.
# That is expected. The objective function of LASSO is mean squared error (divided by 2) plus a linear regularization term, which is a multiply of alpha.
# Therefore, the score and the objective function are only different by alpha*L1(coeffs).
# As we change alpha, the MSE will change linearly, as long as the model structure does not change.
# When we plot it in logspace, we see that the MSE will only drop when the alpha gets closer to 1.
# The logarithmic drop is MSE in the logspace plot mirrors the logarithmic increase in x, so it is not an indication of logarithmic relationship between alpha and MSE.
#######################################

#######################################
# (option b) USE THE CROSS VALIDATION ON TRAINING SET TO FIND THE OPTIMAL REGULARIZATION CONSTANT (alpha)
# We can either perform the cross-validation manually (create a grid of alphas; find MSE for each alpha value; pick the alpha value with the lowest MSE)
# Or, we can use the off-the-shelf function LassCV() to give us the best alpha based on cross-validation.
# clf = linear_model.LassoCV()

from sklearn import cross_validation, datasets, linear_model

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]

lasso = linear_model.Lasso()
alphas = np.logspace(-4, -.5, 30)

scores = list()
scores_std = list()

# Find the cross-validated MSE value for different values of alpha
for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_validation.cross_val_score(lasso, X, y, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

plt.figure(figsize=(8, 6))
plt.semilogx(alphas, scores)  # log scale for x axis. Or I could just use ax.set_xscale('log') and use regular plot() commands from that point on
# plot error lines showing +/- std errors of the scores
plt.semilogx(alphas, np.array(scores) + np.array(scores_std) / np.sqrt(len(X)), 'b--')
plt.semilogx(alphas, np.array(scores) - np.array(scores_std) / np.sqrt(len(X)), 'b--')
plt.ylabel('CV score')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.draw()
plt.close('all')

# The other way to plot the figure above (note the use of set_xscale('log') )
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.set_xscale('log')
ax.plot(alphas, scores)
ax.plot(alphas, np.array(scores) + np.array(scores_std) / np.sqrt(len(X)), 'b--')
ax.plot(alphas, np.array(scores) - np.array(scores_std) / np.sqrt(len(X)), 'b--')
plt.ylabel('CV score')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.draw()


### How much can you trust the selection of alpha?

# To answer this question we use the LassoCV object that sets its alpha
# parameter automatically from the data by internal cross-validation (i.e. it
# performs cross-validation on the training data it receives).
# We use external cross-validation to see how much the automatically obtained
# alphas differ across different cross-validation folds.

lasso_cv = linear_model.LassoCV(alphas=alphas)
k_fold = cross_validation.KFold(len(X), 3)

print("Alpha parameters maximising the generalization score on different subsets of the data:")
for k, (train, test) in enumerate(k_fold):
    lasso_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
# We cannot trust the alpha chosen by CV very much since we obtained different alphas for different
# subsets of the data and moreover, the scores for these alphas differ quite substantially.

#######################################


#######################################
# (option c) USE AIC AND BIC TO FIND THE OPTIMAL ALPHA
# LassoLarsIC uses the Akaike information criterion (AIC) and the Bayes Information criterion (BIC) to find the optimal regularization constant alpha.
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html
# Author: Olivier Grisel, Gael Varoquaux, Alexandre Gramfort
# License: BSD 3 clause

import time

from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# add some bad features
rng = np.random.RandomState(42)
X = np.c_[X, rng.randn(X.shape[0], 14)]

# Normalize data as done by Lars to allow for comparison
# To normalize, divide each feature by its square root of sum of squares.
X /= np.sqrt(np.sum(X ** 2, axis=0))


### LassoLarsIC: least angle regression with BIC/AIC criterion
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html

# BIC
model_bic = LassoLarsIC(criterion='bic')
model_bic.fit(X, y)
alpha_bic_ = model_bic.alpha_

# AIC
model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X, y)
alpha_aic_ = model_aic.alpha_


# Plot the score (AIC and BIC) against alpha, and choose the alpha that minimizes AIC or BIC
def plot_ic_criterion(model, name, color, plt_ref):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt_ref.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt_ref.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt_ref.set_xlabel('-log(alpha)')
    plt_ref.set_ylabel('criterion')

plt.figure(figsize=[8,10])
ax1 = plt.subplot(3,1,1)
plot_ic_criterion(model_aic, 'AIC', 'b', ax1)
plot_ic_criterion(model_bic, 'BIC', 'r', ax1)
plt.legend()
plt.title('Information-criterion for model selection')
plt.draw()


### LassoCV: coordinate descent

## Compute paths
model = LassoCV(cv=20).fit(X, y)

## Display results
# model.alphas_ is the grid of alphas used for fitting.
m_log_alphas = -np.log10(model.alphas_)

# model.mse_path_ is the mean square error for the test set on each fold, varying alpha
# In this example, we used 100 different alphas, and performed a 20-fold cross validation.
# The matrix of mses is (100,20). There are 100 sub-lists, each with 20 elements.
# Each list is for one value of alpha. Each of the 20 elements in each list is the mse of that alpha for different CV folds.
# This is why we average tje scores over the second dimension. We want to know average MSE for a particular alpha, averaged over 20 folds.
mses = model.mse_path_

ymin, ymax = 2300, 3800
xmin, xmax = -0.5, 3.5

ax2 = plt.subplot(3,1,2)
ax2.plot(m_log_alphas, mses, ':')
ax2.plot(m_log_alphas, mses.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
ax2.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent ')
plt.axis('tight')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.draw()


### LassoLarsCV: least angle regression

# Compute paths
model = LassoLarsCV(cv=20).fit(X, y)

# Display results
m_log_alphas = -np.log10(model.cv_alphas_)

ax3 = plt.subplot(3,1,3)
plt.plot(m_log_alphas, model.cv_mse_path_, ':')
plt.plot(m_log_alphas, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: Lars')
plt.axis('tight')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.draw()

plt.close('all')
#######################################
##############################################################################

#######################################
### COMPRESSIVE SENSING: FEATURE SELECTION FOR SPARSE SIGNALS
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_recovery.html

# For a well-conditioned design matrix (small mutual incoherence) we are exactly in
# compressive sensing conditions (i.i.d Gaussian sensing matrix),
# and L1-recovery with the Lasso performs very well.
# For an ill-conditioned matrix (high mutual incoherence),
# regressors are very correlated, and the Lasso randomly selects one.
# However, randomized-Lasso can recover the ground truth well.

# Author: Alexandre Gramfort and Gael Varoquaux
# License: BSD 3 clause

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

from sklearn.linear_model import (RandomizedLasso, lasso_stability_path,
                                  LassoLarsCV)
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
    ####################################
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

    ####################################
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

    #####################################
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
#######################################

# More on LASSO: http://scikit-learn.org/stable/modules/linear_model.html#lasso
##############################################################################

##############################################################################
########################## ELASTIC NET REGRESSION ############################
##############################################################################
# http://scikit-learn.org/stable/modules/linear_model.html#elastic-net
# Elastic Net is a linear regression model with both L1 and L2 regularization terms.
# This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge.
# Elastic-net is useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.
# Using both L1 and L2 reguralizers (finding a tradeoff between LASSO and Ridge) is that it allows Elastic-Net to inherit some of Ridge’s stability under rotation.
# The objective function to minimize is:
#   min { (1/2*n)*L2(errors) + alpha*rho*L1(coefs) + 0.5*alpha*(1-rho)*L2(coefs) }

# The parameter rho is called 'L1 ratio'
# For l1_ratio = 0 the penalty is an L2 penalty.
# For l1_ratio = 1 it is an L1 penalty.
# For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

########################################
### CHOOSING ELASTIC NET PARAMETERS BY CROSS VALIDATION
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html

# When fitting an Elastic Net model using ElasticNetCV(), the l1_ratio parameter can be a list, in which case the different values are tested by cross-validation and the one giving the best prediction score is used.
# A good choice of list of values for l1_ratio is often to put more values close to 1 (i.e. Lasso) and less close to 0 (i.e. Ridge), as in [.1, .5, .7, .9, .95, .99, 1]

# To set alphas to try, we can set the two parameters (eps, n_alphas), or to pass on the alphas paramters directly.
# eps sets the length of the path. eps=1e-3 means that alpha_min / alpha_max = 1e-3.
# n_alphas is the number of alphas along the regularization path, used for each l1_ratio.


#######################################
### LASSO AND ELASTIC NET
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)


### Compute paths

eps = 5e-3  # the smaller it is the longer is the path

# compute regularization path using the lasso
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)

# Compute regularization path using the positive lasso
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
    X, y, eps, positive=True, fit_intercept=False)

# Compute regularization path using the elastic net
alphas_enet, coefs_enet, _ = enet_path(
    X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)

# Compute regularization path using the positve elastic net
alphas_positive_enet, coefs_positive_enet, _ = enet_path(
    X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)


### Display results
plt.figure(figsize=[8,10])

plt.subplot(3,1,1)
ax = plt.gca()
ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
l1 = plt.plot(-np.log10(alphas_lasso), coefs_lasso.T)
l2 = plt.plot(-np.log10(alphas_enet), coefs_enet.T, linestyle='--')

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and Elastic-Net Paths')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
plt.axis('tight')


plt.subplot(3,1,2)
ax = plt.gca()
ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
l1 = plt.plot(-np.log10(alphas_lasso), coefs_lasso.T)
l2 = plt.plot(-np.log10(alphas_positive_lasso), coefs_positive_lasso.T,
              linestyle='--')

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and positive Lasso')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
plt.axis('tight')


plt.subplot(3,1,3)
ax = plt.gca()
ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
l1 = plt.plot(-np.log10(alphas_enet), coefs_enet.T)
l2 = plt.plot(-np.log10(alphas_positive_enet), coefs_positive_enet.T,
              linestyle='--')

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Elastic-Net and positive Elastic-Net')
plt.legend((l1[-1], l2[-1]), ('Elastic-Net', 'positive Elastic-Net'),
           loc='lower left')
plt.axis('tight')

plt.draw()
plt.close('all')
#######################################

#######################################
### LASSO AND ELASTIC NET FOR SPARSE SIGNALS
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#example-linear-model-plot-lasso-and-elasticnet-py

from sklearn.metrics import r2_score

### generate some sparse data to play with
np.random.seed(42)

n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)

coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0  # sparsify coef

y = np.dot(X, coef)


### add noise
y += 0.01 * np.random.normal((n_samples,))

# Split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[:n_samples / 2], y[:n_samples / 2]
X_test, y_test = X[n_samples / 2:], y[n_samples / 2:]


### Lasso
from sklearn.linear_model import Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)


### ElasticNet
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)


# Plot
plt.plot(enet.coef_, label='Elastic net coefficients')
plt.plot(lasso.coef_, label='Lasso coefficients')
plt.plot(coef, '--', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.draw()
plt.close('all')
#######################################

#######################################
# Another example of LASSO vs RIDGE
# http://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html#example-applications-plot-tomography-l1-reconstruction-py
#######################################

##############################################################################

##############################################################################
########################### LOGISTIC REGRESSION ##############################
##############################################################################

##############################################################################
### LOGISTIC REGRESSION
# Logistic Regression is a linear model for classification rather than regression.

# This class implements "regularized" logistic regression using the liblinear library, newton-cg and lbfgs solvers.
# The C parameter controls the amount of regularization in the Logistic Regression object:
# a large value for C results in less regularization.
# Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

# penalty="l2" gives Shrinkage (i.e. non-sparse coefficients), while penalty="l1" gives Sparsity.

logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(iris_X_train, iris_y_train)

# In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’ and
# uses the cross-entropy loss, if the ‘multi_class’ option is set to ‘multinomial’.
# (Currently the ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers.)


# The confidence score for a sample is the signed distance of that sample to the hyperplane.


### More on Logistic Regression:
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# More on Logistic Regression: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
##############################################################################

##############################################################################
######################### SUPPORT VECTOR REGRESSION ##########################
##############################################################################

##############################################################################
### SUPPORT VECTOR MACHINES (SVM)
# SVMs are discriminant models.
# They find a combination of samples to build a plane maximizing the margin between the two classes.
# Regularization is set by the C parameter:
# a small value for C means the margin is calculated using many or all
# of the observations around the separating line (more regularization);
# a large value for C means the margin is calculated on observations
# close to the separating line (less regularization).

# One advantage of SVM is that it does not depend on the entire data set, but only
# takes into account the data points that define the support vectors.
# Therefore, it is a good alternative in cases where we have too many data points.

# On the other hand, SVMs are not reliable when the number of features is very high compared to number of data points.

# SVMs can be used in regression SVR (Support Vector Regression), or in classification SVC (Support Vector Classification).

from sklearn import svm
svc = svm.SVC(kernel='linear')
svc.fit(iris_X_train, iris_y_train)
svc.predict(iris_X_test)
iris_y_test


#######################################
### KERNEL TRICK
# Classes are not always linearly separable in feature space.
# The solution is to build a decision function that is not linear but may be polynomial instead.
# This is done using the kernel trick that can be seen as creating
# a decision energy by positioning kernels on observations.

# Polynomial kernel
svc = svm.SVC(kernel='poly', degree=3)
svc.fit(iris_X_train, iris_y_train)
svc.predict(iris_X_test)
iris_y_test

# RBF kernel (Radial Basis Function)
# gamma: inverse of size of radial kernel
svc = svm.SVC(kernel='rbf')
svc.fit(iris_X_train, iris_y_train)
svc.predict(iris_X_test)
iris_y_test


#######################################
### SUPPORT VECTOR REGRESSION
# http://scikit-learn.org/stable/modules/svm.html#regression
from sklearn import svm
clf = svm.SVR()
# clf.fit(X, y)
clf.fit(iris_X_train, iris_y_train)
np.round(clf.predict(iris_X_test))
iris_y_test + 0.0  # added 0.0 to make the list better visually comparable to the list above.


#######################################
### SVR USING NON-LINEAR KERNELS
# http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

# Create estimator objects
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

# Fit models
fit_rbf = svr_rbf.fit(X, y)
fit_lin = svr_lin.fit(X, y)
fit_poly = svr_poly.fit(X, y)

# Estimated values
y_rbf = fit_rbf.predict(X)
y_lin = fit_lin.predict(X)
y_poly = fit_poly.predict(X)

# Compare model scores
fit_rbf.score(X, y)
fit_lin.score(X, y)
fit_poly.score(X, y)

# Plot the data points and model results
plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(X, y_rbf, c='g', label='RBF model')
plt.plot(X, y_lin, c='r', label='Linear model')
plt.plot(X, y_poly, c='b', label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
plt.draw()
plt.close()

#######################################
### A comparison of linear model, SVR, and Random Forests
# http://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html#example-applications-plot-prediction-latency-py

#######################################
### A comparison of kernel Ridge regression and SVR
# http://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html#example-plot-kernel-ridge-regression-py

#######################################
### A note on the performance of linear SVR when number of observations scales up
# http://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution

#######################################
# More on SVM: http://scikit-learn.org/stable/modules/svm.html#svm
##############################################################################

##############################################################################
######################### RANDOM FOREST REGRESSION ###########################
##############################################################################

##############################################################################
### RANDOM FOREST REGRESSION
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# A random forest is a meta estimator that fits a number of classifying decision trees on
# various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.

import numpy as np

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.cross_validation import cross_val_score

dataset = load_boston()
X_full, y_full = dataset.data, dataset.target
n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# Estimate the score on the entire dataset
rfr_estimator = RFR(n_estimators=100)  # random_state=0,
rfr_fit = rfr_estimator.fit(X_full, y_full)
rfr_fit.score(X_full, y_full)
fitted = rfr_fit.predict(X_full)
y_full
scores = cross_val_score(rfr_estimator, X_full, y_full, cv=10)
print(scores)
print("Score with the entire dataset = %.2f" % scores.mean())

# Plot fitted vs actual training data points
plt.figure()
plt.scatter(fitted, y_full)
plt.draw()
plt.close()
##############################################################################

###############################################################################
####################### ARTIFICIAL NEURAL NETWORKS ############################
###############################################################################
# scikit-learn has an unsupervised ANN module based on Restricted Boltzmann Machines (RBM):
# http://scikit-learn.org/stable/modules/neural_networks.html
# The RBM tries to maximize the likelihood of the data using a particular graphical model.
# The parameter learning algorithm used (Stochastic Maximum Likelihood)
# prevents the representations from straying far from the input data,
# which makes them capture interesting regularities,
# but makes the model less useful for small datasets, and usually not useful for density estimation.

# The features extracted by an RBM or a hierarchy of RBMs often give good results when
# fed into a linear classifier such as a linear SVM or a perceptron.

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


###############################################################################
### SET UP

#######################################
def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


#######################################
### Load Data
digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
X, Y = nudge_dataset(X, digits.target)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)


### CREATE THE PIEPINE OF THE MODELS THAT WE WILL USE
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])


###############################################################################
### TRAINING

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 20

# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X_train, Y_train)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)


###############################################################################
### EVALUATION

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))

print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))


###############################################################################
### PLOTTING

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
plt.draw()
plt.close()




