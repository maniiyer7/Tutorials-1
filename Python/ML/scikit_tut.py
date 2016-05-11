###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: April 26, 2015
# summary: scikit playground
# other resources:
# http://scikit-learn.org/stable/tutorial/
###############################################################################
from sklearn import svm
from sklearn import datasets
import pickle

###############################################################################
############################## SCIKIT-LEARN ###################################
###############################################################################
# Scikit-learn is a Python module integrating classic machine learning algorithms
# in the tightly-knit world of scientific Python packages (NumPy, SciPy, matplotlib).

### From pandas to scikit
# The ndarray object obtained via the values method has object dtype,
# if values contain more than float and integer dtypes.
# Now even if you slice the str columns away, the resulting array
# will still consist of object dtype and
# might not play well with other libraries such as
# scikit-learn which are expecting a float dtype.

# More on hand-writing identification problem in scikit learn:
# http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#example-classification-plot-digits-classification-py

### Score parameter:
# http://scikit-learn.org/stable/modules/model_evaluation.html
# the score parameter in many estimators corresponds to the appropriate scoring
# method for that estimator. Refer to individual estimator to find out what
# score means in the context of each estimator.

### scikit learn package reference
# http://scikit-learn.org/stable/modules/classes.html

### The mother of all charts: a reference map for deciding which predictor model to choose.
# http://scikit-learn.org/stable/tutorial/machine_learning_map/
###############################################################################

#######################################
# A few general points about scikit functions:
#######################################
# (a) All estimator objects expose a fit method that takes a dataset (usually a 2-d array)
estimator.fit(data)

#######################################
# (b) Every estimator exposes a score method that can judge the quality of the fit (or the prediction) on new data.
estimator.score()

# You can use the score() function without any paremeters, to get the score using training dataset, or
# you can pass test set to the score() function to get test score.
svc.fit(X_train, y_train).score(X_test, y_test)

# The following two parameters also help you find the best models.
clf.best_score_
clf.best_estimator_.C

#######################################
# (c) All the parameters of an estimator can be set when it is instantiated or
# by modifying the corresponding attribute
estimator = Estimator(param1=1, param2=2)
estimator.param1

# When data is fitted with an estimator, parameters are estimated from the data at hand.
# All the estimated parameters are attributes of the estimator object ending by an underscore
estimator.estimated_param_

#######################################
# (d) All supervised estimators in scikit-learn implement a fit(X, y) method to fit the model and
# a predict(X) method that, given unlabeled observations X, returns the predicted labels y.
estimator.predict()

#######################################
# (e) An estimator object can be imported from sklearn module
from sklearn import svm
svc = svm.SVC(kernel='linear')  # equivalent to estimator in examples above
svc.fit(iris_X_train, iris_y_train)

# An estimator is any object that learns from data;
# it may be a classification, regression or clustering algorithm or
# a transformer that extracts/filters useful features from raw data.
# Here the estimator is the class sklearn.svm.SVC that implements support vector classification.
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)  # classifier

# We take out only the last sample as a test set and fit a model.
clf.fit(digits.data[:-1], digits.target[:-1])
# Classify the test set using the learned model
clf.predict(digits.data[-1])


#######################################
# (e) You can save a model to disk for future loading
### SAVE A MODEL IN scikit
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
clf.predict(X[0])

# Option a: use pickle
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0])

# Option b: use joblib
# joblib can only pickle to the disk and not to a string
from sklearn.externals import joblib
joblib.dump(clf, 'filename.pkl')
# NOTE: All files are required in the same folder when reloading the model with joblib.load.
clf = joblib.load('filename.pkl')
# More on model persistence: http://scikit-learn.org/stable/modules/model_persistence.html#model-persistence

###############################################################################
### MORE RESOURCES FOR MACHINE LEARNING WITH scikit-learn
## MACHINE LEARNING EXAMPLES
# http://scikit-learn.org/stable/auto_examples/
## FULL DOCUMENTATION
# http://scikit-learn.org/stable/documentation.html
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

# SQL: (record, attribute)
# ML: (sample, feature)
# Stats: (observation, variable)



###############################################################################
############################## SCIKIT TUTORIAL ################################
###############################################################################
# http://scikit-learn.org/stable/tutorial/
#   http://scikit-learn.org/stable/tutorial/statistical_inference/index.html

###############################################################################
### MODULE 1: Statistical learning
# http://scikit-learn.org/stable/tutorial/
#   http://scikit-learn.org/stable/tutorial/statistical_inference/index.html
#       http://scikit-learn.org/stable/tutorial/statistical_inference/settings.html

from sklearn import datasets
iris = datasets.load_iris()
data = iris.data

digits = datasets.load_digits()
digits.images.shape

# We will be using the last digit example as a test case, so we should visualize it to see the ground truth.
import pylab as pl
pl.imshow(digits.images[-1], cmap=pl.cm.gray_r)
pl.draw()
pl.close()


#######################################
### scikit-learn object model:
# (a) Fitting data: All estimator objects expose a fit method that takes a dataset (usually a 2-d array):
#     estimator.fit(data)
# (b) Estimator parameters: All the parameters of an estimator can be set when it is instantiated or
#     by modifying the corresponding attribute:
#     estimator = Estimator(param1=1, param2=2)
# (c) Estimated parameters: All the estimated parameters are attributes of the estimator object ending by an underscore:
#     estimator.estimated_param_


######################################
### USING OUT-OF-CORE LEARNING AND OTHER STRATEGIES TO SCALE COMPUTATIONALLY
# http://scikit-learn.org/stable/modules/scaling_strategies.html

# Example: out-of-core classification of text documents
# http://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html#example-applications-plot-out-of-core-classification-py
######################################



###############################################################################
################################## SUMMARY ####################################
###############################################################################

###############################################################################
### MODULE 5: Putting it all together
# http://scikit-learn.org/stable/tutorial/
#   http://scikit-learn.org/stable/tutorial/statistical_inference/index.html
#       http://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html





###############################################################################
############################# scikit PIPELINE #################################
###############################################################################
### What is a pipeline?
# http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# Pipeline of transforms with a final estimator.
# Sequentially apply a list of transforms and a final estimator.
# Intermediate steps of the pipeline must be ‘transforms’, that is,
# they must implement fit and transform methods.
# The final estimator only needs to implement fit.

# The purpose of the pipeline is to assemble several steps that
# can be cross-validated together while setting different parameters.

# Parameters of pipelines can be set using ‘__’ separated parameter names.

#######################################
### Example: A pipeline consisting of feature selection using ANOVA and then fitting a SVM model
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline

### generate some data to play with
X, y = samples_generator.make_classification(
        n_informative=5, n_redundant=0, random_state=42)


### ANOVA SVM-C
# Step 1: create the filter
anova_filter = SelectKBest(f_regression, k=5)

# Step 2: create the classifier
clf = svm.SVC(kernel='linear')

# Step 3: create the pipeline
anova_svm = Pipeline([('anova', anova_filter),
                      ('svc', clf)])

# Step 4 (optional): fine-tune the pipeline.
# You can set the parameters using the names of pipeline elements and double underscors __ to attach parameters values to them.
# For instance, fit using a k of 10 in the SelectKBest
# and a parameter 'C' of the svm
anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)

prediction = anova_svm.predict(X)

anova_svm.score(X, y)


#######################################
### Example: Imputing missing values before building an estimator
# http://scikit-learn.org/stable/auto_examples/missing_values.html

# NOTE: Imputing does not always improve the predictions,
# so please check via cross-validation.
# Sometimes dropping rows or using marker values is more effective.

import numpy as np

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score

rng = np.random.RandomState(0)

dataset = load_boston()
X_full, y_full = dataset.data, dataset.target
n_samples = X_full.shape[0]
n_features = X_full.shape[1]

### Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_full, y_full).mean()
print("Score with the entire dataset = %.2f" % score)


### Add missing values in 75% of the lines
missing_rate = 0.75
n_missing_samples = np.floor(n_samples * missing_rate)
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                     dtype=np.bool)))
rng.shuffle(missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)


### Estimate the score without the lines containing missing values
X_filtered = X_full[~missing_samples, :]
y_filtered = y_full[~missing_samples]
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_filtered, y_filtered).mean()
print("Score without the samples containing missing values = %.2f" % score)


### Estimate the score after imputation of the missing values
X_missing = X_full.copy()
# We are using 0 as the indicator variable, so that the Impute() function can pick it up. We could have used any other indicator value.
X_missing[np.where(missing_samples)[0], missing_features] = 0
y_missing = y_full.copy()

estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ("forest", RandomForestRegressor(random_state=0,
                                                       n_estimators=100))])

score = cross_val_score(estimator, X_missing, y_missing).mean()
print("Score after imputation of the missing values = %.2f" % score)
###############################################################################

