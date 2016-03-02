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
##############################################################################

##############################################################################
############################### PREPROCESSING ################################
##############################################################################
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

##############################################################################
### IMPUTING MISSING VALUES
# Many scikit algorithms do not support missing values
# http://stackoverflow.com/questions/9365982/missing-values-in-scikits-machine-learning

# To handle missing values in the trainig set, use the Imputer module:
# http://scikit-learn.org/stable/auto_examples/missing_values.html

import numpy as np

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score

rng = np.random.RandomState(0)  # random num generator

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
rng.shuffle(missing_samples)  # whether a given row (sample) has any missing values
missing_features = rng.randint(0, n_features, n_missing_samples)  # which features on the rows selected above are missing


### Estimate the score without the lines containing missing values
# ~ is the negative boolean operator. We are unselecting rows with missing values in them
X_filtered = X_full[~missing_samples, :]
y_filtered = y_full[~missing_samples]
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_filtered, y_filtered).mean()
print("Score without the samples containing missing values = %.2f" % score)


### Estimate the score after imputation of the missing values
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
y_missing = y_full.copy()
estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ("forest", RandomForestRegressor(random_state=0,
                                                       n_estimators=100))])
score = cross_val_score(estimator, X_missing, y_missing).mean()
print("Score after imputation of the missing values = %.2f" % score)

# Example of imputation: http://stackoverflow.com/questions/11441751/how-to-get-svms-to-play-nicely-with-missing-data-in-scikit-learn
##############################################################################

##############################################################################
### CATEGORICAL FEATURES
# scikit does not natively handle categorical variables the way R does.
# You need to encode your categorical variables manually (using OneHotEncoder) and use the resulting binary variables as new features.
# This means that a categorical variable with n different levels will be encoded as (n-1) binary variables. This will of course increase the feature space size, which is a drawback.
# Note: R does all of this on the background.
# Note: tree-based methods such as RandomForests can natively work with categorical data and do not need encoding.


### Three modules you may need when processing categorical data are:
# OneHotEncoder, DictVectorizer, LabelEncoder
#   LabelEncoder encodes string values into integer values.
#   OneHotEncoder takes as input categorical values encoded as integers - you can get them from LabelEncoder.
#   DictVectorizer expects data as a list of dictionaries, where each dictionary is a data row with column names as keys.

# http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
# http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#example-hetero-feature-union-py

#######################################
### LabelEncoder
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
# Encode labels with value between 0 and n_classes-1. LabelEncoder can be used to normalize labels.

# Example 1: using numerical categories
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
le.classes_
le.transform([1, 1, 2, 6])
le.inverse_transform([0, 0, 1, 2])

# Example 2: using non-numeric categories
from sklearn import preprocessing
cat_var_orig = ['BMW', 'Mercedes', 'Tesla', 'Audi']  # original categorical variable
le.fit(cat_var_orig)
le.classes_
cat_var_encoded = le.transform(cat_var_orig)
cat_var_orig == le.inverse_transform(cat_var_encoded)
# or, use a different list of the same categories
le.transform(['BMW', 'BMW', 'Audi', 'Mercedes', 'Mercedes'])


#######################################
### DictVectorizer
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html
# Transforms lists of feature-value mappings to vectors.
# This transformer turns lists of mappings (dict-like objects) of feature names to feature values
# into Numpy arrays or scipy.sparse matrices for use with scikit-learn estimators.

# When feature values are strings, this transformer will do a binary one-hot (aka one-of-K) coding:
# one boolean-valued feature is constructed for each of the possible string values that the feature can take on.
# For instance, a feature “f” that can take on the values “ham” and “spam” will become
# two features in the output, one signifying “f=ham”, the other “f=spam”.
# This behavior is not desired in modeling, due to multi-collinearity. Instead, for a cat variable with n classes, we need (n-1) binary values.
# To fix this issue, we can first encode the categorical variables using LabelEncoder, then turn them into separate binary variables using DictVectorizer.

from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
X = v.fit_transform(D)
X

v.inverse_transform(X) == [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]

v.transform({'foo': 4, 'unseen_feature': 3})


# FeatureHasher vs DictVectorizer
# http://scikit-learn.org/stable/auto_examples/text/hashing_vs_dict_vectorizer.html

#######################################
### OneHotEncoder
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# Encode categorical integer features using a one-hot aka one-of-K scheme.
# This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels.

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])

enc.n_values_
enc.feature_indices_
enc.transform([[0, 1, 1]]).toarray()



# Read the following pages for more on encoding categorical variables:
# http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/
# https://www.quora.com/In-scikit-learn-what-is-the-best-way-to-handle-categorical-features-of-high-cardinality-Using-one-hot-encoder-seems-to-blow-up-my-feature-space-Does-assigning-a-number-to-each-cardinality-work-better-or-do-we-run-into-the-risk-of-the-ML-algorithm-assuming-some-ordering-in-these-numeric-labels
# http://stats.stackexchange.com/questions/95212/improve-classification-with-many-categorical-variables
##############################################################################

