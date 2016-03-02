###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: April 26, 2015
# summary: pandas playground
# other resources:
# http://pandas.pydata.org/pandas-docs/stable/tutorials.html
###############################################################################

import numpy as np
import pandas as pd
# to get ggplot-like style for plots
pd.set_option('display.mpl_style', 'default')
pd.set_option('display.line_width', 5000)
pd.set_option('display.max_columns', 60)

import sklearn
from sklearn import hmm
# https://github.com/hmmlearn/hmmlearn
sys.path.append('/usr/local/lib/pythonPckgGits/hmmlearn')
import hmmlearn

import matplotlib.pyplot as plt
plt.ion()  # turn on interactive plotting
plt.style.use('ggplot')  # make matplotlib appearance similar to ggplot

import functools
import itertools
import os, sys

# Get Python environment parameters
print 'Python version ' + sys.version
print 'Pandas version: ' + pd.__version__


# Specify file locations
plotPath = '/Users/amirkavousian/Documents/Py_Codes/Plots'
resultsPath = '/Users/amirkavousian/Documents/Py_Codes/Results'

###############################################################################
###############################################################################
###############################################################################

###############################################################################
### HMM Modeling Using sklearn
# Classes in this module include MultinomalHMM GaussianHMM, and GMMHMM.
# They implement HMM with emission probabilities determined by
# multimomial distributions, Gaussian distributions and mixtures of Gaussian distributions.

### Building HMM and generating samples
startprob = np.array([0.6, 0.3, 0.1])
transmat = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
means = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
covars = np.tile(np.identity(2), (3, 1, 1))

model = hmm.GaussianHMM(3, "full", startprob, transmat)
model.means_ = means
model.covars_ = covars
X, Z = model.sample(100)


### Training HMM parameters and infering the hidden states
# You can train an HMM by calling the fit method.
model2 = hmm.GaussianHMM(3, "full")
model2.fit([X])

# To avoid getting stuck in a local optima, run fit with various initializations and select the highest scored model.
# The score of the model can be calculated by the score method.

# The inferred optimal hidden states can be obtained by calling predict method.
Z2 = model2.predict(X)

plt.figure()
plt.plot(Z, Z2, 'g')
plt.draw()
plt.close()



###############################################################################
###############################################################################
###############################################################################
### HMM generating example (sampling from a HMM)
# http://scikit-learn.org/0.13/auto_examples/plot_hmm_sampling.html#example-plot-hmm-sampling-py
# Prepare parameters for a 3-components HMM
# Initial population probability
start_prob = np.array([0.6, 0.3, 0.1, 0.0])

# The transition matrix, note that there are no transitions possible
# between component 1 and 3
trans_mat = np.array([[0.7, 0.2, 0.0, 0.1],
                      [0.3, 0.5, 0.2, 0.0],
                      [0.0, 0.3, 0.5, 0.2],
                      [0.2, 0.0, 0.2, 0.6]])

# The means of each component
# In this case, each component (state) has a two-dimensional observation (x, y coordinates) associated with it.
# To show the observation probability matrix, we need a multivariable Normal variable for each component.
means = np.array([[0.0,  0.0],
                  [0.0, 11.0],
                  [9.0, 10.0],
                  [11.0, -1.0],
                  ])

# The covariance of each component
covars = .5 * np.tile(np.identity(2), (4, 1, 1))

# Build an HMM instance and set parameters
model = hmm.GaussianHMM(4, "full", start_prob, trans_mat,
                        random_state=42)

# Instead of fitting it from the data, we directly set the estimated
# parameters, the means and covariance of the components
model.means_ = means
model.covars_ = covars

###############################################################
# Generate samples
# X: observation matrix; Z: hidden state matrix
X, Z = model.sample(500)

# Plot the sampled data
plt.plot(X[:, 0], X[:, 1], "-o", label="observations", ms=6,
         mfc="orange", alpha=0.7)

# Indicate the component numbers
for i, m in enumerate(means):
    plt.text(m[0], m[1], 'Component %i' % (i + 1),
             size=17, horizontalalignment='center',
             bbox=dict(alpha=.7, facecolor='w'))

plt.legend(loc='best')
plt.draw()
plt.close()

###############################################################################
###############################################################################
###############################################################################

###############################################################################
### FITTING HMM TO FINANCIAL DATA
print __doc__

import datetime_tut
import numpy as np
import pylab as pl
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from sklearn.hmm import GaussianHMM

###############################################################################
### Downloading and cleansing the data
date1 = datetime_tut.date(1995, 1, 1)  # start date
date2 = datetime_tut.date(2012, 1, 6)  # end date

# get quotes from yahoo finance
quotes = quotes_historical_yahoo("INTC", date1, date2)
if len(quotes) == 0:
    raise SystemExit

# unpack quotes
dates = np.array([q[0] for q in quotes], dtype=int)
close_v = np.array([q[2] for q in quotes])
volume = np.array([q[5] for q in quotes])[1:]

# take diff of close value
# this makes len(diff) = len(close_t) - 1
# therefore, others quantity also need to be shifted
diff = close_v[1:] - close_v[:-1]
dates = dates[1:]
close_v = close_v[1:]

# pack diff and volume for training
X = np.column_stack([diff, volume])

###############################################################################
# Run Gaussian HMM
print "fitting to HMM and decoding ...",
n_components = 5

# make an HMM instance and execute fit
model = GaussianHMM(n_components, covariance_type="diag", n_iter=1000)

model.fit([X])

# predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

print "done\n"

###############################################################################
# print trained parameters and plot
print "Transition matrix"
print model.transmat_
print ""

print "means and vars of each hidden state"
for i in xrange(n_components):
    print "%dth hidden state" % i
    print "mean = ", model.means_[i]
    print "var = ", np.diag(model.covars_[i])
    print ""

years = YearLocator()   # every year
months = MonthLocator()  # every month
yearsFmt = DateFormatter('%Y')
fig = pl.figure()
ax = fig.add_subplot(111)

for i in xrange(n_components):
    # use fancy indexing to plot data in each state
    idx = (hidden_states == i)
    ax.plot_date(dates[idx], close_v[idx], 'o', label="%dth hidden state" % i)
ax.legend()

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.autoscale_view()

# format the coords message box
ax.fmt_xdata = DateFormatter('%Y-%m-%d')
ax.fmt_ydata = lambda x: '$%1.2f' % x
ax.grid(True)

fig.autofmt_xdate()
pl.draw()
pl.close()


###############################################################################
#################################### HMM  #####################################
###############################################################################
# http://www.cs.colostate.edu/~anderson/cs440/index.html/doku.php?id=notes:hmm2

startprob = np.array([0.6, 0.3, 0.1])
transmat = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
means = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
covars = np.tile(np.identity(2), (3, 1, 1))
model = hmm.GaussianHMM(3, "full", startprob, transmat)
model.means_ = means
model.covars_ = covars
X, Z = model.sample(100)

# ###
model2 = hmm.GaussianHMM(3, "full")
model2.fit(obMat.iloc[0, :])
Z2 = model2.predict(X)



# data_df
# ### Fit HMM
n_components = 3
# # make an HMM instance and execute fit
model = hmm.GaussianHMM(n_components, covariance_type="diag", n_iter=1000)
model.fit([X])
# # predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)


