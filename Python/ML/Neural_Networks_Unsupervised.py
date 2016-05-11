
###############################################################################
####################### ARTIFICIAL NEURAL NETWORKS ############################
###############################################################################
# scikit-learn has an unsupervised ANN module based on Restricted Boltzmann Machines (RBM):
# http://scikit-learn.org/stable/modules/neural_networks.html

"""
The RBM tries to maximize the likelihood of the data using a particular graphical model.
The parameter learning algorithm used (Stochastic Maximum Likelihood)
prevents the representations from straying far from the input data,
which makes them capture interesting regularities,
but makes the model less useful for small datasets, and usually not useful for density estimation.
"""

### Restricted Boltzmann machines (RBMs)
"""
Restricted Boltzmann machines (RBM) are unsupervised nonlinear feature learners
based on a probabilistic model.
The features extracted by an RBM or a hierarchy of RBMs often give good results when
fed into a linear classifier such as a linear SVM or a perceptron.

The model makes assumptions regarding the distribution of inputs.
At the moment, scikit-learn only provides BernoulliRBM, which
assumes the inputs are either binary values or values between 0 and 1,
each encoding the probability that the specific feature would be turned on.
The RBM tries to maximize the likelihood of the data using a particular graphical model.
The parameter learning algorithm used (Stochastic Maximum Likelihood)
prevents the representations from straying far from the input data.

This method is known as unsupervised pre-training.
It is used for initializing deep neural networks with the weights of independent RBM

"""

###############################################################################
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

