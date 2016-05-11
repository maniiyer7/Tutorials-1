
"""
http://statsmodels.sourceforge.net/0.6.0/_modules/statsmodels/stats/power.html
http://multithreaded.stitchfix.com/blog/2015/05/26/significant-sample/
http://jpktd.blogspot.com/2013/03/statistical-power-in-statsmodels.html
https://vwo.com/blog/how-to-calculate-ab-test-sample-size/

"""
import numpy as np
import pandas as pd
import scipy as sc
from scipy.stats import norm
import matplotlib.pyplot as plt

def test_power(effect_size, N, p1, p2, significance, two_sided):

    n1 = int(N/2)
    n2 = N - n1
    p = (p1*n1 + p2*n2) / (n1+n2)  # overall group event rate


    #######################################
    ### COMPUTE TEST STATISTIC
    # assumption: the standard dev is the same under the null and alternative hypotheses
    if two_sided:
        a = (1-significance)/2
    else:
        a = (1-significance)

    Z_crit = norm.ppf((1-a))

    # compute Z based on observations
    std_pooled = np.sqrt( p*(1.-p) * (1./n1 + 1./n2) )
    std_unpooled = np.sqrt( (p1*(1.-p1)/n1) + (p2*(1.-p2)/n2) )


    #######################################
    ### HYPOTHESIS TESTING

    # H0: x ~ N(0, std_pooled)
    # H1: x ~ N(effect_size, std_unpooled)

    x = p1-p2  # random variable based on our observations
    Z = (x-0.0) / std_pooled  # standard random variable under H0  # 2.7679175482857881

    if Z > Z_crit:
        print('The effect is significant. Reject H0')
    else:
        print('The effect is not significant. Do not reject H0')


    #######################################
    ### POWER ANALYSIS
    A = ( (0-Z_crit)*std_pooled - (effect_size) ) / std_unpooled  # right-side critical value
    B = ( (Z_crit)*std_pooled - (effect_size) )  / std_unpooled  # left-side critical value

    if two_sided:
        power = norm.cdf(A) + 1. - norm.cdf(B)
    else:
        power = 1. - norm.cdf(B)

    print('Test power: {0}'.format(round(power,3)))

    return power, std_pooled, std_unpooled, Z_crit
###############################################################################


###############################################################################
def get_power(effect_size, N, p1, p2, significance, two_sided):
    # assumption 1: n1=n2
    # assumption 2: one-sided test

    p2 = p1 - effect_size

    # Our random var is the difference between event rate p1 and event rate p2.
    # So the variance of our random variable is Var[x] = Var[p1] + Var[p2]
    sigma = np.sqrt(p1*(1-p1) + p2*(1-p2))

    if two_sided:
        Z_crit = norm.ppf((1- (1-significance)/2 ))
    else:
        Z_crit = norm.ppf((1- (1-significance) ))


    # Note: our random var is the difference between control and test, so for every pair of
    # control/test observations, we have only one observation for our rand var. Thus, use n_control
    # or n_test but do not use n_total. Hence the N/2 sizes in the formulas below.
    if two_sided:
        power2 = 1 - norm.cdf(Z_crit - effect_size * np.sqrt(N/2)/sigma) + norm.cdf(-Z_crit - effect_size * np.sqrt(N/2)/sigma)
    else:
        power2 = 1 - norm.cdf(Z_crit - effect_size * np.sqrt(N/2)/sigma)

    return power2, Z_crit
###############################################################################


###############################################################################
def get_sample_size(power2, effect_size, p1, p2, significance, two_sided):
    # assumption 1: n1=n2
    # assumption 2: one-sided test

    p2 = p1 - effect_size

    # Our random var is the difference between event rate p1 and event rate p2.
    # So the variance of our random variable is Var[x] = Var[p1] + Var[p2]
    sigma = np.sqrt(p1*(1-p1) + p2*(1-p2))

    if two_sided:
        Z_crit = norm.ppf((1- (1-significance)/2 ))
    else:
        Z_crit = norm.ppf((1- (1-significance) ))


    # Find the sample size to create desired effect and power
    N = np.power((Z_crit - norm.ppf(1-power2)) * sigma / effect_size, 2)

    return N
###############################################################################


def plot_ab_test(x, y1, y2, Z, Z_crit, step):

    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot2grid((1,1),(0, 0))  # create a 2x2 grid, and give me a reference to the plot located at (0,0).

    ax.plot(x, y1, 'b-', lw=3, alpha=0.6, label='H0 is true')
    ax.plot(x, y2, 'r-', lw=3, alpha=0.6, label='H1 is true')

    ax.fill_between(x[x>Z_crit], 0, y1[x>Z_crit], color='black', alpha=0.3)
    ax.fill_between(x[x>Z_crit], 0, y2[x>Z_crit], color='blue', alpha=0.3)

    ax.set_xlabel('Event Rates')
    ax.set_ylabel('Probability Density')
    plt.suptitle('A/B Test')

    ax.text(0.6*max(x), 0.9*max(y1), r'$\mu={0},\ \sigma={1}$'.format(0, 1), color='b')
    ax.text(0.6*max(x), 0.85*max(y1), r'$\mu={0},\ \sigma={1}$'.format(round(np.mean(Z),2), 1), color='r')

    significance_emp = np.trapz(y1[x>Z_crit], dx=step)
    power_emp = np.trapz(y2[x>Z_crit], dx=step)

    ax.text(0.6*max(x), 0.8*max(y1), r'power={0}, Sig.={1}'.format(round(power_emp,3), round(significance_emp,3)), color='black')

    ax.set_xlim(-4, 6)
    ax.set_ylim(0, 0.5)
    ax.grid(True)

    plt.legend()
###############################################################################


###############################################################################
def plot_nsample_effectsize(dd, Ns):
    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot2grid((1,1),(0, 0))  # create a 2x2 grid, and give me a reference to the plot located at (0,0).

    ax.plot(dd, Ns, 'b-', lw=3, alpha=0.6, label='H0 is true')
    ax.grid(True)

    ax.set_xlabel('Effect Size (Detectable Difference)')
    ax.set_ylabel('Sample Size (total)')
    plt.suptitle('A/B Test Sample Size vs Effect Size')

    ax.grid(True)

    plt.legend()
###############################################################################

