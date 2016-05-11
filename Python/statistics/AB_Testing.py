
import numpy as np
import pandas as pd
import scipy as sc
from scipy.stats import norm
import matplotlib.pyplot as plt

from statistics import AB_Test_Func as abfunc

#######################################
### SET UP THE EXPERIMENT

# desired test parameters
power = 0.8  # 1 - P(type II error)    the probability of rejecting the null hypothesis when it is false
significance = 0.95  # 1 - P(type I error)    the probability of
two_sided = True  # if we are looking for any change from baseline, use two-sided. If we are specifically looking for an increase or decrease, use one-sided (set this to False)
effect_size = 0.016  # The detectable difference: the level of impact we want to be able to detect with our test

# experiment with sample sizes
N = 10000  # overall size of the experiment

# observed event rates
p1 = 0.10  # test group event rate
p2 = p1-effect_size  # control group event rate  #TODO: in reality, p1 and p2 are both observed. so remove the equation

power, std_pooled, std_unpooled, Z_crit = abfunc.test_power(effect_size, N, p1, p2, significance, two_sided)


#######################################
### COMPUTE SAMPLE SIZE

# Range of detectable differences
dd = np.linspace(0.01, 0.03, 2000)
p1 = 0.1
Ns = [abfunc.get_sample_size(0.80, ddi, p1, p2, significance, two_sided) for ddi in dd]

abfunc.plot_nsample_effectsize(dd, Ns)

plt.savefig(ow.proj_dir+'/Results/Plots/Sample_Size_vs_Detectable_Difference.png', dpi=300)
plt.close('all')


#######################################
### PLOT
Z = effect_size / std_pooled

x, step = np.linspace(-4, 6, 500, retstep=True)
y1 = norm.pdf(x, loc=0, scale=1)  # dont use std_pooled
y2 = norm.pdf(x, loc=Z, scale=1)  # dont use std_unpooled

abfunc.plot_ab_test(x, y1, y2, Z, Z_crit, step)

plt.savefig(ow.proj_dir+'/Results/Plots/AB_Test_Plot.png', dpi=300)
plt.close('all')
