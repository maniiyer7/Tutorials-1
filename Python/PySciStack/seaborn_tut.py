


###############################################################################
################################# SEABORN #####################################
###############################################################################

###############################################################################
### seaborn
# Seaborn comes with a number of customized themes and a high-level interface for controlling the look of matplotlib figures.
# To switch to seaborn defaults, simply import the package.
# To control the style, use the axes_style() and set_style() functions.
# To scale the plot, use the plotting_context() and set_context() functions.
# Tutorial source:
# http://stanford.edu/~mwaskom/software/seaborn/tutorial.html
# See the following pages for updates on this bug:
# https://github.com/matplotlib/matplotlib/issues/1266
# https://github.com/matplotlib/matplotlib/issues/2654
# https://github.com/matplotlib/matplotlib/issues/166/


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set()  # reset seaborn settings


def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)


plt.figure();
sinplot()
plt.draw();
plt.close();


# There are five preset seaborn themes: darkgrid, whitegrid, dark, white, and ticks.
sns.set_style("whitegrid")
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
plt.figure();
sns.boxplot(data);
plt.draw()
plt.close()

### BACKGROUND STYLES
# To remove the grid completely
sns.set_style("dark")
plt.figure()
sinplot()
plt.draw()
plt.close();

sns.set_style("white")
plt.figure()
sinplot()
plt.draw()
plt.close()

sns.set_style('ticks')
plt.figure()
sinplot()
plt.draw()
plt.close()

# remove top and right border lines
sns.set_style('ticks')
plt.figure()
sinplot()
sns.despine()
plt.draw()
plt.close()

# offset the borders
f, ax = plt.subplots()
sns.violinplot(data)
sns.despine(offset=10, trim=True);
plt.draw()

# remove specific borders
plt.figure()
sns.set_style("whitegrid")
sns.boxplot(data, color="deep")
sns.despine(left=True)
plt.draw()
plt.close()

### Further customizing Seaborn parameters
# To set seaborn plot arguments,
# pass a dictionary of parameters to the rc argument of axes_style() and set_style().
# To see a list of argments available to change:
# http://stanford.edu/~mwaskom/software/seaborn/tutorial/aesthetics.html
# In addition to set_style(), the higher-level set() function takes a dictionary of any matplotlib parameters
# To reset the parameters to their default:
sns.set()

### Scaling plot elements with plotting_context() and set_context()
sns.set()
sns.set_context("paper")
plt.figure(figsize=(8, 6))
sinplot()
plt.draw()
plt.close()

sns.set_context("talk")
plt.figure(figsize=(8, 6))
sinplot()
plt.draw()
plt.close()

sns.set_context("poster")
plt.figure(figsize=(8, 6))
sinplot()
plt.draw()
plt.close()

# To set specific context parameters, pass a dictionary of those elements to set_context() function.
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.figure()
sinplot()
plt.draw()
plt.close()

#*#TODO: continue here:
# http://stanford.edu/~mwaskom/software/seaborn/tutorial/color_palettes.html#palette-tutorial





###############################################################################
### FACTORPLOT()
# factorplot function is a wrapper function for a number of plots for categorical data.
# The factorplot function can call lmplot(), regplot(), pointplot(), barplot().
a=['A', 'B', 'C', 'D', 'E', 'F']
b=np.random.randint(0, 3, 20)
c=np.random.randint(2, 6, 20)
ver = pd.DataFrame({
    'Fac1' : [a[i] for i in b],
    'Fac2' : [a[i] for i in c],
    'Int1' : np.random.randint(0, 50, 20),
    'Int2' : np.random.randint(15, 75, 20)
})

#NOTE: Some seaborn plots have known bugs when used with IPython on Mac OSX.
# The following workarounds have been suggested on forums, not none worked for me.
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import seaborn as sns

# import matplotlib
# matplotlib.use('MacOSX')
# from matplotlib import pyplot as plt

# factorplot(0 function takes 3 positional params: x-axis, y-axis, hue

# point plots and bar plots focus on the central tendency of the data
# (with a measure of the error associated with that value)

# point plots are better for comparing between conditions:
g = sns.factorplot("Fac1", "Int1", "Fac2", ver, kind="point",
                   palette="PRGn", aspect=2.25)
plt.draw()
plt.close()


# bar plots are better for understanding overall magnitude and how far it is from 0:
g = sns.factorplot("Fac1", "Int1", "Fac2", ver, kind="bar",
                   palette="PRGn", aspect=2.25)
plt.draw()
plt.close()


# boxplots visualize the distribution of the data in different categories.
g = sns.factorplot("Fac1", "Int1", "Fac2", ver, kind="box")
plt.draw()
plt.close()


# specify palette
plt.figure()
g = sns.factorplot("Fac1", "Fac2", ver, kind="box",
                   palette="PRGn", aspect=2.25)
g.set(ylim=(0, 600))
plt.draw()
plt.close()


# barplot using seaborn
# When y is missing, the height of the plot shows the count of observations in each category
plt.figure()
sns.factorplot("Fac1", data=ver, hue="Fac2", size=3, aspect=2)
plt.draw()
plt.close()


# By default the height of the bars/points shows the mean and 95% confidence interval, but both can be changed.
g = sns.factorplot("Fac1", "Int1", "Fac2", ver, kind="bar",
                   palette="PRGn", aspect=2.25, estimator=np.median)
plt.savefig(plotPath+'/myfig2.png')
plt.draw()
plt.close()

##

###############################################################################
################## PLOTTING CATEGORICAL DATA WITH SEABORN #####################
###############################################################################

# https://stanford.edu/~mwaskom/software/seaborn/tutorial/categorical.html