###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: April 26, 2015
# summary: matplotlib playground
###############################################################################

import matplotlib.pyplot as plt
plt.ion()  # turn on interactive plotting
plt.style.use('ggplot')  # make matplotlib appearance similar to ggplot

import numpy as np
import pandas as pd
import seaborn as sns

# to get ggplot-like style for plots
pd.set_option('display.mpl_style', 'default')
pd.set_option('display.line_width', 5000)
pd.set_option('display.max_columns', 60)

# File locations4
plotPath = '/Users/amirkavousian/Documents/Py_Codes/Plots'

###############################################################################
### MORE RESOURCES
## matplotlib beginners guide:
# http://matplotlib.org/users/beginner.html

## To get a list of all line style markers:
# http://matplotlib.org/api/lines_api.html

# all sequences are converted to numpy arrays internally.

# matplotlib API reference (looks very useful)
# http://matplotlib.org/api/pyplot_api.html


###############################################################################
############################ POINT (SCATTER) PLOTS ############################
###############################################################################
# http://matplotlib.org/users/pyplot_tutorial.html

#######################################
### SCATTERPLOT USING SEPARATE LISTS FOR x AND y VALUES
# Note that the only way that the red dots and green lines are coded differently in matplotlib is by their point/line type ('ro' vs 'g--').
# matplotlib does not have separate functions for line vs point plots.
# The type of the plot is determined by the line/point style parameter that is passed to the plt.plot() function.
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.plot([1,2,3,4], [1,4,9,16], 'g--')
plt.axis([0, 6, 0, 20])  #  [xmin, xmax, ymin, ymax]
plt.draw()
plt.close('all')
#######################################


#######################################
### UNIVARIATE PLOTS USING ndarray
# Time series, or a sequential unvariate variable
# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.draw()
plt.close('all')
#######################################

#######################################
### SCATTER
# http://matplotlib.org/examples/shapes_and_collections/scatter_demo.html
import numpy as np
import matplotlib.pyplot as plt

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
plt.close('all')

###############################################################################
################################ LINE PLOTS ###################################
###############################################################################
# http://matplotlib.org/users/pyplot_tutorial.html
###############################################################################

#######################################
### LINE PLOT OF SEQUENTIAL VALUES (no x values; example: time-series)
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()

plt.plot([3.6, 4.7])
plt.title("Plot y values")
plt.xlabel("index")

plt.plot(ts, 'k-', color='blue')

plt.draw()
# plt.savefig(plotPath+'/myfig2.png')
plt.close()
#######################################

#######################################
### LINE PLOT BY x,y COORDINATES
# plt.plot([x], [y])

# By default, matplotlib assumes the (x,y) coordinates refer to a line.
plt.figure()
y1 = [1,4,9,16]  # list of x values of all points
x1 = [1,2,3,4]  # list of y values of all points
plt.plot(x1, y1, 'ro')
plt.plot(x1, y1, 'g-')
plt.axis([0, 6, 0, 20])  # [xmin, xmax, ymin, ymax]
plt.draw()
plt.close()
#######################################

#######################################
### LINE SEGMENTS USING COORDINATES OF START/END POINTS
# plt.plot([ [x_start, y_start]  ,
#            [x_end,   y_end]   ],
#          [ ...                ])

a=np.asarray([[[0,0],[1,1]],
              [[1,2],[2,3]],
              [[5,2],[2,6]],
              [[7,3],[5,9]]])
plt.figure()
plt.plot(a[:,:,0].T, a[:,:,1].T)
plt.draw()
plt.close()
#######################################

#######################################
### LINE SEGMENTS USING START/END COORDINATES (no collection of points)
# http://stackoverflow.com/questions/20058711/plotting-line-segments-from-the-set-of-startpoints-and-endpoints
# Each start/end point must be
# plt.plot([[x_start], [x_end]], [[y_start], [y_end]], 'b-')

plt.figure()
plt.plot([[0, 1, 5, 7], [1, 2, 2, 5]],   # [[x_start], [x_end]]
         [[0, 2, 2, 3], [1, 3, 6, 9]])   # [[y_start], [y_end]]
plt.draw()
plt.close()
#######################################

#######################################
### MULTIPLE LINES ON THE SAME FIGURE BY REPEATING x, y VALUES
# You can repeat the pattern to plot several lines with a single command.
t = np.arange(0., 5., 0.2)
plt.figure()
plt.plot(t, t, t, t**2, t, t**3)
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')  # optional: you can even change the patterns between different lines
plt.draw()
plt.close()
#######################################

#######################################
### MULTIPLE LINES ON SAME FIGURE USING AN ARRAY OF Y VALUES
# y is a (N, k) array, with N being the number of points on each line, and k being the number of lines.
# Each sub-array within y corresponds to one x value (which is the measurement point).
# Each element of the sub-array is the different observations we took for that measurement point.
# y = [ [l1_p1_y, l2_p1_y, l3_p1_y],
#       [l1_p2_y, l2_p2_y, l3_p2_y],
#       [l1_p3_y, l2_p3_y, l3_p3_y],
#       [l1_p4_y, l2_p4_y, l3_p4_y],
#       [l1_p5_y, l2_p5_y, l3_p5_y],
#       [ ...                     ],
#     ]

x = np.linspace(0, 10, 100)
y = np.random.rand(100, 3)
for i in range(y.shape[1]):
    y[:,i] += i

ax = plt.gca()
ax.set_color_cycle(['b', 'r', 'g'])
ax.plot(x, y)
plt.draw()
plt.close('all')
#######################################

#######################################
### EXAMPLE OF LINE PLOTS USING NumPy ARRAYS
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.draw()
#######################################

###############################################################################
############################### PLOT PROPERTIES ###############################
###############################################################################

###############################################################################
#######################################
### SET LINE WIDTH AND OTHER PROPERTIES

### (option a) use linewidth argument in the plt.plot() function.
plt.plot(t, t, 'r--', linewidth=2)


### (option b) To set linewidth, you can also use the setter methods of the Line2D instance.
# plt.plot() returns a list of lines; e.g., line1, line2 = plot(x1,y1,x2,y2).
line, = plt.plot(t, t**2, '-')  # The comma on the left side is for tuple unpacking
line.set_antialiased(False) # turn off antialising
plt.draw()
plt.close()


### (option c) use setp() function of line objects
x1 = [1,2,3,4]
y1 = [1,4,9,16]
x2 = [1,2,3,4]
y2 = [1,8,27,64]
lines = plt.plot(x1, y1, x2, y2)
# use keyword args
plt.setp(lines, color='r', linewidth=2.0)
# or MATLAB style string value pairs
plt.setp(lines, 'color', 'r', 'linewidth', 2.0)
plt.draw()
plt.close('all')


### To see a list of settable line properties:
# http://matplotlib.org/users/pyplot_tutorial.html
lines = plt.plot([1,2,3])
plt.setp(lines)
#######################################

#######################################
### SET LINE COLOR CYCLE FOR MULTI LINE PLOTS
# The function gca() returns the current axes (a matplotlib.axes.Axes instance).
# The function gcf() returns the current figure (matplotlib.figure.Figure instance)
# To make certain adjustments, we need to grab current axes vs current figure.
# For instance, to set the color cycle (when creating multiple objects on the plot),
# we need current axes.
x = np.linspace(0, 10, 100)
y = np.random.rand(100, 3)
for i in range(y.shape[1]):
    y[:,i] += i

ax = plt.gca()
ax.set_color_cycle(['b', 'r', 'g'])
ax.plot(x, y)
plt.draw()
plt.close('all')
#######################################
###############################################################################

###############################################################################

###############################################################################
############################ TEXT AND ANNOTATIONS #############################
###############################################################################

### WORKING WITH TEXT
# The text() command can be used to add text in an arbitrary location, and
# the xlabel(), ylabel() and title() are used to add text in the indicated locations.
# More on text: http://matplotlib.org/users/text_intro.html
# Example:
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')

# All of the text() commands return an matplotlib.text.Text instance.
# More on text properties: http://matplotlib.org/users/text_props.html
t = plt.xlabel('my data', fontsize=14, color='red')


### TeX expressions:
# (a) add a TeX expression surrounded by dollar signs;
# (b) add 'r' before the text to signify that the string is a raw string and not to treat backslashes as python escapes.
# More details on writing mathematical expressions: http://matplotlib.org/users/mathtext.html#mathtext-tutorial
plt.title(r'$\sigma_i=15$')


### the annotate() method provides helper functionality as a wrapper around text() function.
ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

plt.ylim(-2,2)
plt.draw()
plt.close()

# More on annotating:
# http://matplotlib.org/users/annotations_intro.html
# http://matplotlib.org/users/annotations_guide.html#plotting-guide-annotation
# http://matplotlib.org/examples/pylab_examples/annotation_demo.html#pylab-examples-annotation-demo
###############################################################################

###############################################################################
################################# HISTOGRAM ###################################
###############################################################################
### HISTOGRAM
# If the option normed=1 is set, draws a density version of the histogram: the bin heights
# are chosen such that the area under the curve is equal to 1. Ie, the width of each bar
# multiplied by the height of that bar equals the probability of that bar.
#

import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# DENSITY: the histogram of the data (the area of each bar is equal to its probability
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

# FREQUENCY: remove normed=1 if you want to plot frequeny
# n, bins, patches = plt.hist(x, 50, facecolor='g', alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.draw()
plt.close()


#######################################
### COMPARISON OF PROBABILITY AND FREQUENCY HISTOGRAMS (also an example of a multiplot)
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

plt.figure(figsize=[10,8])  # is not necessary unless you want to change the figure size.
ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)  # the shape parameter (2,3) will dictate plot dimension ratios
ax2 = plt.subplot2grid((2,3), (1,0), colspan=3)

# DENSITY: the histogram of the data (the area of each bar is equal to its probability
n, bins, patches = ax1.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
ax1.set_xlabel('Smarts')
ax1.set_ylabel('Probability')
ax1.set_title('Histogram of IQ')
ax1.text(60, .025, r'$\mu=100,\ \sigma=15$')
ax1.axis([40, 160, 0, 0.03])
ax1.grid(True)

# FREQUENCY: remove normed=1 if you want to plot frequeny
n, bins, patches = ax2.hist(x, 50, facecolor='g', alpha=0.75)
ax2.set_xlabel('Smarts')
ax2.set_ylabel('Frequency')
ax2.set_title('Histogram of IQ')
ax2.text(60, .025, r'$\mu=100,\ \sigma=15$')
ax2.axis([40, 160, 0, 800])
ax2.grid(True)

plt.draw()
plt.close('all')


#######################################
### More examples of histogram
# http://matplotlib.org/examples/pylab_examples/histogram_demo_extended.html
# http://matplotlib.org/1.5.0/examples/pylab_examples/index.html

# Histogram with color-coded bars
# http://matplotlib.org/1.5.0/examples/pylab_examples/hist_colormapped.html

# Histogram with y axis presented as percent
# http://matplotlib.org/1.5.0/examples/pylab_examples/histogram_percent_demo.html
###############################################################################


###############################################################################
############################### TIGHT LAYOUT ##################################
###############################################################################
# http://matplotlib.org/users/tight_layout_guide.html

###############################################################################

###############################################################################
############################# TIME-SERIE PLOTS ################################
###############################################################################

###############################################################################
### TIME-SERIES PLOTTING
# NOTE: when plotting pandas time-series objects, use the native pandas .plot()
# function for better x-axis labeling.
dft = pd.DataFrame(np.random.randn(1000,1),
                   columns=['A'],
                   index=pd.date_range('20130101',periods=1000,freq='D'))

# Regular matplotlib plot
plt.figure()
plt.plot(dft)
plt.draw()
plt.close()

# pandas time-series built-in plot function takes care of x axis titles and ticks
plt.figure()
plt.plot(dft)
dft.plot()
plt.draw()
plt.close('all')
###############################################################################

###############################################################################
################################## BOXPLOT ####################################
###############################################################################
### BOXPLOT WITH matplotlib
# http://matplotlib.org/1.5.0/examples/pylab_examples/boxplot_demo.html

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use('ggplot')  # make matplotlib appearance similar to ggplot
sns.set()  # reset seaborn settings

# fake up some data
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)

# basic plot
plt.boxplot(data)

# notched plot
plt.figure()
plt.boxplot(data, 1)

# change outlier point symbols
plt.figure()
plt.boxplot(data, 0, 'gD')

# don't show outlier points
plt.figure()
plt.boxplot(data, 0, '')

# horizontal boxes
plt.figure()
plt.boxplot(data, 0, 'rs', 0)

# change whisker length
plt.figure()
plt.boxplot(data, 0, 'rs', 0, 0.75)


plt.close('all')
#######################################

#######################################
### BOXPLOT BROKEN DOWN BY GROUPS (FOR COMPARISON OF A VARIABLE ACROSS DIFFERENT GROUPS)
# fake up some more data
spread = np.random.rand(50) * 100
center = np.ones(25) * 40
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
d2 = np.concatenate((spread, center, flier_high, flier_low), 0)

# reshape the ndarray into an array of lists, each with 1 element in them.
# Note that the -1 in reshape means "calculate this dimension based on the values of other dimensions".
data.shape = (-1, 1)
d2.shape = (-1, 1)
# data = concatenate( (data, d2), 1 )

# We need a list of lists to create a multi-box boxplot.
# Making a 2-D array only works if all the columns are the
# same length.  If they are not, then use a list instead.
# This is actually more efficient because boxplot converts
# a 2-D array into a list of vectors internally anyway.
# In the example below, data and d2 are ndarrays of the same shape.
# d2[::2,0] is a sub-sample of d2, and therefore has different length. Therefore, we introduced it as a list.
data_all = [data, d2, d2[::2, 0]]

# multiple box plots on one figure
plt.figure()
plt.boxplot(data_all)

plt.show()
plt.close('all')
#######################################

#######################################
### BOXPLOT BY GROUP, COLOR CODED
# http://stackoverflow.com/questions/16592222/matplotlib-group-boxplots

def set_box_color(bp, color):
    """Set the colors of a given boxplot bp to a specified color"""
    # http://stackoverflow.com/questions/25408393/getting-individual-colors-from-a-color-map-in-matplotlib
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


data_a = [[1,2,5], [5,7,2,2,5], [7,2,5]]
data_b = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]
color_a = '#D7191C'
color_b = '#2C7BB6'
label_a = 'Apples'
label_b = 'Oranges'
categories = ['A', 'B', 'C']
cmap = plt.get_cmap('Blues')


bpl = plt.boxplot(data_a, positions=np.array(xrange(len(data_a)))*2.0 - 0.4, sym='', widths=0.6)
bpr = plt.boxplot(data_b, positions=np.array(xrange(len(data_b)))*2.0 + 0.4, sym='', widths=0.6)
set_box_color(bpl, color_a)
set_box_color(bpr, color_b)

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c=color_a, label=label_a)
plt.plot([], c=color_b, label=label_b)
plt.legend()

# categories
plt.xticks(xrange(0, len(categories) * 2, 2), categories)
plt.xlim(-2, len(categories)*2)
plt.ylim(0, 8)
# plt.tight_layout()
plt.draw()

plt.close('all')
###############################################################################

###############################################################################
################################# HEATMAP #####################################
###############################################################################

###############################################################################
### HEATMAPS
#***#
fig, ax = plt.subplots()
heatmap = ax.pcolor(corr, cmap=plt.cm.Blues, alpha=0.8)

# Format
fig = plt.gcf()
# fig.set_size_inches(11, 11)

# turn off the frame
ax.set_frame_on(False)

# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(corr.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(corr.shape[1]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

# Set the labels

# label source:https://en.wikipedia.org/wiki/Basketball_statistics
labels = train_X.columns

# note I could have used nba_sort.columns but made "labels" instead
ax.set_xticklabels(labels, minor=False)
ax.set_yticklabels(corr.index, minor=False)

# rotate the
plt.xticks(rotation=90)

ax.grid(False)

# Turn off all the ticks
ax = plt.gca()

for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False

plt.draw()
plt.savefig(localPath+'/Results/CorrPlot2.png')
plt.close()

#######################################
### matplotlib documentation calls heatmaps '2d histograms'
# http://matplotlib.org/1.5.0/examples/pylab_examples/hist2d_demo.html
# http://matplotlib.org/1.5.0/examples/pylab_examples/hist2d_log_demo.html


###############################################################################
################################ BAR CHART ####################################
###############################################################################
# http://matplotlib.org/examples/api/barchart_demo.html
"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
N = 5
menMeans = (20, 35, 30, 35, 27)
menStd = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)

womenMeans = (25, 32, 34, 20, 25)
womenStd = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, womenMeans, width, color='y', yerr=womenStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width)
ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))

ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()

plt.close('all')


###############################################################################
##################### STACKED BAR CHART AND PIE CHART #########################
###############################################################################
### VISUALIZING THE DISTRIBUTION OF A CATEGORICAL VARIABLE IN A POPULATION
# A stacked barchart says something about the population as a whole and is almost always based on count of each group,
# while a multiple barchart compares certain value among different groups. This value can be count (size) of each group too;
# but in that can the goal is to compare the counts of individual groups against each other.
# This plot is frequently used in place of a piechart.


###############################################################################
# https://de.dariah.eu/tatom/topic_model_visualization.html
import numpy as np
import matplotlib.pyplot as plt

docnames = ['Austen_Emma',
 'Austen_Pride',
 'Austen_Sense',
 'CBronte_Jane',
 'CBronte_Professor',
 'CBronte_Villette']

doctopic = np.array([
    [ 0.0625,  0.1736,  0.0819,  0.4649,  0.2171],
    [ 0.0574,  0.1743,  0.0835,  0.4008,  0.2839],
    [ 0.0599,  0.1645,  0.0922,  0.2034,  0.4801],
    [ 0.189 ,  0.1897,  0.3701,  0.1149,  0.1362],
    [ 0.2772,  0.2681,  0.2387,  0.0838,  0.1322],
    [ 0.3553,  0.193 ,  0.2409,  0.0865,  0.1243]])

N, K = doctopic.shape  # N documents (populations), K topics (groups)
ind = np.arange(N)  # the x-axis locations for the novels
width = 0.5  # the width of the bars
plots = []
height_cumulative = np.zeros(N)
for k in range(K):
    color = plt.cm.Spectral(k/float(K))
    print color
    if k == 0:
        p = plt.bar(ind, doctopic[:, k], width, color=color)
    else:
        p = plt.bar(ind, doctopic[:, k], width, bottom=height_cumulative, color=color)
    height_cumulative += doctopic[:, k]
    plots.append(p)

plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1
plt.ylabel('Topics')
plt.title('Topics in novels')
plt.xticks(ind+width/2, docnames)
plt.yticks(np.arange(0, 1, 10))
topic_labels = ['Topic #{}'.format(k) for k in range(K)]
plt.legend([p[0] for p in plots], topic_labels)

plt.draw()
plt.close('all')

# see http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend for details
# on making a legend in matplotlib
###############################################################################

###############################################################################
# http://stackoverflow.com/questions/30858138/manipulating-top-and-bottom-margins-in-pyplot-horizontal-stacked-bar-chart-barh/30861795#30861795
from random import random
Y = ['A', 'B', 'C', 'D', 'E','F','G','H','I','J', 'K']
y_pos = np.arange(len(Y))
data = [(r, 1-r) for r in [random() for i in range(len(Y))]]
print data
a,b = zip(*data)

fig = plt.figure(figsize=(8,16))
ax = fig.add_subplot(111)
ax.barh(y_pos, a, color='b', align='center')
ax.barh(y_pos, b, left=a, color='r', align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(Y, size=16)
ax.set_xlabel('X label', size=20)
plt.ylim(min(y_pos)-1, max(y_pos)+1)
plt.xlim(0,1)

plt.show()
plt.close('all')
###############################################################################


###############################################################################
######################## MULTIPLOTS (MULTIPLE PLOTS) ##########################
###############################################################################
# Customizing Location of Subplot Using GridSpec
# There are multiple strategies to specify the shape of the grid, and the placement of individual plots on that grid.
# (a) plt.subplot(): the main subplot function in matplotlib.
# (b) plt.subplot2grid(): a helper function that is similar to “pyplot.subplot” but uses 0-based indexing and let subplot to occupy multiple cells.

# For example, the following two lines are equivalent:
# Note that, in subplot the index starts from 1, while in subplot2grid the index starts from 0.
ax = plt.subplot2grid((2,2),(0, 0))  # create a 2x2 grid, and give me a reference to the plot located at (0,0).
ax = plt.subplot(2,2,1)  # create a 2x2 grid, and give me a reference to the first plot (the order is specified by subplot function)

# To create a subplot that spans multiple cells:
ax2 = plt.subplot2grid((3,3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)

# Note: subplot2grid() does not have a parameter to set figure size. So if you want to change figure size, you first should generate a canvas object using plt.figure(), and then draw a grid on it.
# But in general, you do not need to create the canvas object for subplot2grid() to work.
plt.figure(figsize=[10,8])
ax1 = plt.subplot2grid((3,3), (0,0))
plt.draw()
plt.close('all')


### Example of using subplot2grid() to create a multi-plot:
plt.figure(figsize=[10,8])  # is not necessary unless you want to change the figure size.
ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3,3), (2, 0))
ax5 = plt.subplot2grid((3,3), (2, 1))
plt.draw()
plt.close()


### Example of using GridSpec() to create the same plot as above:
# subplot2grid() creates an instance of gridspec object. You can also import that gridspec module
# A gridspec instance provides array-like (2d or 1d) indexing that returns the SubplotSpec instance.
# For, SubplotSpec that spans multiple cells, use slice.

import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1,:-1])
ax3 = plt.subplot(gs[1:, -1])
ax4 = plt.subplot(gs[-1,0])
ax5 = plt.subplot(gs[-1,-2])

plt.draw()
plt.close()


### MODIFYING INDIVIDUAL SUB-PLOTS
# When a GridSpec is explicitly used, you can adjust the layout parameters of subplots that are created from the gridspec.
gs1 = gridspec.GridSpec(3, 3)
gs1.update(left=0.05, right=0.48, wspace=0.05)
ax1 = plt.subplot(gs1[:-1, :])
ax2 = plt.subplot(gs1[-1, :-1])
ax3 = plt.subplot(gs1[-1, -1])

gs2 = gridspec.GridSpec(3, 3)
gs2.update(left=0.55, right=0.98, hspace=0.05)
ax4 = plt.subplot(gs2[:, :-1])
ax5 = plt.subplot(gs2[:-1, -1])
ax6 = plt.subplot(gs2[-1, -1])

plt.draw()
plt.close()


# You can create GridSpec from the SubplotSpec, in which case its layout parameters are set to that of the location of the given SubplotSpec.
gs0 = gridspec.GridSpec(1, 2)

gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])
gs01 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[1])
plt.draw()
plt.close()


### GridSpec with Varying Cell Sizes
gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1,2],
                       height_ratios=[4,1]
                       )

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])

plt.draw()
plt.close()


### WORKING WITH MUTIPLE FIGURE AND AXES
# The subplot() command specifies numrows, numcols, fignum where fignum ranges from 1 to numrows*numcols.
# The commas in the subplot command are optional if numrows*numcols < 10 (see below).
# The function gca() returns the current axes (a matplotlib.axes.Axes instance), and gcf() returns the current figure
import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)  # same as: plt.subplot(2,1,1)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)  # same as: plt.subplot(2,1,2)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.draw()
plt.close()


#######################################
### MULTIPLOT USING A PLOTTING FUNCTION
# We can have a separate plotting function, and send each subplot to that function for plotting.
# The handle to send each plot is the ax parameter:

def plot_hist_cauchy(x, pltTitle, dist, fileName="", ax=None):

    x[x < 0.5] = 0.5  # severely underperforming systems
    x[x > 2] = 2  # happens in winter in shade-y systems

    ### plot the histogram of the data
    if (ax is None):
        fig, ax = plt.subplots(figsize=(8,6))

    n, bins, patches = ax.hist(x, 50, normed=1, facecolor='green', alpha=0.75)


    ### add a 'best fit' line
    if (dist.lower()=='cauchy'):
        locParam = np.median(x)
        scaleParam = (np.percentile(x,0.75)-np.percentile(x,0.25))/2
        y = stats.cauchy.pdf(x=bins, loc=locParam, scale=scaleParam)  # cauchy dist

    # Note: can add other distributions here. It is safe to fall back to Gaussian if none is provided.
    else:
        mu, sigma = np.mean(x), np.std(x)
        y = mlab.normpdf( bins, mu, sigma)  # normal dist

    l = ax.plot(bins, y, 'r--', linewidth=1)


    ax.set_xlabel("""Actual / Expected""")  #
    ax.set_ylabel('Probability')
    ax.axis([0, 2, 0, math.ceil(max(y))])
    ax.grid(True)
    plt.suptitle(r'$\mathrm{Histogram\ of\ %s:}\ \mu=%s,\ \sigma=%s,\ N=%s$' %( pltTitle, str(round(np.mean(x),3)), str(round(np.std(x),3)), str(len(x)) ) , fontsize=12)
    plt.draw()

    if fileName:
        plt.savefig(localPath+'/Results/plots/'+fileName+'.png')
        plt.close('all')


plt.figure(figsize=[13,10])
pltTitle = 'Normally distributed numbers'
dist = 'normal'
for i in range(12):
    x = np.random.normal(0,1,500)
    axi = plt.subplot2grid((4,3), (i/3,i%3))
    plot_hist_cauchy(x, pltTitle, dist, "", axi)
#######################################

## More axes examples:
# http://matplotlib.org/examples/pylab_examples/axes_demo.html
## More multiple plots examples:
# http://matplotlib.org/examples/pylab_examples/subplots_demo.html#pylab-examples-subplots-demo
###############################################################################

###############################################################################
### SUBPLOT WITH subplot2grid layout parameters
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

plt.figure(figsize=[10,8])  # is not necessary unless you want to change the figure size.
ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)  # the shape parameter (2,3) will dictate plot dimension ratios
ax2 = plt.subplot2grid((2,3), (1,0), colspan=2)  # this will occupy both (1,0) and (1,1)
ax3 = plt.subplot2grid((2,3), (1,2), colspan=1)  # hence the (1,2) index for this

# DENSITY: the histogram of the data (the area of each bar is equal to its probability
n, bins, patches = ax1.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
ax1.set_xlabel('Smarts')
ax1.set_ylabel('Probability')
ax1.set_title('Histogram of IQ')
ax1.text(60, .025, r'$\mu=100,\ \sigma=15$')
ax1.axis([40, 160, 0, 0.03])
ax1.grid(True)

# FREQUENCY: remove normed=1 if you want to plot frequeny
n, bins, patches = ax2.hist(x, 50, facecolor='g', alpha=0.75)
ax2.set_xlabel('Smarts')
ax2.set_ylabel('Frequency')
ax2.set_title('Histogram of IQ')
ax2.text(60, .025, r'$\mu=100,\ \sigma=15$')
ax2.axis([40, 160, 0, 800])
ax2.grid(True)

# FREQUENCY: remove normed=1 if you want to plot frequeny
n, bins, patches = ax3.hist(x, 50, facecolor='g', alpha=0.75)
ax3.set_xlabel('Smarts')
ax3.set_ylabel('Frequency')
ax3.set_title('Histogram of IQ')
ax3.text(60, .025, r'$\mu=100,\ \sigma=15$')
ax3.axis([40, 160, 0, 800])
ax3.grid(True)

plt.draw()
plt.close('all')
###############################################################################
