import matplotlib.pyplot as plt
plt.ion()  # turn on interactive plotting
plt.style.use('ggplot')  # make matplotlib appearance similar to ggplot


# There are multiple ways to specify colors for a plot.

# (a) Use colormaps
# (b) Use dynamic rc settings
# (c) Use Style Sheets
# (d) Use external packages such as palettable, mpltools, etc.
#######################################


###############################################################################
################################ COLOR MAP ####################################
###############################################################################
### INTRO colormap
# The default color map that comes with matplotlib:
# http://matplotlib.org/examples/color/colormaps_reference.html


"""
WHAT IS A COLORMAP
http://matplotlib.org/users/colormaps.html
A colormap is a palette of colors assembled to be used in a single plot. colormaps are represented by three dimensions:
Lightness, Red-Green, Yellow-Blue


HOW TO USE COLORMAPS
There are multiple elements to a mpl colormap

matplotlib.get_cmap()
    This is the main function to choose the colormap with which the plot will be drawn.
    For a list of colormaps, see:
    http://matplotlib.org/users/colormaps.html

    The result object is a cmap object. If you call a cmap object with an index for example cmap(1),
    it returns the (Lightness, Red, Green, Blue) value of the colormap in that index position.
    The

colors.Normalize()

colors.LinearSegmentedColormap.from_list
    Creates a colormap object from a list containing (L,R,G,B) values for every element in the colormap.

scalarMap.to_rgba()
    To get the RGB values from the ScalarMap object above:

matplotlib.cm.ScalarMappable(norm=None, cmap=None)
    This is a mixin class to support scalar data to RGBA mapping. The ScalarMappable makes use of data normalization before returning RGBA colors from the given colormap.

"""

### Example of how to specify a colormap:
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import random
Z = random.random((50,50))   # Test data
plt.imshow(Z, cmap=plt.get_cmap("Spectral"), interpolation='nearest')
plt.show()
plt.close('all')

### List of colormaps that ship with matplotlib:
print plt.cm.datad.keys()
###############################################################################


###############################################################################
"""
Reference for colormaps included with Matplotlib.

This reference example shows all colormaps included with Matplotlib. Note that
any colormap listed here can be reversed by appending "_r" (e.g., "pink_r").
These colormaps are divided into the following categories:

Sequential:
    These colormaps are approximately monochromatic colormaps varying smoothly
    between two color tones---usually from low saturation (e.g. white) to high
    saturation (e.g. a bright blue). Sequential colormaps are ideal for
    representing most scientific data since they show a clear progression from
    low-to-high values.

Diverging:
    These colormaps have a median value (usually light in color) and vary
    smoothly to two different color tones at high and low values. Diverging
    colormaps are ideal when your data has a median value that is significant
    (e.g.  0, such that positive and negative values are represented by
    different colors of the colormap).

Qualitative:
    These colormaps vary rapidly in color. Qualitative colormaps are useful for
    choosing a set of discrete colors. For example::

        color_list = plt.cm.Set3(np.linspace(0, 1, 12))

    gives a list of RGB colors that are good for plotting a series of lines on
    a dark background.

Miscellaneous:
    Colormaps that don't fit into the categories above.

"""
###############################################################################

###############################################################################
### EXISTING COLOR MAPS
# http://matplotlib.org/examples/color/colormaps_reference.html

import numpy as np
import matplotlib.pyplot as plt

cmaps = [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]


nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap_category, cmap_list):
    fig, axes = plt.subplots(nrows=nrows)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()


### PLOT THE COLOR MAPS
for cmap_category, cmap_list in cmaps:
    plot_color_gradients(cmap_category, cmap_list)

plt.show()
plt.close('all')
###############################################################################


###############################################################################
### HOW TO NORMALIZE A COLORMAP
### USE matplotlib.colors
# http://stackoverflow.com/questions/8931268/using-colormaps-to-set-color-of-line-in-matplotlib

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

# define some random data that emulates your indeded code:
NCURVES = 10
np.random.seed(101)
curves = [np.array(range(NCURVES))*(i+1) for i in range(NCURVES)]
values = range(NCURVES)

jet = cm = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
print scalarMap.get_clim()

fig = plt.figure()
ax = fig.add_subplot(111)

lines = []
for idx in range(len(curves)):
    line = curves[idx]
    colorVal = scalarMap.to_rgba(values[idx])
    colorText = (
        'color: (%4.2f,%4.2f,%4.2f)'%(colorVal[0],colorVal[1],colorVal[2])
        )
    retLine, = ax.plot(line,
                       color=colorVal,
                       label=colorText)
    lines.append(retLine)

# legend and grid
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper left')
# ax.grid()

plt.show()
plt.close('all')
###############################################################################

###############################################################################
### HOW TO TRUNCATE AN EXISTING COLORMAP (BY SPECIFYING ALL L,R,G,B VALUES)
# http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib

# There are two key functions here:
# matplotlib.get_cmap(<cmap name>)   : returns a colormap object from its name
# matplotlib.colors.LinearSegmentedColormap.from_list()    : creates a new colormap object from a list of (L,R,G,B) values

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import numpy as np

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

arr = np.linspace(0, 50, 100).reshape((10, 10))
fig, ax = plt.subplots(ncols=2)

cmap = plt.get_cmap('Blues')
# Alternatively, use the following syntax to get the colormap
# cmap = mpl.cm.Blues
new_cmap = truncate_colormap(cmap, 0.2, 0.8)
ax[0].imshow(arr, interpolation='nearest', cmap=cmap)
ax[1].imshow(arr, interpolation='nearest', cmap=new_cmap)
plt.show()
plt.close('all')
###############################################################################

###############################################################################
### HOW TO TRUNCATE AN EXISTING COLORMAP (BY SPECIFYING ALL L,R,G,B VALUES)
# http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
# First, get a a dictionary of all the colors that make up the colormap:
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
cdict = cm.get_cmap('spectral_r')._segmentdata

# If you want to change the beginning and end colors:
cdict['red'][0] = (0, 0.5, 0.5) # x=0 for bottom color in colormap
cdict['blue'][0] = (0, 0.5, 0.5) # y=0.5 gray
cdict['green'][0] = (0, 0.5, 0.5) # y1=y for simple interpolation
cdict['red'][-1] = (1, 0.5, 0.5) # x=1 for top color in colormap
cdict['blue'][-1] = (1, 0.5, 0.5)
cdict['green'][-1] = (1, 0.5, 0.5)

my_cmap = LinearSegmentedColormap('name', cdict)

# Then use this cmap in your plotting function.
###############################################################################

###############################################################################
### USE COLOR CYCLE TO CHANGE DEFAULT COLOR CYCLE FOR PLOTTING
# http://matplotlib.org/examples/color/color_cycle_demo.html
"""
Demo of custom property-cycle settings to control colors and such
for multi-line plots.

This example demonstrates two different APIs:

    1. Setting the default rc-parameter specifying the property cycle.
       This affects all subsequent axes (but not axes already created).
    2. Setting the property cycle for a specific axes. This only
       affects a single axes.
"""
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi)
offsets = np.linspace(0, 2*np.pi, 4, endpoint=False)
# Create array with shifted-sine curve along each column
yy = np.transpose([np.sin(x + phi) for phi in offsets])

plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
                           cycler('linestyle', ['-', '--', ':', '-.'])))
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(yy)
ax0.set_title('Set default color cycle to rgby')

ax1.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
                   cycler('lw', [1, 2, 3, 4]))
ax1.plot(yy)
ax1.set_title('Set axes color cycle to cmyk')

# Tweak spacing between subplots to prevent labels from overlapping
plt.subplots_adjust(hspace=0.3)
plt.show()
plt.close('all')
###############################################################################

# Tips on how to choose the colormap:
# https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/

###############################################################################
############################# EXTERNAL PACKAGES ###############################
###############################################################################

###############################################################################
### Use palettable package to manage colormaps
# The package Palettable makes colorbrewer colors accessible.
# https://jiffyclub.github.io/palettable/
print plt.style.available

# You can create custom styles and use them by calling style.use with the path or URL to the style sheet. Alternatively, if you add your <style-name>.mplstyle file to mpl_configdir/stylelib, you can reuse your custom style sheet with a call to style.use(<style-name>).
# See the following link for more:
# http://matplotlib.org/users/style_sheets.html#style-sheets
###############################################################################

###############################################################################
### USE mpltools PACKAGE
# http://tonysyu.github.io/mpltools/auto_examples/color/plot_cycle_cmap.html
###############################################################################


###############################################################################
############################## USE RC SETTINGS ################################
###############################################################################

#######################################
### (b) Use dynamic rc settings

# http://matplotlib.org/users/customizing.html

# matplotlib uses matplotlibrc configuration files to customize all kinds of properties,
# which we call rc settings or rc parameters.
# You can control the defaults of almost every property in matplotlib:
# figure size and dpi, line width, color and style, axes, axis and grid properties, text and font properties and so on.
# rc params can be stored in a file, and transferred across different machines for consistency.

# Find the current location of the rc file
import matplotlib as mpl
mpl.matplotlib_fname()


### DYNAMIC rc SETTINGS
# You can also dynamically change the default rc settings in a python script or
# interactively from the python shell.
# All of the rc settings are stored in a dictionary-like variable called matplotlib.rcParams,
# which is global to the matplotlib package.
# rcParams can be modified directly, for example:
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.color'] = 'r'

# matplotlib also provides a couple of convenience functions for modifying rc settings.
# The matplotlib.rc() command can be used to modify multiple settings in a single group at once, using keyword arguments:
import matplotlib as mpl
mpl.rc('lines', linewidth=2, color='r')

# matplotlib.rcdefaults() command will restore the standard matplotlib default settings.
#######################################

#######################################
### USE RC PARAMS WITH COLOR_CYCLE PARAMETER
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# Set the default color cycle
mpl.rcParams['axes.color_cycle'] = ['r', 'k', 'c']  # use color names
cmap = plt.get_cmap('viridis')
mpl.rcParams['axes.color_cycle'] = cmap([0,0.5,1.0])  # or use colormap values

# Alternately, we could use rc:
# mpl.rc('axes', color_cycle=['r','k','c'])

x = np.linspace(0, 20, 100)

fig, axes = plt.subplots(nrows=2)

for i in range(10):
    axes[0].plot(x, i * (x - 10)**2)

for i in range(10):
    axes[1].plot(x, i * np.cos(x))

plt.show()
plt.close()
#######################################

#######################################
### Technical details of how rc parameters are implemented (extended read)
# http://matplotlib.org/api/matplotlib_configuration_api.html?highlight=rcparams#matplotlib.rcParams

### RcParams is a class in matplotlib package.
# It implements a dictionary object that holds all default values for plotting with matplotlib.
# rcParams is an object instance of RcParams in matplotlib package.
# It implements a dictionary object that stores default matplotlib parameters.

### rc_params() is a function in matplotlib package.
# It returns a matplotlib.RcParams() instance from the default matplotlib rc file.
import matplotlib as mpl
print mpl.rc_params()
# or print the dictionary object:
print mpl.rcParams
# you can either access it through matplotlib itself, or through matplotlib.PyPlot
print plt.rcParams

### plt.rc() is a function in matplotlib package.
# It sets the current rc parameters.
mpl.rc('lines', linewidth=2, color='r')
plt.plot(range(5), range(5))

# rc() sets multiple parameters at once. The code above is equivalent to:
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'r'
#######################################


#######################################
### USING RC SETTINGS TO CHANGE PLOT SETTINGS CYCLE
# http://stackoverflow.com/questions/9397944/default-color-cycle-with-matplotlib
# By default, matplotlib will cycle through its default line colors when plotting multiple lines on the same plot.
# We can change that default cycle using rcParams dictionary.

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Set the default color cycle
mpl.rcParams['axes.color_cycle'] = ['r', 'k', 'c']

# Alternately, we could use rc:
# mpl.rc('axes', color_cycle=['r','k','c'])

x = np.linspace(0, 20, 100)

fig, axes = plt.subplots(nrows=2)

for i in range(10):
    axes[0].plot(x, i * (x - 10)**2)

for i in range(10):
    axes[1].plot(x, i * np.cos(x))

plt.show()
plt.close('all')
#######################################


#######################################
### EXAMPLE OF USING RC PARAMS TO CREATE PAPER-READY PLOTS WITH matplotlib
# http://bikulov.org/blog/2013/10/03/creation-of-paper-ready-plots-with-matlotlib/

# NOTE: in this implementation, we have included the rc params in a function,
# which we can call when we want to swithc to presentation mode.
# Another option is to use style sheets:
# http://matplotlib.org/users/style_sheets.html
# In that case, we can save the style parameters below to a style file, and
# call that style whenever we want to switch to presentation mode.

import matplotlib.pyplot as plt

# set global settings
def init_plotting():
    plt.rcParams['figure.figsize'] = (8, 3)
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['axes.linewidth'] = 1

    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

init_plotting()

# plotting example data
from math import sin
from math import cos

x = [0.31415*xi for xi in xrange(0,10)]
y1 = [sin(xi) for xi in x]
y2 = [cos(xi + 0.5) for xi in x]
y3 = [cos(xi + 0.5) + sin(xi) for xi in x]

# begin subplots region
plt.subplot(121)
plt.gca().margins(0.1, 0.1)
plt.plot(x, y1, linestyle='-', marker='.', linewidth=1, color='r', label='sin')
plt.plot(x, y2, linestyle='.', marker='o', linewidth=1, color='b', label='cos')

plt.gca().annotate(u'point $\\frac{\\tau}{2}$', xy=(x[2], y1[2]),  xycoords='data',
                xytext=(30, -10), textcoords='offset points', size=8,
                arrowprops=dict(arrowstyle='simple', fc='g', ec='none'))

plt.xlabel(u'x label')
plt.ylabel(u'y label')
plt.title(u'First plot title')

plt.gca().legend(bbox_to_anchor = (0.0, 0.1))

plt.subplot(122)
plt.gca().margins(0.1, 0.1)
plt.plot(x, y3, linestyle='--', marker='.', linewidth=1, color='g', label='sum')

plt.gca().annotate(u'$y_x$', xy=(x[2], y3[2]),  xycoords='data',
                xytext=(-30, -20), textcoords='offset points', size=8,
                arrowprops=dict(arrowstyle='simple', fc='orange', ec='none'))

plt.xlabel(u'x label')
plt.ylabel(u'y label')
plt.title(u'Second plot title')

plt.gca().legend(bbox_to_anchor = (0.0, 0.1))
# end subplots region

# output resulting plot to file
plt.tight_layout()
plt.savefig('graph.png')
###############################################################################

###############################################################################
############################### STYLE SHEETS ##################################
###############################################################################

#######################################
### (c) Use Style Sheets
# http://matplotlib.org/users/style_sheets.html

# Style Sheets are anothe way of customizing matplotlib.
# The style package adds support for easy-to-switch plotting “styles” with the same parameters as a matplotlibrc file.
# Style sheets provide a means for more specific and/or temporary configuration modifications,
# but in a repeatable and well-ordered manner.
# A style sheet is a file with the same syntax as the matplotlibrc file, and when applied, it will override the matplotlibrc.

# There are a number of pre-defined styles provided by matplotlib. For example:
plt.style.use('ggplot')

# List all available styles:
print plt.style.available

# You can mix and match style sheets:
# So you can have a style sheet that customizes colors and a separate style sheet that alters element sizes for presentations.

# To create, save, and activate a style:
# http://matplotlib.org/users/style_sheets.html
#######################################

###############################################################################



