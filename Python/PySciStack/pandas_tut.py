###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: April 26, 2015
# summary: pandas playground
# other resources:
# http://pandas.pydata.org/pandas-docs/stable/tutorials.html
###############################################################################
#TODO: read the cookbook: http://pandas.pydata.org/pandas-docs/stable/cookbook.html

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


# Specify file locations
plotPath = '/Users/amirkavousian/Documents/Py_Codes/Plots'
resultsPath = '/Users/amirkavousian/Documents/Py_Codes/Results'

###############################################################################
################################# PANDAS ######################################
###############################################################################
# Remember that numpy arrays have a fixed length and are
# based on efficient C data structures. As a result, they are very fast.

###############################################################################
### Data Structure #1: SERIES
# A Series is a one-dimensional object similar to an array, list, or column in a table.
# It will assign a labeled index to each item in the Series.
# By default, each item will receive an index label from 0 to N, where N is the length of the Series minus one.

# Create a Series by passing a list of values, leaving the indexing to pandas.
s = pd.Series([1,3,5,np.nan,6,8])

# pandas series are internally np.arrays
pd.Series([1,2,3])
pd.Series([1,2,3]).values
np.array([1,2,3])

# Series do not need to have homogenous types.
s = pd.Series([7, 'Heisenberg', 3.14, -1789710578, 'Happy Eating!'])

# Or, you can specify an index when creating Series
s = pd.Series([7, 'Heisenberg', 3.14, -1789710578, 'Happy Eating!'],
              index=['A', 'Z', 'C', 'Y', 'E'])
s

# Convert a dictionary to Series
d = {'Chicago': 1000, 'New York': 1300, 'Portland': 900, 'San Francisco': 1100,
     'Austin': 450, 'Boston': None}
cities = pd.Series(d)

### IDEXING & SLICING SERIES
# Similar to DataFrames, use index names to select Series items
cities['Chicago']
# boolean indexing
cities[cities < 1000]
'San Francisco' in cities

# math operations
cities / 3
np.square(cities)

# match operations between two Series:
# the operations will be done on matching indexes. Non-matching indexes get NaN or NULL values.
cities[['Chicago', 'New York', 'Portland']] + cities[['Austin', 'New York']]

# null checking
cities.notnull()
cities.isnull()
cities[cities.isnull()]
###############################################################################


###############################################################################
### Data Structure #2: TIME SERIES

rng = pd.date_range('1/1/2012', periods=100, freq='S')

# resample time series (modify its frequency)
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min', how='sum')

# time-zone representation (tz_localize, tz_convert)
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts_utc = ts.tz_localize('UTC')
ts_utc.tz_convert('US/Eastern')

# converting between time span representations
rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ps = ts.to_period()
ps.to_timestamp()

# convert a quarterly frequency with year ending in November to 9am of the end of the month following the quarter end.
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
ts.head()


###
datetime.time(18)
dates = pd.date_range('2000-01-01', periods=5)
dates.to_period(freq='M').to_timestamp()

### TIME-SERIES ANALYSIS
# Example: a random time series sequence
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
# Note: you can either show or savefig at a time.
plt.show()
plt.savefig( '/Users/amirkavousian/Documents/myfig.png' )

# Example: a scatterplot
x = np.random.random(10)
y = np.random.random(10)
plt.plot(x,y, '.')
plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# To get coefficient of determination (r_squared)
print "r-squared:", r_value**2
###############################################################################


###############################################################################
### Data Structure #3: DATAFRAME
# http://pandas.pydata.org/pandas-docs/stable/cookbook.html#cookbook

### CREATING DATA FRAMES: 2 MAIN METHODS:
### Method 1: Define a DataFrame by column names (similar to R data.frame definition):
# Pass a dictionary of lists to DataFrame() function.
df = pd.DataFrame({'A': [1, 2.1, np.nan, 4.7, 5.6, 6.8], \
                   'B': [.25, np.nan, np.nan, 4, 12.2, 14.4]})

# Another example of creating a df using a dictionary
df2 = pd.DataFrame({ 'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D' : np.array([3] * 4, dtype='int32'),
                    'E' : pd.Categorical(["test", "train", "test", "train"]),
                    'F' : 'foo' })


### Method 2: Define a DataFrame by index and column names (similar to R matrix definition):
# Pass a numpy array plus lists for indexes and column names.
df = pd.DataFrame(np.random.randn(5, 3),\
                  index=['a', 'c', 'e', 'f', 'h'],\
                  columns=['one', 'two', 'three'])
df['four'] = 'bar'
df['five'] = df['one'] > 0

# Another example of creating a DataFrame using a numpy array.
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4),
                  index=dates,
                  columns=list('ABCD'))

###############################################################################
### Data Structure #4: PANEL
rng = pd.date_range('1/1/2013', periods=100, freq='D')
data = np.random.randn(100, 4)
cols = ['A','B','C','D']
df1, df2, df3 = pd.DataFrame(data, rng, cols), pd.DataFrame(data, rng, cols), pd.DataFrame(data, rng, cols)
pf = pd.Panel({'df1':df1, 'df2':df2, 'df3':df3});pf
pf.ndim
pf.shape
pf.dtypes
pf.size

# Transpose re-assignment
pf2 = pf.transpose(2,0,1)
pf['E'] = pd.DataFrame(data, rng, cols)
###############################################################################

###############################################################################
################ DATA FRAME SUMMARY STATS / BASIC MANIPULATION ################
###############################################################################
# Expand the data frame. pandas automatically pads the columns.
df2 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
df2['one']

# Check for null values
pd.isnull(df2['one'])
df2['one'].isnull()
pd.notnull(df2['four'])
df2['four'].notnull()

#
df = pd.DataFrame(np.random.randn(1000,4),
                  index=pd.date_range('20000101',periods=1000),
                  columns=list('ABCD'))
df.tail

# Display the index,columns, and the underlying numpy data
df.index
df.columns
df.values

# quick statistic summary of your data
df.describe()

# Transposing a data frame
df.T

# Sorting by an axis
df.sort_index(axis=1, ascending=False)

# Sorting by values
df.sort(columns='B')


###############################################################################
#################### INDEXING AND SLICING DATA FRAMES #########################
###############################################################################

# NOTE: Standard Python and NumPy slicing and selection operators work in pandas context as well.
# However, for production code, it is recommended to use the optimized pandas data access methods, .at, .iat, .loc, .iloc and .ix.

# Select a column by its name. It yields a single column, which yields a Series, equivalent to df.A
df['A']

# Selecting via [] which slices the rows.
df[0:3]

### loc: location-based selection and label-based selection
df.loc[dates[0]]

# Selecting on a multi-axis by label
df.loc[:,['A', 'B']]

# Showing label slicing, both endpoints are included
df.loc['20130102':'20130104',['A','B']]

# Reduction in the dimensions of the returned object
df.loc['20130102',['A','B']]

# Get a scalar value
df.loc[dates[0],'A']

# For getting fast access to a scalar (equiv to the prior method)
df.at[dates[0],'A']


### iloc: index-based selection
# Select via the position of passed integers
df.iloc[3]

# By integer slices, acting similar to numpy/python
df.iloc[3:5,0:2]

# By lists of integer position locations, similar to the numpy/python style
df.iloc[[1,2,4],[0,2]]

# For slicing rows explicitly
df.iloc[1:3,:]
# For slicing columns explicitly
df.iloc[:,1:3]

# For getting a value explicitly
df.iloc[1,1]

# For getting fast access to a scalar (equiv to the prior method)
# df.iat[1,1]

### Boolean indexing
# Using a single column's values to select data
df[df.A > 0]
df[df > 0]

# Using the isin() method for filtering:
df2 = df.copy()
df2['E']=['one', 'one','two','three','four','three']

df2[df2['E'].isin(['two','four'])]

### ix: primarily label-based selection, with location-based fallback.
df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]});
df.ix[df.AAA >= 5,'BBB'] = -1;
df
df.ix[df.AAA >= 5,['BBB','CCC']] = 555;
df


### ASSIGNING VALUES TO DATA FRAME USING INDEXING
# http://stackoverflow.com/questions/17557650/edit-pandas-dataframe-using-indexes


### FINDING THE INDEX OF nan
# http://stackoverflow.com/questions/14016247/python-find-integer-index-of-rows-with-nan-in-pandas
na_readings = df2[str(df2.columns[-1])][np.isnan(df2[str(df2.columns[-1])])].index.tolist()
df2.iloc[na_readings, [-3,-2,-1]]

### DROPPING nan
# http://stackoverflow.com/questions/14991195/how-to-remove-rows-with-null-values-from-kth-column-onward-in-python


# pd.where() works like ifelse() function in R.
df_mask = pd.DataFrame({'AAA' : [True] * 4, 'BBB' : [False] * 4,'CCC' : [True,False] * 2})
df.where(df_mask, -1000)

# Split a dataframe with a boolean criterion
dflow = df[df.AAA <= 5]
dflow
dfhigh = df[df.AAA > 5]
dfhigh

# Select with multi-column criteria
df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]});
newseries = df.loc[(df['BBB'] < 25) & (df['CCC'] >= -40), 'AAA'];
newseries

# Select values closest to a give value using argsort()
aValue = 43.0
df.ix[(df.CCC-aValue).abs().argsort()]

# Dynamically reduce a list of criteria
Crit1 = df.AAA <= 5.5
Crit2 = df.BBB == 10.0
Crit3 = df.CCC > -40.0
AllCrit = Crit1 & Crit2 & Crit3
CritList = [Crit1, Crit2, Crit3]
# Use reduce function to summarize a list of criteria
AllCrit2 = functools.reduce(lambda x,y: x & y, CritList)
df[AllCrit]

# Select using row labels and isin() function
df[(df.AAA <= 6) & (df.index.isin([0,2,4]))]

# Use loc for label-oriented slicing and iloc positional slicing
data = {'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]}
df = pd.DataFrame(data=data, index=['foo','bar','boo','kar']);
df


# data_df['New_AE_Detected'][data_df['D_ASSET_FKEY']==ak + j] = 1
# data_df.loc[data_df['D_ASSET_FKEY']==ak]
# idx = data_df[data_df['D_ASSET_FKEY'] == ak].index.tolist()
# data_df.loc[idx]
# data_df.loc[data_df['D_ASSET_FKEY']==ak]
# data_df.loc[data_df[data_df['D_ASSET_FKEY']==ak].index.tolist(), 'New_AE_Detected'] = 1


# Read this:
# http://stackoverflow.com/questions/21800169/python-pandas-get-index-of-rows-which-column-matches-certain-value
# http://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
# http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

## There are 2 explicit slicing methods, with a third general case
#  Positional-oriented (Python slicing style : exclusive of end) --> iloc
#  Label-oriented (Non-Python slicing style : inclusive of end) --> loc
#  General (Either slicing style : depends on if the slice contains labels or positions) --> ix

# Label
df.loc['bar':'kar'] # Label

# Generic
df.ix[0:3] #Same as .iloc[0:3]
df.ix['bar':'kar'] #Same as .loc['bar':'kar']

# It can get ambiguous when an index consists of integers with a non-zero start or non-unit increment.
df2 = pd.DataFrame(data=data,index=[1,2,3,4]);  # Note index starts at 1.
df2.iloc[1:3]  # Position-oriented, exclusive of end
df2.loc[1:3]  # Label-oriented, inclusive of end
df2.ix[1:3] #General, will mimic loc (label-oriented, inclusive of end)
df2.ix[0:3] #General, will mimic iloc (position-oriented, exclusive of end), as loc[0:3] would raise a KeyError

# Take complement of a mask by using ~ operator
df = pd.DataFrame({'AAA' : [4,5,6,7],
                   'BBB' : [10,20,30,40],
                   'CCC' : [100,50,-30,-50]}); df
df[~((df.AAA <= 6) & (df.index.isin([0,2,4])))]

### ADD COLUMNS USING APPLYMAP
# applymap applies a function to a DataFrame elementwise.
df = pd.DataFrame({'AAA' : [1,2,1,3],
                   'BBB' : [1,1,2,2],
                   'CCC' : [2,1,3,1]}); df
source_cols = df.columns
new_cols = [str(x) + "_cat" for x in source_cols]
df
categories = {1 : 'Alpha', 2 : 'Beta', 3 : 'Charlie' }
df[new_cols] = df[source_cols].applymap(categories.get); df


### DATAFRAME STATS OPERATIONS
# column means
df.mean()  # without specifying the dimension, by default it calculates column means
df.mean(0)  # the first dimension is the columns dimension
# row means
df.mean(1)  # the second dimension is rows


###############################################################################
############ MAP, APPLY, APPLYMAP FUNCTIONS FOR LOOP OPERATIONS ###############
###############################################################################
df = pd.DataFrame({'row'   : [0, 1, 2],
                   'One_X' : [1.1, 1.1, 1.1],
                   'One_Y' : [1.2, 1.2, 1.2],
                   'Two_X' : [1.11, 1.11, 1.11],
                   'Two_Y' : [1.22, np.NaN, 1.25]});

# map() function is similar to R apply() function. It works on arrays (not multiple columns).
df['Two_Y'].dropna().map(lambda x : 'map_' + str(x))

# apply() function is similar to R lapply() function.
# It works on each column (axis=0) or row (axis=1) of a data frame.
# By default, it operates on columns.
df.ix[:, ['One_X','Two_X']].apply(np.sqrt)
df.ix[:, ['One_X','Two_X']].apply(np.sum)
df.ix[:, ['One_X','Two_X']].apply(np.sum, axis=0)
df.ix[:, ['One_X','Two_X']].apply(np.sum, axis=1)

# applymap() applies a function to each element of the DataFrame.
df.applymap(lambda x: x + 5)

#
df = pd.DataFrame(data={"A" : [1, 2],
                        "B" : [1.2, 1.3]})
df["C"] = df["A"]+df["B"]
df["D"] = df["A"] * 4


# Create multiple columns from existing columns of a data frame.
df4 = df.copy()
def two_three_strings(x):
   return x*2, x*3
df4['twice'], df4['thrice'] = zip(*df4['A'].map(two_three_strings))

# Create multiple columns from multiple input columns of a df.
def int_float_squares(series):
   return pd.Series({'int_sq' : series['One_X']**2,
                     'flt_sq' : series['One_Y']**2})
df.apply(int_float_squares, axis = 1)


###############################################################################
### MULTI-INDEXING
df = pd.DataFrame({'row'   : [0,1,2],
                   'One_X' : [1.1,1.1,1.1],
                   'One_Y' : [1.2,1.2,1.2],
                   'Two_X' : [1.11,1.11,1.11],
                   'Two_Y' : [1.22,1.22,1.22]});
df
df = df.set_index('row'); df
df.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in df.columns]); df
df = df.stack(0).reset_index(1); df
df.columns = ['Sample','All_X','All_Y']; df


### ARITHMETIC
cols = pd.MultiIndex.from_tuples([ (x,y) for x in ['A','B','C'] for y in ['O','I']])
df = pd.DataFrame(np.random.randn(2,6), index=['n','m'], columns=cols); df
df = df.div(df['C'],level=1); df


###
nameCounts = namesDF.groupby(['Year', 'Gender']).count()

idx = pd.IndexSlice

nameCounts.loc[idx[2007, 'M']]
nameCounts.loc[idx[2007, :]]

nameCounts.loc[(2007, slice(None, None, None))]
nameCounts.loc[(slice(None, None, None), 'M'), slice(None)]

nameCounts.loc[2007, slice(None)]

# More on multi-indexing:
# http://pandas.pydata.org/pandas-docs/stable/advanced.html
# http://stackoverflow.com/questions/10175068/select-data-at-a-particular-level-from-a-multiindex
###############################################################################

###############################################################################
###############################################################################
###############################################################################

###############################################################################
### SLICING
## Slicing a multi-index with xs, method 1
coords = [('AA','one'),('AA','six'),('BB','one'),('BB','two'),('BB','six')]
index = pd.MultiIndex.from_tuples(coords)
df = pd.DataFrame([11,22,33,44,55], index, ['MyData']); df
# Slice through the first level
df.xs('BB', level=0, axis=0)  #Note : level and axis are optional, and default to zero
# Slice through the second level
df.xs('six',level=1,axis=0)

## Slicing a multi-index with xs, method 2
index = list(itertools.product(['Ada','Quinn','Violet'], ['Comp','Math','Sci']))
headr = list(itertools.product(['Exams','Labs'],['I','II']))
indx = pd.MultiIndex.from_tuples(index,names=['Student','Course'])
cols = pd.MultiIndex.from_tuples(headr) #Notice these are un-named
data = [[70+x+y+(x*y)%3 for x in range(4)] for y in range(9)]
df = pd.DataFrame(data, indx, cols); df
All = slice(None)
df.loc['Violet']
df.loc[(All,'Math'),All]
df.loc[(slice('Ada','Quinn'),'Math'),All]
df.loc[(All,'Math'),('Exams')]
df.loc[(All,'Math'),(All,'II')]
###############################################################################

###############################################################################
### REINDEXING
# Reindexing allows you to change/add/delete the index on a specified axis.
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
###############################################################################

###############################################################################
############################ DATA MANIPULATION ################################
###############################################################################

###############################################################################
### SORTING
df.sort(('Labs', 'II'), ascending=False)
###############################################################################

###############################################################################
### MISSING DATA
# pandas primarily uses the value np.nan to represent missing data.
# np.nan is by default not included in computations.

# To drop any rows that have missing data
df1_noNA = df1.dropna(how='any')
# To fill in for missing data
fd1_fill = df1.fillna(value=5)
# To get the boolean mask where values are nan
pd.isnull(df1)

# In pandas, use NaN for missing data, the same way that NA is used in R.
# NaN and None (in object arrays) are considered missing by the isnull and notnull functions.
# inf and -inf are no longer considered missing by default.
# For datetime64[ns] types, NaT represents missing values.

## Fill-forward (use the next available value to fill in for missed values)
df = pd.DataFrame(np.random.randn(6,1), index=pd.date_range('2013-08-01', periods=6, freq='B'), columns=list('A'))
df.ix[3,'A'] = np.nan
df.reindex(df.index[::-1]).ffill()

# meterData['actual'][meterData['measurement'] != 'MEASURED'] = np.nan
# subset = meterData[meterData['measurement'] != 'MEASURED']
# subset['actual'] = np.nan
# df2['timestamp'] = pd.Timestamp('20120101')

### Indexing pandas DataFrame objects
df2.ix[['a','c','h'],['one','timestamp']] = np.nan
df2.get_dtype_counts()

df2.loc[['a','c','h'],['one','timestamp']] = np.nan
df2.ix[['a','c','h'],['one','timestamp']] = np.nan

df2.mean(1)
df2.mean()

df2['four'] = df2['four'].fillna('missing')


### BROADCASTING
# Operating with objects that have different dimensionality and need alignment
# pandas automatically broadcasts along the specified dimension.
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
df.sub(s, axis='index')


### APPLY
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())
###############################################################################


###############################################################################
### CATEGORICALS
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
df["grade"] = df["raw_grade"].astype("category")
# Rename the categories to more meaningful names. This is similar to setting levels in R factors.
df['grade'].cat.categories = ['very good', 'good', 'very bad']
# Reorder the categories and simultaneously add the missing categories
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
# sort (it is per order, not lexical order)
df.sort("grade")
# Grouping by a categorical column shows also empty categories
df.groupby("grade").size()

### groupby() using a custom function
d = {"my_label": pd.Series(['A','B','A','C','D','D','E'])}
df = pd.DataFrame(d)

def as_perc(value, total):
    return value/float(total)

def get_count(values):
    return len(values)

# .agg() applies the function to each unique element of the series.
grouped_count = df.groupby("my_label").my_label.agg(get_count)
data = grouped_count.apply(as_perc, total=df.my_label.count())
###############################################################################

###############################################################################
### STRING HANDLINE IN pandas
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()
###############################################################################


###############################################################################
##################### DATA FRAME RESHAPING AND PIVOTING #######################
###############################################################################

###############################################################################
### GROUPBY: it is similar to aggregate function in R:
# Splitting the data into groups based on some criteria
# Applying a function to each group independently
# Combining the results into a data structure
df = pd.DataFrame({'AAA' : [1,1,1,2,2,2,3,3], 'BBB' : [2,1,3,4,5,1,2,3]}); df
# Method 1 : idxmin() to get the index of the mins
df.loc[df.groupby("AAA")["BBB"].idxmin()]
# Method 2 : sort then take first of each
df.sort("BBB").groupby("AAA", as_index=False).first()


#############################################################################
### (groupby 1st function type: Grouping)
# This is similar to R function aggregate()
df = pd.DataFrame({
    'A' : ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
    'B' : ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
    'C' : np.random.randn(8),
    'D' : np.random.randn(8)})

df.groupby('A').sum()
df.groupby('B').sum()
df.groupby(['A', 'B']).sum()

def get_letter_type(letter):
    if letter.lower() in 'aeiou':
        return 'vowel'
    else:
        return 'consonant'

grouped = df.groupby(['A']).first()
grouped = df.groupby(['A'], axis=0).first()

df.groupby('A').groups

# You can pass a function to the groupby() function. In that case, the function
# is applied to the index before grouping the dataframe.
# When no grouping column is explicity given, the dataframe index is used to group items.
df.groupby(get_letter_type, axis=1).groups


# You can iterate through a grouped object.
grouped = df.groupby('A')
for name, group in grouped:
    print(name)
    print(group)

# A single group can be selected using GroupBy.get_group():
grouped.get_group('bar')

# Or for an object grouped on multiple columns:
df.groupby(['A', 'B']).get_group(('bar', 'one'))

# With grouped Series you can also pass a list or dict of functions to do aggregation with, outputting a DataFrame:
grouped = df.groupby('A')
grouped['C'].agg([np.sum, np.mean, np.std])

# By passing a dict to aggregate you can apply a different aggregation to the columns of a DataFrame:
grouped.agg({'C' : np.sum,
             'D' : lambda x: np.std(x, ddof=1)})

grouped.agg(lambda x: np.std(x, ddof=1))


s = Series([9, 8, 7, 5, 19, 1, 4.2, 3.3])
g = Series(list('abababab'))
gb = s.groupby(g)
gb.nlargest(3)
gb.nsmallest(3)


#############################################################################
### (groupby 2nd function type: Transform)
# The transform method returns an object that is indexed the same (same size) as the one being grouped.
# Example: standardize the data within each group.
index = date_range('10/1/1999', periods=1100)
ts = Series(np.random.normal(0.5, 2, 1100), index)
ts = rolling_mean(ts, 100, 100).dropna()
ts.head()

key = lambda x: x.year
zscore = lambda x: (x - x.mean()) / x.std()
transformed = ts.groupby(key).transform(zscore)

# test to see if the transformed dataframe has standardized columns
grouped = ts.groupby(key)
grouped.mean()
grouped.std()
grouped_trans = transformed.groupby(key)
grouped_trans.mean()
grouped_trans.std()


#
data_df = DataFrame({'A' : np.random.randint(0, 10, 1000),
                     'B' : np.random.randint(0, 10, 1000),
                     'C' : np.random.randint(0, 10, 1000)})
data_df.loc[np.random.randint(0, 1000, 100), 'C'] = np.NaN
countries = np.array(['US', 'UK', 'GR', 'JP'])
key = countries[np.random.randint(0, 4, 1000)]
grouped = data_df.groupby(key)
grouped.count()

f = lambda x: x.fillna(x.mean())
transformed = grouped.transform(f)

# Verify
f = lambda x: x.fillna(x.mean())
transformed = grouped.transform(f)

grouped_trans = transformed.groupby(key)
grouped.mean()
grouped_trans.mean()

grouped.count()  # original has some missing data points
grouped_trans.count() # counts after transformation
grouped_trans.size()


#############################################################################
### (groupby 3rd function type: Filteration)
# The filter method returns a subset of the original object.
# Example: take only elements that belong to groups with a group sum greater than 2.
sf = Series([1, 1, 2, 3, 3, 3])
sf.groupby(sf).filter(lambda x: x.sum() > 2)

# The argument of filter must be a function that, applied to the group as a whole, returns True or False.
dff = DataFrame({'A': np.arange(8), 'B': list('aabbbbcc')})
dff.groupby('B').filter(lambda x: len(x) > 2)
dff.groupby('B').filter(lambda x: len(x) > 2, dropna=False)

dff['C'] = np.arange(8)
dff.groupby('B').filter(lambda x: len(x['C']) > 2)


### DISPATCHING TO INSTANCE METHODS
grouped = df.groupby('A')
# To get the std of each column, we can use aggregate() function together with another function
grouped.agg(lambda x: x.std())
# or, we can use groupby() capabilities in dispatching a function to each group
grouped.std()


tsdf = DataFrame(np.random.randn(1000, 3),
                 index=date_range('1/1/2000', periods=1000),
                 columns=['A', 'B', 'C'])
tsdf.ix[::2] = np.nan
grouped = tsdf.groupby(lambda x: x.year)
grouped.fillna(method='pad')



#################
df = pd.DataFrame({'animal': 'cat dog cat fish dog cat cat'.split(),
                   'size': list('SSMMMLL'),
                   'weight': [8, 10, 11, 1, 20, 12, 12],
                   'adult' : [False] * 5 + [True] * 2}); df
# List the size of the animals with the highest weight.
df.groupby('animal').apply(lambda subf: subf['size'][subf['weight'].idxmax()])
gb = df.groupby(['animal'])
gb.get_group('cat')

def GrowUp(x):
    avg_weight =  sum(x[x['size'] == 'S'].weight * 1.5)
    avg_weight += sum(x[x['size'] == 'M'].weight * 1.25)
    avg_weight += sum(x[x['size'] == 'L'].weight)
    avg_weight /= len(x)
    return pd.Series(['L', avg_weight, True], index=['size', 'weight', 'adult'])

expected_df = gb.apply(GrowUp)
expected_df

## Expanding Apply
S = pd.Series([i / 100.0 for i in range(1,11)])
def CumRet(x,y):
    return x * (1 + y)

def Red(x):
    return functools.reduce(CumRet,x,1.0)

pd.expanding_apply(S, Red)

## Replacing some values with the mean of the rest of the group
df = pd.DataFrame({'A' : [1, 1, 2, 2], 'B' : [1, -1, 1, 2]})
gb = df.groupby('A')

def replace(g):
   mask = g < 0
   g.loc[mask] = g[~mask].mean()
   return g

gb.transform(replace)

df = pd.DataFrame({'code': ['foo', 'bar', 'baz'] * 2,
    'data': [0.16, -0.21, 0.33, 0.45, -0.59, 0.62],
    'flag': [False, True] * 3})
code_groups = df.groupby('code')
agg_n_sort_order = code_groups[['data']].transform(sum).sort('data')
sorted_df = df.ix[agg_n_sort_order.index]
sorted_df

## Create multiple aggregated columns
rng = pd.date_range(start="2014-10-07", periods=10, freq='2min')
ts = pd.Series(data = list(range(10)), index = rng)

def MyCust(x):
   if len(x) > 2:
      return x[1] * 1.234
   return pd.NaT

mhc = {'Mean' : np.mean, 'Max' : np.max, 'Custom' : MyCust}
ts.resample("5min", how = mhc)
ts


## Create a value re-assign column and re-assign back to the data frame
df = pd.DataFrame({'Color': 'Red Red Red Blue'.split(),
                   'Value': [100, 150, 50, 50]}); df
df['Counts'] = df.groupby(['Color']).transform(len)
df


## Shift groups of values in a column based on the index
df = pd.DataFrame(
        {u'line_race': [10, 10, 8, 10, 10, 8],
         u'beyer': [99, 102, 103, 103, 88, 100]},
         index=[u'Last Gunfighter', u'Last Gunfighter', u'Last Gunfighter',
                u'Paynter', u'Paynter', u'Paynter']); df
df

df['beyer_shifted'] = df.groupby(level=0)['beyer'].shift(1)
df


## Select rwo with maximum value from each group
df = pd.DataFrame({'host':['other','other','that','this','this'],
                   'service':['mail','web','mail','mail','web'],
                   'no':[1, 2, 1, 2, 1]}).set_index(['host', 'service'])
mask = df.groupby(level=0).agg('idxmax')
df_count = df.loc[mask['no']].reset_index()


## Grouping like Python itertools.groupby
df = pd.DataFrame([0, 1, 0, 1, 1, 1, 0, 1, 1], columns=['A'])
df.A.groupby((df.A != df.A.shift()).cumsum()).groups
df.A.groupby((df.A != df.A.shift()).cumsum()).cumsum()

### HISTOGRAMMING
# This is similar to R table() function
s = pd.Series(np.random.randint(0, 7, size=10))
s.value_counts()
###############################################################################


###############################################################################
### RESHAPING
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
    ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]
# The stack function "compresses" a level in the DataFrame's columns.
# It is similar to the melt() function in R.
stacked = df2.stack()
# With a "stacked" DataFrame or Series (having a MultiIndex as the index),
# the inverse operation of stack is unstack,
# which by default unstacks the last level
stacked.unstack()
# Unstack on the other dimension
stacked.unstack(0)
###############################################################################


###############################################################################
### PIVOT TABLES
df = pd.DataFrame({
    'A' : ['one', 'one', 'two', 'three'] * 3,
    'B' : ['A', 'B', 'C'] * 4,
    'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
    'D' : np.random.randn(12),
    'E' : np.random.randn(12)})
# values: the column containing values that will be structured into the new structure
# index: the new indexing structure. If more than one column selected, it creates a hierarchical index system.
# columns: the set of columns that the values will be organized in.
# index specifies the row structure; columns specifies the column structure.
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])


### PIVOT
df = pd.DataFrame(data={'Province' : ['ON','QC','BC','AL','AL','MN','ON'],
                        'City' : ['Toronto','Montreal','Vancouver','Calgary','Edmonton','Winnipeg','Windsor'],
                        'Sales' : [13,6,16,8,4,3,1]})
table = pd.pivot_table(df, values=['Sales'], index=['Province'], columns=['City'], aggfunc=np.sum,margins=True)
table.stack('City')


## Frequency table like plyr in R
grades = [48,99,75,80,42,80,72,68,36,78]
df = pd.DataFrame( {'ID': ["x%d" % r for r in range(10)],
                    'Gender' : ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'M'],
                    'ExamYear': ['2007','2007','2007','2008','2008','2008','2008','2009','2009','2009'],
                    'Class': ['algebra', 'stats', 'bio', 'algebra', 'algebra', 'stats', 'stats', 'algebra', 'bio', 'bio'],
                    'Participated': ['yes','yes','yes','yes','no','yes','yes','yes','yes','yes'],
                    'Passed': ['yes' if x > 50 else 'no' for x in grades],
                    'Employed': [True,True,True,False,False,False,False,True,True,False],
                    'Grade': grades})

df.groupby('ExamYear').agg({
                    'Participated': lambda x: x.value_counts()['yes'],
                    'Passed': lambda x: sum(x == 'yes'),
                    'Employed' : lambda x : sum(x),
                    'Grade' : lambda x : sum(x) / len(x)})


tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]

stacked = df2.stack()
stacked.unstack()  # by default, unstacking happens at the last level
stacked.unstack(0)  # you can provide the level at which the unstacking should happen (here it is 0)
stacked.unstack('second')  # or provide the name


###############################################################################

###############################################################################
### SPLITTING A FRAME
# Create a list of dataframes, split using a delineation based on logic included in rows.
df = pd.DataFrame(data={'Case' : ['A','A','A','B','A','A','B','A','A'],
                        'Data' : np.random.randn(9)})
dfs = list(zip(*df.groupby(pd.rolling_median((1*(df['Case']=='B')).cumsum(),3,True))))[-1]
dfs[0]
dfs[1]
dfs[2]
###############################################################################

###############################################################################
###############################################################################
###############################################################################

###############################################################################
### EXPANDING DATA

## ROLLING AVERAGE WITH pandas
pd.stats.moments.rolling_apply(df, 30, lambda x: pd.Series(x).idxmax())

actSeries = pd.Series(actMeter)
# Rolling average
pd.rolling_mean(pd.Series(actMeter), 11, center=True)
# Plot rolling avg
actSeries.plot()
pd.rolling_mean(actSeries, 10).plot(style='k')
plt.show()

## Alignment and to-date

## rolling computation window based on values instead of counts
###############################################################################



###############################################################################
############################### APPLY, MERGE ##################################
###############################################################################

###############################################################################
### APPLY
df = pd.DataFrame(data={'A' : [[2,4,8,16],[100,200],[10,20,30]], 'B' : [['a','b','c'],['jj','kk'],['ccc']]},index=['I','II','III'])
def SeriesFromSubList(aList):
      return pd.Series(aList)
df_orgz = pd.concat(dict([ (ind,row.apply(SeriesFromSubList)) for ind,row in df.iterrows() ]))


## Roling apply with adata frame returning a series.
df = pd.DataFrame(data=np.random.randn(2000,2)/10000,
                  index=pd.date_range('2001-01-01',periods=2000),
                  columns=['A','B']); df
def gm(aDF,Const):
    v = ((((aDF.A+aDF.B)+1).cumprod())-1)*Const
    return (aDF.index[0],v.iloc[-1])
S = pd.Series(dict([ gm(df.iloc[i:min(i+51,len(df)-1)],5) for i in range(len(df)-50) ])); S


## Rolling apply with a data frame returning a scalar
rng = pd.date_range(start = '2014-01-01',periods = 100)
df = pd.DataFrame({'Open' : np.random.randn(len(rng)),
                   'Close' : np.random.randn(len(rng)),
                   'Volume' : np.random.randint(100,2000,len(rng))}, index=rng); df
def vwap(bars): return ((bars.Close*bars.Volume).sum()/bars.Volume.sum()).round(2)
window = 5
s = pd.concat([ (pd.Series(vwap(df.iloc[i:i+window]), index=[df.index[i+window]])) for i in range(len(df)-window) ]); s
###############################################################################


###############################################################################
### MERGE
# http://pandas.pydata.org/pandas-docs/stable/merging.html
rng = pd.date_range('2000-01-01', periods=6)
df1 = pd.DataFrame(np.random.randn(6, 3), index=rng, columns=['A', 'B', 'C'])
df2 = df1.copy()
# Append two data frames with overlapping index (emulate R rbind)
df = df1.append(df2, ignore_index=True); df

## Self-join of a data frame
df = pd.DataFrame(data={'Area' : ['A'] * 5 + ['C'] * 2,
                        'Bins' : [110] * 2 + [160] * 3 + [40] * 2,
                        'Test_0' : [0, 1, 0, 1, 2, 0, 1],
                        'Data' : np.random.randn(7)}); df
df['Test_1'] = df['Test_0'] - 1
pd.merge(df, df,
         left_on=['Bins', 'Area','Test_0'],
         right_on=['Bins', 'Area','Test_1'],
         suffixes=('_L','_R'))

# Concatenating pandas objects
df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)


###############################################################################
### JOIN
# http://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html#compare-with-sql-join
# pandas emulates SQL merge behavior
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
pd.merge(left, right, on='key')

#
df = pd.DataFrame({
    'float_col' : [0.1, 0.2, 0.2, 10.1, np.NaN],
    'int_col'   : [1, 2, 6, 8, -1],
    'str_col'   : ['a', 'b', None, 'c', 'a']
})

other = pd.DataFrame({'str_col' : ['a','b'],
                      'some_val' : [1, 2]})

pd.merge(df, other, on='str_col', how="inner")
pd.merge(df, other, on='str_col', how="outer")
pd.merge(other, df, on='str_col', how="outer")
pd.merge(df, other, on='str_col', how="left")
pd.merge(df, other, on='str_col', how="right")


### APPEND
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
s = df.iloc[3]
df.append(s, ignore_index=True)

###############################################################################


###############################################################################
################################## PLOTTING ###################################
###############################################################################


###############################################################################
### PLOTTING
# NOTE: Refer to my notes in matplotlib.py file for more info on interactive plotting in PyCharm.
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()

# The plot method on Series and DataFrame is just a simple wrapper around plt.plot().
ts.plot()

# Either show or save to file. Close when done.
plt.draw()
plt.savefig(plotPath+'/myfig2.png')
plt.close()

# On DataFrame, plot is a convenience to plot all of the columns with labels:
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])
df = df.cumsum()

plt.figure();
df.plot();
plt.legend(loc='best')
plt.draw()
plt.close()




###
df = pd.DataFrame(
         {u'stratifying_var': np.random.uniform(0, 100, 20),
          u'price': np.random.normal(100, 5, 20)})
df[u'quartiles'] = pd.qcut(
    df[u'stratifying_var'],
    4,
    labels=[u'0-25%', u'25-50%', u'50-75%', u'75-100%'])
df.boxplot(column=u'price', by=u'quartiles')

###
fig = plt.figure(figsize=(16,8))
ax = [fig.add_subplot(121),fig.add_subplot(122)]

ax[0].things
ax[1].things

plt.show()
###############################################################################


###############################################################################
############################ DATA IMPORT / EXPORT #############################
###############################################################################

###############################################################################
### CSV
# Write to CSV
df.to_csv(resultsPath+'/foo.csv')

# Read from CSV
df2 = pd.read_csv(resultsPath+'foo.csv')

i = pd.date_range('20000101',periods=10000)
df = pd.DataFrame(dict(year = i.year, month = i.month, day = i.day))
df.head()
# %timeit pd.to_datetime(df.year*10000+df.month*100+df.day,format='%Y%m%d')
# %timeit
###############################################################################


###############################################################################
### SQL
# pandas.io.sql
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')

data = pd.DataFrame({
    'id' : [26, 42, 63],
    'Date' : pd.date_range('2015-04-20', periods=3),
    'Col_1' : ['X', 'Y', 'Z'],
    'Col_2' : [25.7, -12.4, 5.73],
    'Col_3' : [True, False, True]
})

# Write data frame to SQL
data.to_sql('data', engine)
# writes data to the database in batches of 1000 rows at a time
# data.to_sql('data', engine, chunksize=1000)

# For example, specifying to use the sqlalchemy String type instead of the default Text type for string columns
from sqlalchemy.types import String
data.to_sql('data_dtype', engine, dtype={'Col_1': String})

# Read table
pd.read_sql_table('data', engine)

# specify a subset of columns to be read
pd.read_sql_table('data', engine, index_col='id')
pd.read_sql_table('data', engine, columns=['Col_1', 'Col_2'])
pd.read_sql_table('data', engine, parse_dates=['Date'])

# explicitly specify a format string, or a dict of arguments to pass to
pd.read_sql_table('data', engine, parse_dates={'Date': '%Y-%m-%d'})
pd.read_sql_table('data', engine, parse_dates={'Date': {'format': '%Y-%m-%d %H:%M:%S'}})

# Check if a table exists
engine.has_table('data')

## Working with schemas. This works if the engine has schema support.
df.to_sql('table', engine, schema='other_schema')
pd.read_sql_table('table', engine, schema='other_schema')

## Querying
pd.read_sql_query('SELECT * FROM data', engine)

## Executing a SQL query
from pandas.io import sql
sql.execute('SELECT * FROM data', engine)
sql.execute('INSERT INTO data VALUES(?, ?, ?)', engine, params=[('id', 1, 12.2, True)])

# To connect with SQLAlchemy you use the create_engine() function to create an engine object from database URI.
# You only need to create the engine once per database you are connecting to.
from sqlalchemy import create_engine
# Examples:
engine = create_engine('postgresql://scott:tiger@localhost:5432/mydatabase')
engine = create_engine('mysql+mysqldb://scott:tiger@localhost/foo')
engine = create_engine('oracle://scott:tiger@127.0.0.1:1521/sidname')
engine = create_engine('mssql+pyodbc://mydsn')

# sqlite://<nohostname>/<path>
# where <path> is relative:
engine = create_engine('sqlite:///foo.db')

# or absolute, starting with a slash:
engine = create_engine('sqlite:////absolute/path/to/foo.db')

## sqlite
# The use of sqlite is supported without using SQLAlchemy.
# This mode requires a Python database adapter which respect the Python DB-API.
import sqlite3
con = sqlite3.connect(':memory:')
data.to_sql('data', cnx)
pd.read_sql_query("SELECT * FROM data", con)

###############################################################################

###############################################################################
############################## pandas TUTORIAL ################################
###############################################################################

###############################################################################
### Clone Git repo for the tutorial first
# git clone --depth 1 https://amirkav:gheli1382@github.com/jvns/pandas-cookbook.git /Users/amirkavousian/Documents/Py_Codes/pdTut/
pdTutPath = '/Users/amirkavousian/Documents/Py_Codes/pdTut'

broken_df = pd.read_csv(pdTutPath+'/data/bikes.csv')
broken_df[:3]
fixed_df = pd.read_csv(pdTutPath+'/data/bikes.csv', sep=';', encoding='latin1',
                       parse_dates=['Date'], dayfirst=True, index_col='Date')
# select a column
fixed_df['Berri 1']

# Plot the column
plt.figure();
fixed_df['Berri 1'].plot()
plt.legend(loc='best')
plt.draw()
plt.close()

# Make the plot bigger
plt.figure();
fixed_df.plot(figsize=(15, 10))
plt.legend(loc='best')
plt.draw()
plt.close()


### DATA SELECTION, SLICING
complaints = pd.read_csv(pdTutPath+'/data/311-service-requests.csv')
complaints['Complaint Type']
complaints[:4]
complaints[:4]['Complaint Type']
complaints['Complaint Type'][:4]
complaints[['Complaint Type', 'Borough']]

# Complaint types
complaints_counts = complaints['Complaint Type'].value_counts()
complaints_counts[:10]

plt.figure()  # needed if we plot directly from DataFrame object
complaints_counts[:10].plot(kind="bar")
plt.draw()
plt.close()

# Slice & dice noise complaints data set
noise_complaints = complaints[complaints['Complaint Type'] == "Noise - Street/Sidewalk"]
noise_complaints[:3]
is_noise = complaints['Complaint Type'] == "Noise - Street/Sidewalk"
in_brooklyn = complaints['Borough'] == "BROOKLYN"
complaints[is_noise & in_brooklyn][:5]
complaints[is_noise & in_brooklyn][['Complaint Type', 'Borough', 'Created Date', 'Descriptor']][:10]


# which borough has the most noise complaints?
is_noise = complaints['Complaint Type'] == "Noise - Street/Sidewalk"
noise_complaints = complaints[is_noise]
noise_complaints['Borough'].value_counts()

# Noise complaints as a proportion of all complaints in a given borough
noise_complaint_counts = noise_complaints['Borough'].value_counts()
complaint_counts = complaints['Borough'].value_counts()
noise_complaint_counts / complaint_counts

plt.figure()  # needed if we plot directly from DataFrame object
(noise_complaint_counts / complaint_counts).plot(kind="bar")
plt.draw()
plt.close()

### GROUPING
bikes = pd.read_csv(pdTutPath+'/data/bikes.csv', sep=';', encoding='latin1',
                    parse_dates=['Date'], dayfirst=True, index_col='Date')
berri_bikes = bikes[['Berri 1']]
berri_bikes[:5]
# In pandas, index is basically a column for itself; it can have different data types.
# Another benefit of treating the index as an independent data type is that
# the index will be attached to any column that will be read from the DataFrame.
# In short, if you have a column that has unique values and is applicable to all
# of the columns of the data set, convert that column to string.
# For example, date/time columnds in times series are good candidates for index.
berri_bikes.index
# using pandas time-series facilities
berri_bikes.index.day
berri_bikes.index.weekday
berri_bikes['weekday'] = berri_bikes.index.weekday
berri_bikes[:5]

weekday_counts = berri_bikes.groupby('weekday').aggregate(sum)
weekday_counts.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                        'Friday', 'Saturday', 'Sunday']

plt.figure()
weekday_counts.plot(kind='bar')
plt.draw()
plt.close()


### WEB SCRAPING
weather_2012_final = pd.read_csv(pdTutPath+'/data/weather_2012.csv', index_col='Date/Time')

plt.figure()
weather_2012_final['Temp (C)'].plot(figsize=(15, 6))
plt.draw()
plt.close()

# URL for Canada weather data
url_template = "http://climate.weather.gc.ca/climateData/bulkdata_e.html?format=csv&stationID=5415&Year={year}&Month={month}&timeframe=1&submit=Download+Data"
url = url_template.format(month=3, year=2012)
# We can just use the same read_csv function as before, and just give it a URL as a filename.
weather_mar2012 = pd.read_csv(url, skiprows=16, index_col='Date/Time', parse_dates=True, encoding='latin1')

# rename columns
# NOTE: there are several methods to rename columns in pandas:
# http://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
weather_2012_final.columns
weather_mar2012.columns = [s.replace(u'\xb0', '') for s in weather_mar2012.columns]

# Remove variables that have missing data
# NOTE: The argument axis=1 to dropna means "drop columns", not rows",
# The argument how='any' means "drop the column if any value is null".
# The default operation for drop and dropna is always to remove rows.
weather_mar2012 = weather_mar2012.dropna(axis=1, how='any')
weather_mar2012[:5]

# Remove a few more columns
weather_mar2012 = weather_mar2012.drop(['Year', 'Month', 'Day', 'Time', 'Data Quality'], axis=1)


temperatures = weather_2012_final[[u'Temp (C)']]
temperatures['Hour'] = [s[11:] for s in weather_2012_final.index]

plt.figure()
temperatures.groupby('Hour').aggregate(np.median).plot()
plt.draw()
plt.close()


### STRING MANIPULATION
weather_2012 = pd.read_csv(pdTutPath+'/data/weather_2012.csv', parse_dates=True, index_col='Date/Time')
weather_2012[:5]
weather_description = weather_2012['Weather']
is_snowing = weather_description.str.contains('Snow')
is_snowing[:5]

plt.figure()
is_snowing.plot()
plt.draw()
plt.close()

# Get the median temperature of each month
# Use resample() to aggregate the values at the monthly level.
# http://stackoverflow.com/questions/17001389/pandas-resample-documentation
# http://pandas.pydata.org/pandas-docs/dev/timeseries.html#up-and-downsampling
# The how parameter can be a function name or numpy array function that takes an array and produces aggregated values.

plt.figure()
weather_2012['Temp (C)'].resample('M', how=np.median).plot(kind='bar')
plt.draw()
plt.close()

# Find the percentage of time that it was snowing in any month
is_snowing.astype(float)[:100]
is_snowing.astype(float).resample('M', how=np.mean)

plt.figure()
is_snowing.astype(float).resample('M', how=np.mean).plot(kind='bar')
plt.draw()
plt.close()

# Combine temperature and snowiness into one DataFrame
temperature = weather_2012['Temp (C)'].resample('M', how=np.median)
is_snowing = weather_2012['Weather'].str.contains('Snow')
snowiness = is_snowing.astype(float).resample('M', how=np.mean)

# Name the columns
temperature.name = "Temperature"
snowiness.name = "Snowiness"

# pd.concat(axis=1) is similar to cbind in R.
stats = pd.concat([temperature, snowiness], axis=1)
stats

# Multiple plots
plt.figure()
stats.plot(kind='bar', subplots=True, figsize=(15, 10))
plt.draw()
plt.close()


### MESSY DATA
requests = pd.read_csv(pdTutPath+'/data/311-service-requests.csv')
requests['Incident Zip'].unique()

# First, code all unacceptable values as NA
na_values = ['NO CLUE', 'N/A', '0']
requests = pd.read_csv(pdTutPath+'/data/311-service-requests.csv',
                       na_values=na_values, dtype={'Incident Zip': str})
requests['Incident Zip'].unique()

# Remove all lines with dashes
rows_with_dashes = requests['Incident Zip'].str.contains('-').fillna(False)
len(requests[rows_with_dashes])
requests[rows_with_dashes]
# But we should not remove these lines, since dashed zipcodes are normal.
# Take a closer look at zipcodes that may be wrong.
long_zip_codes = requests['Incident Zip'].str.len() > 5
requests['Incident Zip'][long_zip_codes].unique()

# To standardize zipcodes, just truncate them at length 5
requests['Incident Zip'] = requests['Incident Zip'].str.slice(0, 5)
requests.shape

# Setting 00000 zipcodes to nan
zero_zips = requests['Incident Zip'] == '00000'
requests['Incident Zip'][zero_zips] = np.nan

unique_zips = requests['Incident Zip'].unique()
unique_zips.sort()
unique_zips

# Cleanse zipcode data based on values
zips = requests['Incident Zip']
is_close = zips.str.startswith('0') | zips.str.startswith('1')
# Everyhing that is NA also considered as "far"
is_far = ~(is_close.fillna(True).astype(bool))
requests[is_far][['Incident Zip', 'Descriptor', 'City']].sort('Incident Zip')

# Since there may be requests coming from cities far from NY, it is better to filter by city
requests['City'].str.upper().value_counts()


### Unix TIMESTAMPS
popcon = pd.read_csv(pdTutPath+'/data/popularity-contest', sep=' ', )[:-1]
popcon.columns = ['atime', 'ctime', 'package-name', 'mru-program', 'tag']
popcon['atime'].dtype

# numpy datetimes are already stored as Unix timestamps.
# So all we need to do is tell pandas that these integers are actually datetimes.
popcon['atime'] = popcon['atime'].astype(int)
popcon['ctime'] = popcon['ctime'].astype(int)
popcon['atime'].dtype

# Use the pd.to_datetime function to convert our integer timestamps into datetimes.
popcon['atime'] = pd.to_datetime(popcon['atime'], unit='s')
popcon['ctime'] = pd.to_datetime(popcon['ctime'], unit='s')
popcon['atime'].dtype

# Remove anything with timestampe less than 0
popcon = popcon[popcon['atime'] > '1970-01-01']

# just look at rows where the package name doesn't contain 'lib'.
nonlibraries = popcon[~popcon['package-name'].str.contains('lib')]
nonlibraries.sort('ctime', ascending=False)[:10]


### NOTE: I have completed the Tutorials on this page.
# http://pandas.pydata.org/pandas-docs/stable/tutorials.html
# The next step is to start from "Lessons for New pandas Users" section.
# On PDF file, I am at the beginning of Cookbook section.

###############################################################################
###################### PYTHON FOR NEW USERS TUTORIAL ##########################
###############################################################################
# http://nbviewer.ipython.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/01%20-%20Lesson.ipynb
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]
# zip() creates a list of tuples, with each tuple being a combination of contributing elements.
babyDataSet = zip(names, births)
type(babyDataSet)

# Convert to DataFrame
df = pd.DataFrame(data = babyDataSet, columns=['Names', 'Births'])

df.to_csv(pdTutPath+'/pdTut2/births1880.csv', index=False, header=False)

Location = pdTutPath+r'/pdTut2/births1880.csv'
# The parameter "Names" is additive; it does not replace existing column names if any exists.
df = pd.read_csv(Location, names=['Names','Births'])

# Delete the original file from disk
import os
os.remove(Location)

# Check data integrity
df.dtypes

# Sort data frame
Sorted = df.sort(['Births'], ascending=False)

# Find maximum frequency
Sorted.head(1)
maxValue = df['Births'].max()
df.Births.max()

# Find the name with maximum frequency
df['Names'][df['Births'] == df['Births'].max()]
maxName = df['Names'][df['Births'] == df['Births'].max()].values
Sorted['Names'].head(1)

# Plot
plt.figure()
df['Births'].plot()
annText = str(maxValue) + " - " + maxName
plt.annotate(annText, xy=(1, maxValue), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
plt.draw()
plt.close()
###############################################################################


########################################################################################################################
### LESSON 3: Creating functions - Reading from EXCEL - Exporting to EXCEL - Outliers - Lambda functions - Slice and dice data
import numpy.random as npr
npr.seed(111)
def CreateDataSet(Number=1):

    Output = []

    for i in range(Number):

        # Create a weekly (mondays) date range
        rng = pd.date_range(start='1/1/2009', end='12/31/2012', freq='W-MON')

        # Create random data
        data = npr.randint(low=25, high=1000, size=len(rng))

        # Status pool
        status = [1,2,3]

        # Make a random list of statuses
        random_status = [status[npr.randint(low=0,high=len(status))] for i in range(len(rng))]

        # State pool
        states = ['GA','FL','fl','NY','NJ','TX']

        # Make a random list of states
        random_states = [states[npr.randint(low=0,high=len(states))] for i in range(len(rng))]

        Output.extend(zip(random_states, random_status, data, rng))

    return Output

### DATA CLEANING
dataSet = CreateDataSet(4)
df = pd.DataFrame(data=dataSet, columns=['State','Status','CustomerCount','StatusDate'])

# Get a quick summary of the DataFrame
df.info()
df.head()
df.dtypes
df.index
df.shape


# Write to csv
df.to_csv(pdTutPath+'/Lesson3.xlsx', index=False)
print 'Done'


# Parse a specific sheet
Location = pdTutPath+r'/Lesson3.csv'
df = pd.read_csv(Location, 0, index_col='StatusDate')

# State names may have upper/lower case issues.
df.State.unique()
df['State'].unique()

df.State = df.State.apply(lambda x: x.upper())
df.State.unique()

# Only grab where status == 1
mask = df.Status == 1
# NOTE: pandas DataFrames default to selecting rows.
# R data frames throw an error if the dimensions of the mask does not match the dimensions of the target data frame.
df = df[mask]
df.shape

# Convert NJ to NY
mask = df.State == "NJ"
df.State[mask] = "NY"

# Plot
plt.figure()
df.CustomerCount.plot()
plt.draw()
plt.close()

#
dfSorted = df[df['State'] == "NY"].sort(axis=0)
dfSorted.shape
dfSorted.head()

### Get daily commute by state and date
# the function reset_index() brings the index back as a separate column (as opposed to being an index)
df.reset_index()

# NOTE how groupby() function needs a chaser function to specify the FUN to use in aggregating.
# The following line removes duplicate state-date pairs by adding their counts up.
daily = df.reset_index().groupby(['State', 'StatusDate']).sum()
df.head()
daily.head()

# We can now delete the column named 'Status'
del daily['Status']

# Note how groupby puts the aggregating dimensions in the index of the resulting DataFrame.
# pandas DataFrame index is similar to the key of a database, but without the constraint of having only unique values.
daily.index  # all indexes (may be multiple dimensions)
daily.index.levels[0]  # unique values of the index
daily.index.get_level_values(0)  # all values of the index
# NOTE: pandas is smart about different columns in a DataFrame. When plotting a dataframe,
# pandas uses the DataFrame index for x values. It skips the column called "index" too.
daily.loc['FL']

# Plot a particular state
plt.figure()
daily.loc['FL']['CustomerCount'].plot()
plt.legend(loc='Best')
plt.draw()
plt.close()

# Plot a particular state for only 2012
plt.figure()
daily.loc['FL']['2012':]['CustomerCount'].plot()
plt.legend(loc='Best')
plt.draw()
plt.close()

### FIND AND REMOVE OUTLIERS

stateYearMonth = daily.groupby([daily.index.get_level_values(0),
                                daily.index.get_level_values(1).year,
                                daily.index.get_level_values(1).month])
daily['Lower'] = stateYearMonth['CustomerCount'].transform( lambda x: x.quantile(q=.25) - (1.5*x.quantile(q=.75)-x.quantile(q=.25)) )
daily['Upper'] = stateYearMonth['CustomerCount'].transform( lambda x: x.quantile(q=.75) + (1.5*x.quantile(q=.75)-x.quantile(q=.25)) )
daily['Outlier'] = (daily['CustomerCount'] < daily['Lower']) | (daily['CustomerCount'] > daily['Upper'])
daily = daily[daily['Outlier'] == False]
daily.shape

# Get the max customer count by Date
sumDaily = pd.DataFrame(daily['CustomerCount'].groupby(daily.index.get_level_values(1)).sum())
sumDaily.columns = ['CustomerCount']

# Group by year and month
# NOTE: you can use a lambda function to define groupby dimensions on the fly
yearMonth = sumDaily.groupby([lambda x: x.year, lambda x: x.month])

# Find the maximum custoemr count per year and month
# NOTE: transform() will keep the shape of the DataFrame intact, but apply() will not.
# This means that even though it aggregates the numbers, it does not change the dimensions.
sumDaily['Max'] = yearMonth['CustomerCount'].transform(lambda x: x.max())
# If we had used apply, we would have got a dataframe with (Year and Month) as the index and just the Max column with the value of 901


### COMPARE AGAINST A BASELINE
data = [1000,2000,3000]
idx = pd.date_range(start='12/31/2011', end='12/31/2013', freq='A')
bhag = pd.DataFrame(data, index=idx, columns=['BHAG'])
bhag

# combine bhag with actual data
combined = pd.concat([sumDaily, bhag], axis=0)
combined = combined.sort(axis=0)
combined.tail()

# Plot
plt.figure()
combined.plot()
plt.draw()
plt.close()

# Plot using subplots
fig, axes = plt.subplots(figsize=(12, 7))
combined['BHAG'].fillna(method='pad').plot(color='green', label='BHAG')
combined['Max'].plot(color='blue', label='All Markets')
plt.legend(loc='best');
plt.draw()
plt.close()

# Get max number of customers in each year
year = combined.groupby(lambda x: x.year).max()

# Add a column to show percent change per year
year['YR_PCT_Change'] = year['Max'].pct_change(periods=1)
# To estimate next year count, assume the change rate stays the same between this year and the next.
(1 + year.ix[2012, 'YR_PCT_Change']) * year.ix[2012, 'Max']

# Plot
plt.figure()
sumDaily['Max'].plot(figsize=(10, 5));
plt.title('ALL Markets')
plt.legend(loc="best")
plt.draw()
plt.close()

# Multiple plots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
fig.subplots_adjust(hspace=1.0)  # create space between plots

daily.loc['FL']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[0,0])
daily.loc['GA']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[0,1])
daily.loc['TX']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[1,0])
daily.loc['NY']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[1,1])

# Add titles
axes[0,0].set_title('Florida')
axes[0,1].set_title('Georgia')
axes[1,0].set_title('Texas')
axes[1,1].set_title('North East');

plt.draw()
plt.close()
###############################################################################


########################################################################################################################
### LESSON 4: Adding/deleting columns - Index operations
d = [0,1,2,3,4,5,6,7,8,9]
df = pd.DataFrame(d)

### ADDING AND MANIPULATING COLUMNS
df.columns = ['Rev']
df['NewCol'] = 5
df['NewCol'] = df['NewCol'] + 1
df['test'] = 3
df['col'] = df['Rev']

### MODIFY INDEX
i = ['a','b','c','d','e','f','g','h','i','j']
df.index
df.index = i
df.index

### SELECT ROWS
# Slicing by loc (uses indexes for rows)
# df.loc[inclusive:inclusive]
df.loc['a']
df.loc['a':'e']

# df.iloc[inclusive:exclusive]
# Note: .iloc is strictly integer position based.
df.iloc[0:3]

# Select using column name
df['Rev']
df.Rev
df[['Rev', 'test']]

# df['ColumnName'][inclusive:exclusive]
df['Rev'][0:3]
df['col'][5:]
df[['col', 'test']][:3]

# Select top N number of records (default = 5)
df.head()
df.head(3)
# Select bottom N number of records (default = 5)
df.tail()
###############################################################################


###############################################################################
### LESSON 5: Stack/Unstack/Transpose functions
d = {'one':[1,1], 'two':[2,2]}  # a dictionary
i = ['a','b']  # a list
# Create a DataFrame from a dictionary. Note that each key becomes a column header, and values become records.
df = pd.DataFrame(data = d, index = i)
df.index

# Stack the columns and place them in the index (similar to R melt function)
stack = df.stack()
stack
# NOTE how MultiIndex objects have levels and labels attributes.
stack.index
type(stack.index)
stack.index.shape
stack.index[0]

# NOTE: stack() stacks elements by row first.
# NOTE: unstack() stacks elements by column first.
unstack = df.unstack()
unstack.index

# T attribute transposes a DataFrame
transpose = df.T
###############################################################################


###############################################################################
### LESSON 6: groupby
# Starting with a dictionary
d = {'one':[1,1,1,1,1],
     'two':[2,2,2,2,2],
     'letter':['a','a','b','b','c']}
# Convert to DataFrame
df = pd.DataFrame(d)

# NOTE: groupby() does not create a new df by itself. You still need to provide the aggregation function to it.
one = df.groupby('letter')
one.sum()
one.max()
one.min()
one.count()

letterone = df.groupby(['letter','one'])
letterone.sum()
letterone.sum().index

# groupby() automatically creates index out of the grouping columns. You can prevent that by the following option:
letterone = df.groupby(['letter', 'one'], as_index=False)
letterone.sum()

###############################################################################
### LESSON 7: outlier detection
# Create a dataframe out of a list, with dates as your index
States = ['NY', 'NY', 'NY', 'NY', 'FL', 'FL', 'GA', 'GA', 'FL', 'FL']
data = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
idx = pd.date_range('1/1/2012', periods=10, freq='MS')
df1 = pd.DataFrame(data, index=idx, columns=['Revenue'])
df1['State'] = States

# Create a second dataframe
data2 = [10.0, 10.0, 9, 9, 8, 8, 7, 7, 6, 6]
idx2 = pd.date_range('1/1/2013', periods=10, freq='MS')
df2 = pd.DataFrame(data2, index=idx2, columns=['Revenue'])
df2['State'] = States

###############################################################################
### LESSON 8: connection with MS SQL database
### OPTION 1
from sqlalchemy import create_engine, MetaData, Table, select
ServerName = "RepSer2"
Database = "BizIntel"
TableName = "DimDate"

# Create the connection
engine = create_engine('mssql+pyodbc://' + ServerName + '/' + Database)
conn = engine.connect()

# Required for querying tables
metadata = MetaData(conn)

# Table to query
tbl = Table(TableName, metadata, autoload=True, schema="dbo")
#tbl.create(checkfirst=True)

# Select all
sql = tbl.select()

# run sql code
result = conn.execute(sql)

# Insert to a dataframe
df = pd.DataFrame(data=list(result), columns=result.keys())

# Close connection
conn.close()

print 'Done'


### OPTION 2
import pyodbc
# Parameters
server = 'repser2'
db = 'BizIntel'

# Create the connection
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + db + ';Trusted_Connection=yes')

# query db
sql = """

SELECT top 5 *
FROM DimDate

"""
df = pandas.io.sql.read_sql(sql, conn)
df.head()

### OPTION 3
# Parameters
ServerName = "RepSer2"
Database = "BizIntel"

# Create the connection
engine = create_engine('mssql+pyodbc://' + ServerName + '/' + Database)

df = pd.read_sql_query("SELECT top 5 * FROM DimDate", engine)
df

###############################################################################
### LESSION 10: Convering between diff data types and formats
# Create DataFrame
d = [1,2,3,4,5,6,7,8,9]
df = pd.DataFrame(d, columns = ['Number'])
df

# Export to Excel
df.to_excel(pdTutPath+'/pdTut2'+'/Lesson10.xlsx', sheet_name = 'testing', index = False)
print 'Done'



###############################################################################
########################### TIME SERIES IN PANDAS #############################
###############################################################################
# http://pandas.pydata.org/pandas-docs/stable/timeseries.html
# NOTE: time series frequency values in pandas:
# H hourly
# T minutes
# S seconds
# D daily
# M monthly
# B business day
# BH business hour
# BM business month (last business day in each month)
# 2BM every two business months (last business day in each month)
# A year end
# Q querter end
# MS month start
# C custom business day
# W weekly
# W-SUN weekly, sundays. Dashed frequencies are called "offset" frequencies.
# BH business hour (will move to next day if reaching the end of the business hour on a day)
# '2h20min', '1D10S', etc.
# see more here:
# http://pandas.pydata.org/pandas-docs/stable/timeseries.html


### TIME SERIES RESAMPLING
# Use resample() function. Specify resampling operation. Specify frequency.
import random as rd
rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min') # default is mean
ts.resample('5Min', how='sum')
ts.resample('5Min', how='ohlc')  # open, high, low, close
ts.resample('5Min', how=np.max)
ts.resample('5Min', closed='right')  # right-bounded. The boundary timestamp belongs to the left-side range.
ts.resample('5Min', closed='left')  # left-bounded. The boundary timestamp belongs to the right-side range.
# By specifying a finer frequency, we can fill in the time series values
ts[:2].resample('250L')  # from secondly to every 250 milliseconds
ts[:2].resample('250L', fill_method='pad')
ts[:2].resample('250L', fill_method='pad', limit=2)

# label specifies whether the result is labeled with the beginning or the end of the interval.
ts.resample('5Min') # by default label='right'
ts.resample('5Min', label='left')
# loffset performs a time adjustment on the output labels.
ts.resample('5Min', label='left', loffset='1s')


###
# A Period represents a span of time (e.g., a day, a month, a quarter, etc).
pd.Period('2012', freq='A-DEC')
pd.Period('2012-1-1', freq='D')
pd.Period('2012-1-1 19:00', freq='H')

# Adding or subtracting integers from a period will shift the period by its own frequency.
p = pd.Period('2012', freq='A-DEC')
p + 1
p - 3

# If Period freq is daily or higher (D, H, T, S, L, U, N), offsets and timedelta-like can be added if the result can have the same freq.
p = pd.Period('2014-07-01 09:00', freq='H')
p + Hour(2)
p + pd.timedelta(minutes=120)


### Parsing UNIX Timestamps
popcon = pd.read_csv(pdTutPath+'/data/popularity-contest', sep=' ', )[:-1]
popcon.columns = ['atime', 'ctime', 'package-name', 'mru-program', 'tag']

# numpy datetimes are already stored as Unix timestamps.
# So all we need to do is tell pandas that these integers are actually datetimes -- it doesn't need to do any conversion at all.

# First convert the time columns to integer
popcon['atime'] = popcon['atime'].astype(int)
popcon['ctime'] = popcon['ctime'].astype(int)
popcon['atime'].dtype


# We can use the pd.to_datetime function to convert our integer timestamps into datetimes.
popcon['atime'] = pd.to_datetime(popcon['atime'], unit='s')
popcon['ctime'] = pd.to_datetime(popcon['ctime'], unit='s')
popcon['atime'].dtype
# To understand the code types:
# http://pandas.pydata.org/pandas-docs/stable/timeseries.html

# Even though the atime column is now coded as date, we can compare it to a string representation of a date (pandas converts the string to date on the fly)
popcon = popcon[popcon['atime'] > '1970-01-01']

# From popularity contest file, we want to look at rows where the package name does not include 'lib'
nonlibraries = popcon[~popcon['package-name'].str.contains('lib')]
nonlibraries.sort('ctime', ascending=False)[:10]


###############################################################################
################# PRACTICAL DATA ANALYSIS WITH PYTHON GUIDE ###################
###############################################################################
# http://wavedatalab.github.io/datawithpython/index.html

###############################################################################
### LESSON 1: DATA MUNGING
# http://wavedatalab.github.io/datawithpython/munge.html
df = pd.DataFrame({'row'   : np.random.randint(0, 21, 100),
                   'One_X' : np.random.randint(10, 41, 100),
                   'One_Y' : np.random.randint(-20, 11, 100),
                   'Two_X' : np.random.randint(50, 161, 100),
                   'Two_Y' : np.random.randint(15, 61, 100)
                  });


# View the first few rows of the file and set the number of columns displayed.
# pd.set_option('display.max_columns', 80)
df.head(3)
# Determine the number of rows and columns in the dataset
df.shape
# Find the number of rows in the dataset
len(df)
# Get the names of the columns
df.columns
# Get the first three rows of a column by name
df['One_Y'][:3]

# Create categorical ranges for numerical data.
incomeranges = pd.cut(df['Two_X'], 14)
incomeranges[:5]
# Look at the value counts in the ranges created above (similar to R table() function)
pd.value_counts(incomeranges)

# Index into the first six columns of the first row
df.ix[0, 0:6]

# Order the data by specified column
df['One_X'].order()[:5]
# Sort by a column and that obtain a cross-section of that data
sorteddata = df.sort(['One_X'])
sorteddata.ix[:,0:6].head(3)
# Obtain the first three rows and first three columns of the sorted data
sorteddata.iloc[0:3,0:3]
# Obtain value counts of specific column
df['One_X'].value_counts()

# A way to obtain the datatype for every column
# zip() function creates a list from a collection of items.
zip(df.columns, [type(x) for x in df.ix[0,:]])
# The pandas way to obtain datatypes for every column
df.dtypes

# Get the unique values for a column by name.
df['One_X'].unique()
len(df['One_X'].unique())

# Index into a column and get the first four rows
# This is similar to R's way of slicing a dataframe or matrix
df.ix[0:3, 'One_X']

# Obtain binary values (boolean indicators)
df.ix[0:30, 'One_X'] == 16


###############################################################################
### LESSON 2: AGGREGATING DATA
ver = pd.read_csv(pdTutPath+'/data/311-service-requests.csv')
ver = pd.DataFrame({
    'Fac1' : ['A', 'A', 'B', 'C', 'A', 'C', 'C', 'A'],
    'Fac2' : ['d', 'e', 'e', 'f', 'f', 'e', 'd', 'd'],
    'Int1' : np.random.randint(0, 5, 8),
    'Int2' : np.random.randint(5, 15, 8)
})

ver.ix[list(np.random.randint(0,10,3)), 'Int1'] = np.NaN


# melt() function keeps the ids as independent columns, and stacks the other columns on top of each other.
melt1 = pd.melt(ver, id_vars = 'Fac1')
melt = pd.melt(ver, id_vars = ['Fac1', 'Fac2'])
ver.shape
melt.shape
melt1.shape

# Obtain the first five rows of melted data
melt.iloc[0:5, :]
melt.iloc[:5, :]  # same as above
ver.iloc[0:5, :]


### SUMMARY OF DATA
# Get descriptive statistics of the dataset (similar to R summary() function)
ver.describe()
# NOTE: by default, most operations in pandas exclude NaN values.
# As seen, count of values also excludes NaN values.

ver.cov()
ver.corr()



# Crosstab of the data by specified columns (similar to R table() function)
# NOTE: crosstab function automatically turns the variables into categorical variables.
pd.crosstab(ver['Fac1'], ver['Fac2'])
pd.crosstab(ver['Int1'], ver['Int2'])
pd.crosstab(ver['Int1'], ver['Fac2'])
pd.crosstab(ver['Fac1'], ver['Int2'])

# pivot_table()

# group data and find mean (similar to R function aggregate() )
ver.groupby(['Fac1', 'Fac2']).mean()
ver.groupby(['Fac1', 'Fac2']).aggregate(np.mean)
ver.groupby('Fac1')['Int1'].agg(np.mean)

### SUBSET SELECTION
ver[(ver['Int1'] > 2) & (ver['Int2'] < 8)]
ver.query('(Int1 > 2) & (Int2 < 8)')

# Find unique elements on an array
ver['Fac1'].unique()
ver['Int1'].unique()

# Check boolean conditions
ver.ix[:, 'Int1'] > 2
(ver.ix[:, 'Int1'] > 2).any()  # True if at least one element is true

# Return boolean values based on a specific criteria
ver['Fac1'].map(lambda x: x.startswith('B'))


###############################################################################
### LESSON 3: VISUALIZE DATA
import seaborn as sns

a=['A', 'B', 'C', 'D', 'E', 'F']
b=np.random.randint(0, 3, 20)
c=np.random.randint(2, 6, 20)
ver = pd.DataFrame({
    'Fac1' : [a[i] for i in b],
    'Fac2' : [a[i] for i in c],
    'Int1' : np.random.randint(0, 50, 20),
    'Int2' : np.random.randint(15, 75, 20)
})

# barplot of counts
plt.figure();
ver.Fac1.value_counts()
ver.Fac1.value_counts().plot(kind='barh')
plt.legend(loc='best')
plt.draw()
plt.close()

# Bar plot of median values
plt.figure();
ver.groupby('Fac1')['Int1'].agg(np.mean).plot(kind='bar')
plt.legend(loc='best')
plt.draw()
plt.close()


###############################################################################
### LESSON 4: TIME-SERIES
# http://wavedatalab.github.io/datawithpython/timeseries.html

# pandas_datareader module is needed to download data from online sources.
import datetime as dt
from pandas_datareader import data, wb

yhoo = data.DataReader("yhoo", "yahoo", dt.datetime(2007, 1, 1),
    dt.datetime(2012,1,1))

# Plot stock price and volume
plt.figure(figsize=(15,8))
top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
top.plot(yhoo.index, yhoo["Close"])
plt.title('Yahoo Price from 2007 - 2012')

bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
bottom.bar(yhoo.index, yhoo['Volume'])
plt.title('Yahoo Trading Volume')

plt.gcf().set_size_inches(15,8)
plt.draw()
plt.close()

# Calcuate moving average
mavg = yhoo['30_MA_Open'] = pd.stats.moments.rolling_mean(yhoo['Open'], 30)
yhoo['30_MA_Open'].tail()
yhoo[160:165]
yhoo.ix['2007-08-24']

# look at the volume
plt.figure()
yhoo.Volume.plot()
plt.draw()


# plot everything
plt.figure()
yhoo.plot(subplots=True, figsize=(8,8))
plt.legend(loc='best')
plt.draw()
plt.close()


# moving avg plot
close_px = yhoo['Adj Close']
mavg = pd.rolling_mean(close_px, 30)

plt.figure(figsize=(15,8))
yhoo.Close.plot(label='Yahoo')
mavg.plot(label='mavg')
plt.legend()
plt.gcf().set_size_inches(15,8)
plt.draw()
plt.close()


# KDE plot
plt.figure()
yhoo.Close.plot(kind='kde')
plt.draw()
plt.close()

# Time-series range
rng = pd.date_range('1/1/2011', periods=72, freq='H')

# Index pandas objects with dates
ts = pd.Series(np.random.randn(len(rng)), index=rng)


ts.ix['2011-01-03 17:00:00']
ts.ix[8:10]
ts.ix[[8,10,25]]
ts[8:10]

# loc and iloc have superseded ix
ambiguous = pd.DataFrame(np.random.randn(4, 4), index=[1, 1, 3, 4])
ambiguous.ix[1]  # by index
ambiguous.loc[1]  # by index
ambiguous.iloc[1]  # by location
ambiguous[1]  # by location

ambiguous[ambiguous.index.values > 1]
ambiguous.iloc[ambiguous.index.values > 1]
ambiguous.loc[ambiguous.index.values > 1]


# Multi-level index
arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
          ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
tuples = zip(*arrays)
h_ind = pd.MultiIndex.from_tuples(tuples, names=['level1', 'level2'])
hi_s = pd.Series(np.random.randn(8), index=h_ind)
hi_f = pd.DataFrame(np.random.randn(8, 4), index=h_ind)

# Use cartesian product of a few lists.
from itertools import product
words = ['hi', 'hello', 'hola']
nums = [10, 20]
letters = ['a', 'b']
ind = pd.MultiIndex.from_tuples(list(product(words, nums, letters)),
                            names=['word', 'num', 'let'])
hi_f2 = pd.DataFrame(np.random.randn(12, 3), columns=['A', 'B', 'C'], index=ind)

# have column names of DataFrame become outer level of series index using unstack()
df1 = pd.DataFrame(np.random.randn(4, 4))
df2 = pd.DataFrame(np.arange(16).reshape(4, 4), columns=list('abcd'))
s1 = pd.Series(np.random.randn(5))
s2 = pd.Series([i ** 2 for i in xrange(1, 6)])
df3 = pd.DataFrame({'one': s1, 'two':s2})

hi_f3 = df3.unstack()







#*#TODO: To continue where I left, start from cookbook on pandas documentation:
# http://pandas.pydata.org/pandas-docs/stable/cookbook.html


