
import pandas as pd

###############################################################################
#################### INDEXING AND SLICING DATA FRAMES #########################
###############################################################################

# NOTE: Standard Python and NumPy slicing and selection operators work in pandas context as well.
# However, for production code, it is recommended to use the optimized pandas data access methods, .at, .iat, .loc, .iloc and .ix.

#######################################
### There are 2 explicit slicing methods, with a third general case
#  Positional-oriented (Python slicing style : exclusive of end) --> iloc
#  Label-oriented (Non-Python slicing style : inclusive of end) --> loc
#  General (Either slicing style : depends on if the slice contains labels or positions) --> ix
df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]});

# Label
df.loc['bar':'kar'] # Label

# Generic
df.ix[0:3] #Same as .iloc[0:3]
df.ix['bar':'kar'] #Same as .loc['bar':'kar']


#######################################
rng = dates = pd.date_range('1/1/2013', periods=100, freq='D')
data = np.random.randn(100, 4)
cols = ['A','B','C','D']
df1, df2, df3 = pd.DataFrame(data, rng, cols), pd.DataFrame(data, rng, cols), pd.DataFrame(data, rng, cols)
df = df1

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
df.iat[1,1]


### It can get ambiguous when an index consists of integers with a non-zero start or non-unit increment.
data = {'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]}
df2 = pd.DataFrame(data=data,index=[1,2,3,4]);  # Note index starts at 1.
df2.iloc[1:3]  # Position-oriented, exclusive of end
df2.loc[1:3]  # Label-oriented, inclusive of end
df2.ix[1:3] # General, will first mimic loc (label-oriented, inclusive of end); if that raises an error, will fall back on iloc with (position-oriented, exclusive of end)
df2.ix[0:3] # General, will mimic iloc (position-oriented, exclusive of end), as loc[0:3] would raise a KeyError



###############################################################################
### ASSIGNING VALUES TO DATA FRAME USING INDEXING
# http://stackoverflow.com/questions/17557650/edit-pandas-dataframe-using-indexes


### FINDING THE INDEX OF nan
# http://stackoverflow.com/questions/14016247/python-find-integer-index-of-rows-with-nan-in-pandas
na_readings = df2[str(df2.columns[-1])][np.isnan(df2[str(df2.columns[-1])])].index.tolist()
df2.iloc[na_readings, [-3,-2,-1]]

### DROPPING nan
# http://stackoverflow.com/questions/14991195/how-to-remove-rows-with-null-values-from-kth-column-onward-in-python


###############################################################################
############################### BOOLEAN INDEXING ##############################
###############################################################################

### Boolean indexing
# Using a single column's values to select data
df[df.A > 0]
df[df > 0]

# Using the isin() method for filtering:
df2 = df.copy()
df2['E']=['one', 'one','two','three','four']*20

df2[df2['E'].isin(['two','four'])]

### ix: primarily label-based selection, with location-based fallback.
df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]});
df.ix[df.AAA >= 5,'BBB'] = -1;
df
df.ix[df.AAA >= 5,['BBB','CCC']] = 555;
df


s = pd.Series(range(-3, 4))
s[s > 0]
s[(s < -1) | (s > 0.5)]
s[~(s < 0)]
df[df['A'] > 0]


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


###############################################################################
### where() METHOD AND BOOLEAN MASKING
# http://pandas.pydata.org/pandas-docs/stable/indexing.html#the-where-method-and-masking

"""
The method where() is a masking method. Meaning it will preserve the cells that satisfy the criteria,
and it sets the other cells to NaN or to whatever value you pass to the method.
"""
df.where(df < 0, 1)


"""
To understand dataframe and Series item selections in this section, we need to distinguish between indexing and boolean criterion.
A criterion has the same shape as the original data.
An index does not have the same shape as the original data. Instead, it has multiple value along different dimensions of the original data.
"""


"""
Selecting values from a Series with a boolean vector generally returns a subset of the data.
To guarantee that selection output has the same shape as the original data,
you can use the where method in Series and DataFrame.
"""
s = pd.Series(np.arange(5), index=np.arange(5)[::-1], dtype='int64')

# This returns only rows that satisfy the boolean requirement
s[s > 0]

# .where() method returns all rows, with NaN for rows that do not satisfy the requirment
s.where(s > 0)


"""
For a dataframe, boolean criterion works the same as .where() method,
preserving the shape of the original dataframe.
"""
df[df < 0]


"""
Note: do not mistake a criterion with boolean index.
A criterion has the same shape as the original dataframe.
An index selects elements along the dimensions of the original dataframe.
"""
df[df['CCC'] < 0]


"""
In addition, where takes an optional other argument for replacement of values
where the condition is False, in the returned copy.
"""
df.where(df < 0, -df)

# You may wish to set values based on some boolean criteria.
s2 = s.copy()
s2[s2 < 0] = 0

df2 = df.copy()
df2[df2 < 0] = 0
df2.where(df2 < 0, 0)


"""
By default, where returns a modified copy of the data.
There is an optional parameter inplace so that the original data can be modified without creating a copy
"""
df_orig = df.copy()
df_orig.where(df > 0, -df, inplace=True);
df_orig


# We can use another column within the same dataframe as the replacement option for .where() statement
df2 = df.copy()
df2.where(df2>0,df2['AAA'],axis='index')


# (New in version 0.18.1.) we can use a function as replacement option for .where() method
df3 = pd.DataFrame({'A': [1, 2, 3],
                    'B': [4, 5, 6],
                    'C': [7, 8, 9]})
df3.where(lambda x: x > 4, lambda x: x + 10)


### Read this:
# http://stackoverflow.com/questions/21800169/python-pandas-get-index-of-rows-which-column-matches-certain-value
# http://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
###############################################################################


###############################################################################
### MASK
# http://pandas.pydata.org/pandas-docs/stable/indexing.html#the-where-method-and-masking

# pd.where() works like ifelse() function in R.
# It does not change cells with mask value of True,
# but changes cells with False mask values to the parameter value provided.

# mask is the inverse boolean operation of where

df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]});
df_mask = pd.DataFrame({'AAA' : [True] * 4, 'BBB' : [False] * 4,'CCC' : [True,False] * 2})
df.where(df_mask, -1000)


s = pd.Series(np.arange(-3,4))
s.mask(s >= 0)


df.mask(df >= 0)


#######################################
# Take complement of a mask by using ~ operator
df = pd.DataFrame({'AAA' : [4,5,6,7],
                   'BBB' : [10,20,30,40],
                   'CCC' : [100,50,-30,-50]}); df
df[~((df.AAA <= 6) & (df.index.isin([0,2,4])))]



###############################################################################

### BOOLEAN INDEXING IN PANDAS VERSUS NUMPY
# http://pandas.pydata.org/pandas-docs/stable/indexing.html#boolean-indexing
# http://docs.scipy.org/doc/numpy-1.10.1/user/basics.indexing.html
"""
Boolean indexing works for both pandas dataframes/series and numpy arrays.
The only difference is the use of index columns in pandas df/series:
Because pandas references rows by index, the boolean array that performs the
indexing on a pandas dataframe must have the same indexes as the dataframe being indexed.
However, numpy arrays do not reference rows by index, and therefore there is no need
for the same indexing between the boolean index and the main array. The boolean
index should still have the same length as the main array in both pandas and numpy.
"""

import pandas as pd
import numpy as np

y = np.arange(35).reshape(5,7)
b = y>20
y[b]
y[b[:,5]]

# Read the documentations for the latest updates on how to index numpy arrays and pandas dataframes using boolean indexes.

### EXAMPLE
main_df = pd.DataFrame({'response': [-1,0,5,3,6,-2,4]}, index=[1,2,3,5,6,8,9])
main_array = np.array(main_df)

type(main_df)  # data frame
type(main_array)  # array

main_array.shape  # same shape
main_df.shape  # same shape

main_df[main_df > 0]  # note the result has the same shape as idexed array. When boolean index array is False, NaN values are returned.
main_array[main_array > 0]  # note that only the values corresponding to True values of the index array are retirned

# a good practice is to convert boolean indexes to numpy arrays to prevent clashing the internal indexes with the internal indexes of the original array/dataframe
bool_ind_array = np.array(main_df > 0)
bool_ind_df = pd.DataFrame(main_df > 0)  # note the internal index values of the boolean index

main_df[bool_ind_df]  # in this case this works because the index array was built from the original df, so their internal indexes are the same
main_array[bool_ind_df]  # This does not work as expected because the dataframe boolean index has internal indexes

# a hack is to convert the boolean index to numpy array to remove its internal index
main_array[bool_ind_array]  # this works because arrays do not have internal index
main_df[bool_ind_array]  # this works because arrays do not have internal index


###############################################################################
####################### MULTI-INDEXING AND SLICING ############################
###############################################################################
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
################################ VIEW VS COPY #################################
###############################################################################
# http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
dfmi = pd.DataFrame([list('abcd'),
                    list('efgh'),
                    list('ijkl'),
                    list('mnop')],
                    columns=pd.MultiIndex.from_product([['one','two'],
                                                        ['first','second']]))

"""
The following lines both result in similar outputs. However, the second syntax is preferred.

dfmi['one'] selects the first level of the columns and returns a DataFrame that is singly-indexed.
Then another python operation dfmi_with_one['second'] selects the series indexed by 'second' happens.

On the other hand, df.loc[:,('one','second')] passes a nested tuple of (slice(None),('one','second'))
to a single call to __getitem__.
This allows pandas to deal with this as a single entity.
This order of operations can be significantly faster, and allows one to index both axes if so desired.
"""

dfmi['one']['second']
dfmi.loc[:,('one','second')]

"""
Another reason why the .loc indexing is preferred to chained indexing is that
assigning to the product of chained indexing has inherently unpredictable results.


"""
value = 'k'
dfmi.loc[:,('one','second')] = value
# becomes
dfmi.loc.__setitem__((slice(None), ('one', 'second')), value)

# On the other hand
dfmi['one']['second'] = value
# becomes
dfmi.__getitem__('one').__setitem__('second', value)

"""
it’s very hard to predict whether __getitem__() will return a view or a copy
(it depends on the memory layout of the array, about which pandas makes no guarantees),
and therefore whether the __setitem__ will modify dfmi or a temporary object that gets thrown out immediately afterward.
"""

"""
On the other hand, concerned about the loc property in the first example.
But dfmi.loc is guaranteed to be dfmi itself with modified indexing behavior,
so dfmi.loc.__getitem__ / dfmi.loc.__setitem__ operate on dfmi directly.
Of course, dfmi.loc.__getitem__(idx) may be a view or a copy of dfmi.
"""


###################
# The order of evaluation matters on whether a copy is returned or a view is returned.

"""
You can control the action of a chained assignment via the option mode.chained_assignment,
which can take the values ['raise','warn',None], where showing a warning is the default.
"""
pd.set_option('mode.chained_assignment','warn')
# pd.set_option('mode.chained_assignment','raise')

dfb = pd.DataFrame({'a' : ['one', 'one', 'two',
                    'three', 'two', 'one', 'six'],
                    'c' : np.arange(7)})


# (a) First selecting a column of a df and subsetting its rows using a boolean array
# This will show the SettingWithCopyWarning but the frame values will be set
dfb['c'][dfb.a.str.startswith('o')] = 42

# (b) First subsetting a df rows using a boolean array and choosing a column
# This however is operating on a copy and will not work.
dfb[dfb.a.str.startswith('t')]['c'] = 42

# (c) use .loc .iloc .ix
# The correct access method is to use .loc[row_index, col_indexer] = value
dfc = pd.DataFrame({'A':['aaa','bbb','ccc'],'B':[1,2,3]})
dfc.loc[0,'A'] = 11

# (d) Creating a copy first and accessing its cells using chained indexing
# This can work at times, but is not guaranteed, and so should be avoided
dfc = dfc.copy()
dfc['A'][0] = 111

# (e) combination of .loc and column indexing
# This will not work at all, and so should be avoided
dfc.loc[0]['A'] = 1111

#######################################
# When does numpy create a copy versus a view?
# http://stackoverflow.com/questions/11524664/how-can-i-tell-if-numpy-creates-a-view-or-a-copy

"""
Two different arrays can share the same memory. This can cause some unexpected behavior.
For example, by changing one of the arrays which is the view of another array, the original array may be impacted too.
This page offers some strategies to identify if an array is a copy or view of another array.

This page explains this phenomena:
http://scipy.github.io/old-wiki/pages/Tentative_NumPy_Tutorial.html#Copies_and_Views

The most important point is that slicing an array returns a view of it.
This means that if we assign a value to an array that is a slice of another array,
the original array will change too.
"""


#######################################
### How to check if two arrays are sharing the same data or not
# http://stackoverflow.com/questions/11286864/is-there-a-way-to-check-if-numpy-arrays-share-the-same-data


#######################################
### There are three different scenarios when operating and manipulating array
# http://scipy.github.io/old-wiki/pages/Tentative_NumPy_Tutorial.html#Copies_and_Views
import numpy as np

### (a) no copy at all
a = np.arange(12)
b = a            # no new object is created
b is a           # a and b are two names for the same ndarray object

b.shape = 3,4    # changes the shape of a
a.shape

# Python passes mutable objects as references, so function calls make no copy.
def f(x):
    print id(x)

id(a)                           # id is a unique identifier of an object
f(a)


### (b) View or shallow copy
# Different array objects can share the same data. The view method creates a new array object that looks at the same data.
c = a.view()
c is a                             # c is not the same object as a but is a view of the data owned by a

c.base is a                        # c is a view of the data owned by a

c.flags.owndata

c.shape = 2,6                      # a's shape doesn't change
a.shape

c[0,4] = 1234                      # a's data changes
a

# slicing an array returns a view of it. So changing a slice will change the original array.
s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"
s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
a


### (c) Deep copy
d = a.copy()                          # a new array object with new data is created
d is a

d.base is a                           # d doesn't share anything with a

d[0,0] = 9999
a

# so instead of assigning an array to a new array when you want to do independent operations on them,
# copy the array to the new array and then run the operations.


