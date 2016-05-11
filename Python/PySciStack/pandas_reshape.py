


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
