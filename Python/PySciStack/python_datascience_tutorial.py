
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


