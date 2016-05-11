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
################################### APPLY #####################################
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
### APPLYMAP

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


