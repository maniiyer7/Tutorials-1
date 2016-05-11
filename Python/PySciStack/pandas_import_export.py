
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

