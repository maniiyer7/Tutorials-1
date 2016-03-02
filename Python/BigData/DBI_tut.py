
import cx_Oracle
import MySQLdb
import numpy as np
import pandas as pd

###############################################################################
################################# ORACLE ######################################
###############################################################################

# Creds for new BI prod
oracleUser = 'EDWR'
oraclePwd = 'Sunrun123'
oracleTns = '(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=10.2.40.146)(PORT=1521))(CONNECT_DATA=(SERVER=DEDICATED)(SERVICE_NAME=BIPRODDB.sunrun.com)))'

# Establish connection
data=[]
con = cx_Oracle.connect(oracleUser+'/'+oraclePwd+'@'+oracleTns)
cursor = con.cursor()

# Form and execute the query
# etl_repgen_peer_values
sql="""SELECT
                *
            FROM
                etl_repgen_peer_values
            WHERE
                SERVICE_CONTRACT_NAME = (2024005578)
            """

cursor.execute(sql)

# Add results to python data table
data.extend(cursor.fetchall())

# Get table column names
desc = cursor.description
columns = [i[0] for i in cursor.description]

# Close the connection and cursor
cursor.close()
con.close()

# Option a: Convert the data table to NumPy ndarray
# arrayDtype = [('service_contract_name','O')]
arrayDtype = [(x,'O') for x in columns]
data_array = np.array([tuple(x) for x in data], dtype=arrayDtype)

# Option b: Convert the data table to pandas DataFrame
import pandas as pd
data_df = pd.DataFrame(data)
data_df.columns = columns

#*#TODO: write a function that takes sql queries and returns data frame with proper column data types.

# Examine different columns
data_df.SRC_DELETED
data_df.SRC_DELETED.dtype
###############################################################################

###############################################################################
################################## MySQL ######################################
###############################################################################

###############################################################################
# Creds for monitoring_db
db1Host = 'corp-misc.cgk3erx0zg16.us-west-1.rds.amazonaws.com'
db1UName = 'gbruer'
db1Pwd = 'ohz(eiQu4uqu4Ci'
db1Name = 'performance_ops'

# Establish connection
data=[]
db = MySQLdb.connect(host=db1Host,user=db1UName,passwd=db1Pwd,db=db1Name)
cursor = db.cursor()

# Form and execute the query
# monitoring_db.ae_alert_type
# monitoring_db.ae_alerts
sql = """SELECT
            *
         FROM
            monitoring_db.ae_alert_type
         LIMIT 10
            """

cursor.execute(sql)

# Add results to python data table
data.extend(cursor.fetchall())

# Get table column names
desc = cursor.description
columns = [i[0] for i in cursor.description]

# Close the connection and cursor
cursor.close()
db.close()

# Convert the data table to NumPy ndarray
# arrayDtype = [('service_contract_name','O')]
arrayDtype = [(x,'O') for x in columns]
data_array = np.array([tuple(x) for x in data], dtype=arrayDtype)

# Convert the data table to pandas DataFrame
data_df = pd.DataFrame(data)
data_df.columns = columns

#*#TODO: write a function that takes sql queries and returns data frame with proper column data types.

###############################################################################
