#!/usr/local/bin/python
###############################################################################
# author: @amirkav based on @gbruer work
# email: amir.kavousian@sunrun.com
# created on: Nov 1, 2015
# summary: functions to call MySQL from Python
###############################################################################

###############################################################################
############################### LOAD MODULES ##################################
###############################################################################
### LOAD STANDARD MODULES
from datetime import datetime as dt
import ConfigParser, os, sys, traceback, MySQLdb, pytz
import numpy as np
import pCommon as com

### LOAD CUSTOM MODULES
# Detect interactive mode (iMode)
if not os.isatty(sys.stdout.fileno()):
    print 'Python is in interactive mode. __file__ and other OS parameters may not be available to Python.'
    __file__ = '/Users/amirkavousian/Documents/Py_Codes/Tutorials/BigData/Mysql_tut.py'
    iMode = True
else:
    iMode = False

print __file__

db1Host = 'localhost'
db1UName = 'cbuser'
db1Pwd = 'cbpass'
db1Name = 'cookbook'

localtz = pytz.timezone('America/Los_Angeles')

unixFormat='%Y-%m-%d %H:%M:%S'
oracleFormat='%Y%m%d'
sqlUnixFormat = 'YYYY-MM-DD HH24:MI:SS'
maxInsertChunk = 1000

###############################################################################

###############################################################################
############################ STANDARD FUNCTIONS ###############################
###############################################################################
def connect_to_db(dbHost, dbUser, dbPwd, dbName):
    # function to guarantee connection to DB
    connected = False
    # Open database connection
    while connected is False:
        try:
            db = MySQLdb.connect(host=dbHost,user=dbUser,passwd=dbPwd,db=dbName)
            connected=True
        except MySQLdb.Error, e:
            fm = traceback.format_exc().splitlines()
            print(fm[-1]+' '+fm[1])
            connected = False

    return db
################################################

################################################
def ensure_connection(conn, host, uName, pwd, name):
    if conn.open:
        pass
    else:
        conn = connect_to_db(host,uName,pwd,name)
    return conn
################################################

################################################
def mysql_escape(conn, val):
    if val is None:
        return 'NULL'
    else:
        return "'"+conn.escape_string(val)+"'"
################################################


################################################
def user_defined_query(sql, arrayDtype = None, printErr = False):

    db = connect_to_db(db1Host,db1UName,db1Pwd,db1Name)
    try:
        cursor = db.cursor()
        cursor.execute(sql)

        # get insert ID if an insert query
        if "insert" in sql.lower():
            firstId = cursor.lastrowid

        # get results set if query is not an insert query, otherwise commit the changes to database.
        if(arrayDtype is not None):
            data = cursor.fetchall()
        else:
            db.commit()

        # disconnect from server
        db.close()
    except:
        fm = traceback.format_exc().splitlines()
        print fm[-1]+' '+fm[1]

        # disconnect from server
        db.close()
        return False

    if(arrayDtype is not None):
        return np.array(map(tuple,data), dtype=arrayDtype)
    else:
        if "insert" in sql.lower():
            return firstId
        else:
            return True
################################################

###############################################################################
############################# MODULE FUNCTIONS ################################
###############################################################################


def get_perf_decline_estimates():

    sql = """
    SELECT subject, age, sex, score FROM cookbook.testscore ORDER BY subject;
    """

    data=[]
    db = connect_to_db(db1Host,db1UName,db1Pwd,db1Name)
    cursor = db.cursor()

    try:
        cursor.execute(sql)
        data = cursor.fetchall()
    except Exception:
        fm = traceback.format_exc().splitlines()
        print fm[-1]+' '+fm[1]
        return False

    # disconnect from server
    cursor.close()
    db.close()
    arrayDtype=[('asset_key','O'),
                ('service_contract_name','O'),
                ('degr_ens','O'),
                ('int_ref','O'),
                ('shading_loss_kwh_per_kw','O'),
                ('soiling_loss_kwh_per_kw','O')]

    return np.array(list(data), dtype=arrayDtype)
#######################################


#######################################
def insert_weather_sites(sites, maxInsertChunk=500):

    if sites is None:
        sites = []

    # chunk list of assets to avoid dropped queries
    chunks = com.chunk_list(sites, maxInsertChunk)

    db = connect_to_db(db1Host,db1UName,db1Pwd,db1Name)
    totalFailed, insertedRows, updatedRows = 0, 0, 0
    currTimeStr = dt.utcnow().strftime(unixFormat)

    for chunk in chunks:
        db = ensure_connection(db, db1Host, db1UName, db1Pwd, db1Name)
        cursor = db.cursor()

        # get current num rows for calcs (we need this to calculate how many rows were inserted)
        s = """SELECT COUNT(id) FROM monitoring_db.noc_weather_sites"""
        cursor.execute(s)
        firstId = cursor.fetchone()[0]

        alertValuesString = ", ".join(["('{cpr_id}', {lat}, {lng}, '{curr_date}')"
                                           .format(cpr_id = site.get('cpr_id'),
                                                   lat = site.get('lat'),
                                                   lng = site.get('lng'),
                                                   curr_date = currTimeStr)
                                       for site in chunk])

        try:
            sql = """INSERT INTO monitoring_db.noc_weather_sites
                        (cpr_id,
                        lat,
                        lng,
                        date_created)
                    VALUES {alert_values_insert}
                    ON DUPLICATE KEY UPDATE date_created=CURRENT_TIME()
                    """.format(alert_values_insert = alertValuesString)

            cursor.execute(sql)
            affectedRows = cursor.rowcount

            # again, get current num rows for calcs (we need this to calculate how many rows were inserted)
            s = """SELECT COUNT(id) FROM monitoring_db.noc_weather_sites"""
            cursor.execute(s)
            lastId = cursor.fetchone()[0]

            ins = lastId - firstId
            updatedRows += (affectedRows - ins)/2
            insertedRows += ins

            db.commit()
        except Exception:
            fm = traceback.format_exc().splitlines()
            print fm[-1]+' '+fm[1]
            totalFailed += len(chunk)
            db.rollback() # Rollback in case there is any error

    db.close()

    return {'status' : totalFailed==0,
            'total_inserted' : int(insertedRows),
            'total_updated' : int(updatedRows),
            'total_succeeded' : len(sites) - totalFailed,
            'total_failed' : totalFailed}
################################################

