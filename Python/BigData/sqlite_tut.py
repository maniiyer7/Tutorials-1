

"""
NOTE: start a database by
(1) install sqlite3 on your command line. For Mac, use the command "brew install sqlite3"
(2) go to the directory where this script file is located and run "sqlite test.db"
 This will create a test.db binary file on whatever directory you ran it on.
 The test.db is the binary file containing your database.

Another option is to create the database from within Python.
The command below will create a database and give a connection object to it.
con = sqlite3.connect('sample.db')


REQUIREMENTS:
To work with this tutorial, we must have Python language, SQLite database,
pysqlite language binding and the sqlite3 command line tool installed on the system.
If we have Python 2.5+ then we only need to install the sqlite3 command line tool.
Both the SQLite library and the pysqlite language binding are built into the Python languge.


RUNNING sqlite in command line shell
$ sqlite test.db  # this will create the database if it does not exist, or opens it in sqlite command line tool if already exists
sqlite> .tables
sqlite> .exit


SOURCE
http://zetcode.com/db/sqlitepythontutorial/
https://docs.python.org/2/library/sqlite3.html
"""

###############################################################################
import sqlite3 as lite
import sys

con = None

try:
    con = lite.connect('test.db')

    cur = con.cursor()
    cur.execute('SELECT SQLITE_VERSION()')

    data = cur.fetchone()

    print "SQLite version: %s" % data

except lite.Error, e:

    print "Error %s:" % e.args[0]
    sys.exit(1)

finally:

    if con:
        con.close()


#######################################
"""
With the use of the with keyword. The code is more compact.
With the with keyword, the Python interpreter automatically releases the resources.
It also provides error handling.
"""

import sqlite3 as lite
import sys

con = lite.connect('test.db')

with con:

    cur = con.cursor()
    cur.execute('SELECT SQLITE_VERSION()')

    data = cur.fetchone()

    print "SQLite version: %s" % data


###############################################################################
### INSERTING DATA
"""
We will create a Cars table and insert several rows to it.
"""

import sqlite3 as lite
import sys

con = lite.connect('test.db')

with con:

    cur = con.cursor()
    cur.execute("CREATE TABLE Cars(Id INT, Name TEXT, Price INT)")
    cur.execute("INSERT INTO Cars VALUES(1,'Audi',52642)")
    cur.execute("INSERT INTO Cars VALUES(2,'Mercedes',57127)")
    cur.execute("INSERT INTO Cars VALUES(3,'Skoda',9000)")
    cur.execute("INSERT INTO Cars VALUES(4,'Volvo',29000)")
    cur.execute("INSERT INTO Cars VALUES(5,'Bentley',350000)")
    cur.execute("INSERT INTO Cars VALUES(6,'Citroen',21000)")
    cur.execute("INSERT INTO Cars VALUES(7,'Hummer',41400)")
    cur.execute("INSERT INTO Cars VALUES(8,'Volkswagen',21600)")


#######################################
"""
Create the same table. This time using the convenience executemany() method.
"""
import sqlite3 as lite
import sys

cars = (
    (1, 'Audi', 52642),
    (2, 'Mercedes', 57127),
    (3, 'Skoda', 9000),
    (4, 'Volvo', 29000),
    (5, 'Bentley', 350000),
    (6, 'Hummer', 41400),
    (7, 'Volkswagen', 21600)
)

con = lite.connect('test.db')

with con:

    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS Cars")
    cur.execute("CREATE TABLE Cars(Id INT, Name TEXT, Price INT)")
    cur.executemany("INSERT INTO Cars VALUES(?, ?, ?)", cars)  # use tuple of tuples for data


#######################################
"""
Manually do the exception handling (not using with)
Without the with keyword, the changes must be committed using the commit() method.
We also need to do rollback() if an error is thrown.
When using with function, we dont need to worry about commit, rollback, exception handling.
"""

import sqlite3 as lite
import sys

try:
    con = lite.connect('test.db')

    cur = con.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS Cars;
        CREATE TABLE Cars(Id INT, Name TEXT, Price INT);
        INSERT INTO Cars VALUES(1,'Audi',52642);
        INSERT INTO Cars VALUES(2,'Mercedes',57127);
        INSERT INTO Cars VALUES(3,'Skoda',9000);
        INSERT INTO Cars VALUES(4,'Volvo',29000);
        INSERT INTO Cars VALUES(5,'Bentley',350000);
        INSERT INTO Cars VALUES(6,'Citroen',21000);
        INSERT INTO Cars VALUES(7,'Hummer',41400);
        INSERT INTO Cars VALUES(8,'Volkswagen',21600);
        """)

    con.commit()

except lite.Error, e:

    if con:
        con.rollback()

    print "Error %s:" % e.args[0]
    sys.exit(1)

finally:

    if con:
        con.close()


###############################################################################
### GET LAST INSERTED ROW ID
"""
Use the lastrowid attribute of the cursor object to
get the ID of the last inserted row.
Also note the use of :memory: as argument to lite.connect() function.
"""

"""
In SQLite, INTEGER PRIMARY KEY column is auto incremented.
There is also an AUTOINCREMENT keyword.
When used in INTEGER PRIMARY KEY AUTOINCREMENT a slightly different algorithm for Id creation is used.
"""

import sqlite3 as lite
import sys

con = lite.connect(':memory:')  # to create a table in memory

with con:

    cur = con.cursor()
    cur.execute("CREATE TABLE Friends(Id INTEGER PRIMARY KEY, Name TEXT);")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Tom');")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Rebecca');")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Jim');")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Robert');")

    lid = cur.lastrowid
    print "The last Id of the inserted row is %d" % lid


###############################################################################
### FETCHING DATA
import sqlite3 as lite
import sys

con = lite.connect('test.db')

with con:

    cur = con.cursor()
    cur.execute("SELECT * FROM Cars")

    rows = cur.fetchall()

    for row in rows:
        print row

# Returning one by one
import sqlite3 as lite
import sys

con = lite.connect('test.db')

with con:

    cur = con.cursor()
    cur.execute("SELECT * FROM Cars")

    while True:

        row = cur.fetchone()

        if row == None:
            break

        print row[0], row[1], row[2]


###############################################################################
### THE DICTIONARY CURSOR
"""
The default cursor returns the data in a tuple of tuples.
When we use a dictionary cursor, the data is sent in the form of Python dictionaries.
This way we can refer to the data by their column names.
This is possible by setting the row_factory attribute of the connection to Row type.
"""

import sqlite3 as lite

con = lite.connect('test.db')

with con:

    con.row_factory = lite.Row  # set the row_factory attribute of the connection to Row type.

    cur = con.cursor()
    cur.execute("SELECT * FROM Cars")

    rows = cur.fetchall()

    for row in rows:
        print "%s %s %s" % (row["Id"], row["Name"], row["Price"])


###############################################################################
### PARAMETRIZED QUERIES
"""
The Python sqlite3 module supports two types of placeholders:
question marks and named placeholders.
The named placeholders start with a colon character.
"""

import sqlite3 as lite
import sys

uId = 1
uPrice = 62300

con = lite.connect('test.db')

with con:
    cur = con.cursor()
    cur.execute("UPDATE Cars SET Price=? WHERE Id=?", (uPrice, uId))
    con.commit()
    print "Number of rows updated: %d" % cur.rowcount


# using parameterized statements with named placeholders
import sqlite3 as lite
import sys

uId = 4

con = lite.connect('test.db')

with con:
    cur = con.cursor()
    cur.execute("SELECT Name, Price FROM Cars WHERE Id=:Id",
        {"Id": uId})
    con.commit()
    row = cur.fetchone()
    print row[0], row[1]


###############################################################################
### METADATA
"""
Metadata in SQLite can be obtained using the PRAGMA command.
SQLite objects may have attributes, which are metadata.
Finally, we can also obtain specific metatada from querying the SQLite system sqlite_master table.
Columns in the result set of PRAGMA command include
the column order number, column name, data type,
whether or not the column can be NULL, and the default value for the column.
"""

import sqlite3 as lite
import sys

con = lite.connect('test.db')

with con:
    cur = con.cursor()
    cur.execute('PRAGMA table_info(Cars)')
    data = cur.fetchall()
    for d in data:
        print d[0], d[1], d[2]


#######################################
"""
Print all rows from the Cars table with their column names.
"""

import sqlite3 as lite
import sys

con = lite.connect('test.db')

with con:
    cur = con.cursor()
    cur.execute('SELECT * FROM Cars')
    col_names = [cn[0] for cn in cur.description]
    rows = cur.fetchall()
    print "%s %-10s %s" % (col_names[0], col_names[1], col_names[2])

    for row in rows:
        print "%2s %-10s %s" % row


"""
list all tables in the test.db database using sqlite_master
"""
import sqlite3 as lite
import sys

con = lite.connect('test.db')

with con:
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    rows = cur.fetchall()
    for row in rows:
        print row[0]


###############################################################################
### EXPORT AND IMPORT OF DATA
"""
iterdump() function returns an iterator to dump the database in an SQL text format.
It is useful when saving an in-memory database for later restoration.
It dumps the entire database in a script that can be later run to recreate the original database.
"""
import sqlite3 as lite
import sys

def writeData(data):
    f = open('cars.sql', 'w')
    with f:
        f.write(data)

cars = (
    (1, 'Audi', 52643),
    (2, 'Mercedes', 57642),
    (3, 'Skoda', 9000),
    (4, 'Volvo', 29000),
    (5, 'Bentley', 350000),
    (6, 'Hummer', 41400),
    (7, 'Volkswagen', 21600)
)

con = lite.connect(':memory:')

with con:
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS Cars")
    cur.execute("CREATE TABLE Cars(Id INT, Name TEXT, Price INT)")
    cur.executemany("INSERT INTO Cars VALUES(?, ?, ?)", cars)
    cur.execute("DELETE FROM Cars WHERE Price < 30000")
    data = '\n'.join(con.iterdump())
    writeData(data)

"""
Now the reverse operation. We will import the dumped table back into memory
"""
import sqlite3 as lite
import sys

def readData(file_name):
    f = open(file_name, 'r')
    with f:
        data = f.read()
        return data


con = lite.connect(':memory:')

with con:
    cur = con.cursor()
    sql = readData('cars.sql')
    cur.executescript(sql)
    cur.execute("SELECT * FROM Cars")
    rows = cur.fetchall()
    for row in rows:
        print row


##############################################################################
#### TRANSACTIONS

"""
In SQLite, any command other than the SELECT will start an implicit transaction.
Also, within a transaction a command like CREATE TABLE ..., VACUUM, PRAGMA, will commit previous changes before executing.

Manual transactions are started with the BEGIN TRANSACTION statement and
finished with the COMMIT or ROLLBACK statements.
"""

"""
SQLite supports three non-standard transaction levels:
DEFERRED, IMMEDIATE and EXCLUSIVE.
SQLite Python module also supports an autocommit mode,
where all changes to the tables are immediately effective.
"""

import sqlite3 as lite
import sys

try:
    con = lite.connect('test.db')
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS Friends")
    cur.execute("CREATE TABLE Friends(Id INTEGER PRIMARY KEY, Name TEXT)")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Tom')")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Rebecca')")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Jim')")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Robert')")

    """
    If we comment out the following line,
    the table is created but the data is not written to the table.
    """
    con.commit()

except lite.Error, e:
    if con:
        con.rollback()

    print "Error %s:" % e.args[0]
    sys.exit(1)

finally:
    if con:
        con.close()


"""
Some commands implicitly commit previous changes to the database.
So in this example, since we already have a CREATE TABLE statement, we did not use a commit statement.
"""
import sqlite3 as lite
import sys


try:
    con = lite.connect('test.db')
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS Friends")
    cur.execute("CREATE TABLE Friends(Id INTEGER PRIMARY KEY, Name TEXT)")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Tom')")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Rebecca')")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Jim')")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Robert')")

    cur.execute("CREATE TABLE IF NOT EXISTS Temporary(Id INT)")

except lite.Error, e:

    if con:
        con.rollback()

    print "Error %s:" % e.args[0]
    sys.exit(1)

finally:

    if con:
        con.close()


#######################################
"""
In the autocommit mode, an SQL statement is executed immediately.
We have an autocommit mode, when we set the isolation_level to None.
con = lite.connect('test.db', isolation_level=None)
"""
import sqlite3 as lite
import sys

try:
    con = lite.connect('test.db', isolation_level=None)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS Friends")
    cur.execute("CREATE TABLE Friends(Id INTEGER PRIMARY KEY, Name TEXT)")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Tom')")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Rebecca')")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Jim')")
    cur.execute("INSERT INTO Friends(Name) VALUES ('Robert')")


except lite.Error, e:
    print "Error %s:" % e.args[0]
    sys.exit(1)

finally:
    if con:
        con.close()





