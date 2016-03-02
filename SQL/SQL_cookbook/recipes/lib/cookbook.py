# cookbook.py: library file with utility method for connecting to MySQL
# using the Connector/Python module

import MySQLdb
conn_params = {
	"db": "cookbook",
	"host": "localhost",
	"user": "cbuser",
	"passwd": "cbpass",
	}
# Establish a connection to the cookbook database, returning a connection
# object. Raise an exception if the connection cannot be established.
def connect():
	return MySQLdb.connect(**conn_params)

