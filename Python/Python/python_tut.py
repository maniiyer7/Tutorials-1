###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: April 26, 2015
# summary: python playground
###############################################################################

import cx_Oracle
import numpy as np
import pandas as pd


###############################################################################
################################# PYTHON ######################################
###############################################################################

### IMPORTING
import numpy as np  # recommended
import numpy  # not recommended. You will have to spell our numpy for every function you want to use.
from numpy import *  # not recommended. This will add numpy to the namespace, which means you will not have to spell out numpy in function calls. But that can cause confusing function calls.
import datetime as dt
import pytz


###############################################################################
### MUTABLE AND IMMUTABLE DATA TYPES
test = (1,2)
test = 'message'
message = 'test'
message[2] = 'b'
###############################################################################

###############################################################################
for i in range(5):
    print i

# enumerate() binds an index to elements of an object
elements = ('foo', 'bar', 'baz')
for elem in elements:
    print elem
for idx, elem in enumerate(elements):
    print idx, elem
###############################################################################

###############################################################################
# pd.DataFrame(zip(('sin1', 'cos1', 'sin2', 'cos2'), lm.coef_))
# print lm.coef_
# print lm.intercept_
###############################################################################

###############################################################################
### CREATING LISTS
# List comprehensions: they work like a loop that populates a list with a specific pattern
nums = [str(n) for n in range(20)]
nums

some_list = [None] * 5
four_lists = [[] for __ in xrange(4)]
four_lists = [n for n in xrange(0,8,2)]
for index, item in enumerate(four_lists):
    print index
    print item

###############################################################################

###############################################################################
### CREATING STRINGS
# A common idiom for creating strings is to use str.join() on an empty string.
letters = ['s', 'p', 'a', 'm']
word = ''.join(letters)
###############################################################################

###############################################################################
### DICTIONARY
d = {'s': [1,2,4], 'p': ['just chilling'], 'a': [5,7], 'm': ['this is a test']}
l = ['s', 'p', 'a', 'm']
# To check whether a key exists in a dictionary
print 's' in d  # much faster
print 's' in l  # list is not as fast as dictionary
d.has_key('s')
print d.get('s', 'default_value')
print d.get('d', 'default_value')
print d['s']
###############################################################################

###############################################################################
################################ DATA TYPES ###################################
###############################################################################

### tuples vs lists
# Tuples are fixed size in nature whereas lists are dynamic.
# In other words, a tuple is immutable whereas a list is mutable.
# You can't add elements to a tuple. Tuples have no append or extend method.
# You can't remove elements from a tuple. Tuples have no remove or pop method.
# You can find elements in a tuple, since this doesn’t change the tuple.
# You can also use the in operator to check if an element exists in the tuple.
# Tuples have an order associated with elements. Lists are not ordered.
# Tuples are faster than lists. If you're defining a constant set of values and all you're ever going to do with it is iterate through it, use a tuple instead of a list.
# For collections of heterogeneous objects (like a address broken into name, street, city, state and zip) I prefer to use a tuple. They can always be easily promoted to named tuples.
# If the collection is going to be iterated over, I prefer a list. If it's just a container to hold multiple objects as one, I prefer a tuple
# Tuples are heterogeneous data structures (i.e., their entries have different meanings), while lists are homogeneous sequences.
# Tuples have structure, lists have order.


# python dictionary is similar to R list
# python list is similar to R vector; i.e., c() object
# python tuple is a vector object (similar to python list) that is optimized for performance (at the cost of flexibility)
#   You cant add elements to a tuple. Tuples have no append or extend method.
#   You cant remove elements from a tuple. Tuples have no remove or pop method.
#   You can find elements in a tuple, since this does not change the tuple.
#   You can also use the in operator to check if an element exists in the tuple.
#   Tuples are faster than lists. If you are defining a constant set of values and all you are ever going to do with it is iterate through it, use a tuple instead of a list.
#   It makes your code safer if you “write-protect” data that does not need to be changed. Using a tuple instead of a list is like having an implied assert statement that this data is constant, and that special thought (and a specific function) is required to override that.
#   Some tuples can be used as dictionary keys (specifically, tuples that contain immutable values like strings, numbers, and other tuples). Lists can never be used as dictionary keys, because lists are not immutable.
#   Tuples are heterogeneous data structures (i.e., their entries have different meanings), while lists are homogeneous sequences. Tuples have structure, lists have order.
#   Use lists when you want to iterate over its items, because (a) they are heterogous, and (b) they have order.

# The point of a tuple is that the i-th slot means something specific. In other words, it's a index-based (rather than name based) datastructure.
# http://news.e-scribe.com/397
# http://stackoverflow.com/questions/626759/whats-the-difference-between-list-and-tuples


###############################################################################
### LISTS
# http://effbot.org/zone/python-list.htm
# List comprehensions prevent the use of for loops. They are similar to R apply() functions.

# filter() function can be used to select a subset of the list based on a criteria. It is similar to R subset() function.
a = [3, 4, 5, 6, 7, 8]
# [lambda x, i=i : i * x for i in range(5)]
b = [i for i in a if i > 4]
c = filter(lambda x: x < 7, a)  # select a subset

# map() function can be used to apply functions to all elements of a list
a = [i + 3 for i in a]
a = map(lambda i: i + 3, a)  # apply a function

# enumerate() function gives an iterator and the individual element in the list.
a = [3, 4, 5]
for i, item in enumerate(a):
    print i, item
###############################################################################

###############################################################################
###############################################################################
###############################################################################

###############################################################################
### READING FROM FILES
# The with statement is better because it will ensure you always close the file, even if an exception is raised inside the with block.
with open('/Users/amirkavousian/gits/performance-engineering/deployment/scripts/CaseHandling/test-delete.py') as f:
    for line in f:
        print line
###############################################################################

### MULTI-LINE COMMANDS
# Break across lines using parantheses
from numpy import (
    stats)

###############################################################################
############################### WEB SCRAPING ##################################
###############################################################################
# modules: lxml, Requests, urllib2
# functions: XPath, CSSSelect
from lxml import html
import requests
page = requests.get("http://amirkavousian.com/publications")
tree = html.fromstring(page.text)

###############################################################################
######################### COMMAND-LINE APPLICATIONS ###########################
###############################################################################
# modules: clint, click, docopt,
# frameworks: Plac, Cliff,

###############################################################################
############################## GUI APPLICATIONS ###############################
###############################################################################
# frameworks: Camelot, Cocoa, GTk, Kivy, PyObjC, PySide, PyQt, Pyjs, Qt, Tk,
# toolkit: wxPython

###############################################################################
################################# DATABASES ###################################
###############################################################################
# DB-API, SQLAlchemy, DjangoORM

###############################################################################
################################ NETWORKING ###################################
###############################################################################
# engines: Twisted, PyZMQ,
# modules: gevent

###############################################################################
############################### SYSTEMS ADMIN #################################
###############################################################################
# Fabric, Psutil, Ansible, Chef, Puppet, Blueprint, Buildout,

###############################################################################
########################## CONTINUOUS INTEGRATION #############################
###############################################################################
# engines: Jenkins CI, Buildbot, Mule, Tox, Travis-CI,


###############################################################################
################################# LOGGING #####################################
###############################################################################
import traceback, logging

################################################
### (option a) Define a logger-initiator function using Python logging module
def init_logger(name):
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./app_log.log',
                    filemode='a')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO) # only reveal info level to console
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s') # set a format which is simpler for console use
    console.setFormatter(formatter) # tell the handler to use this format
    mLogger = logging.getLogger(name) # name logger
    mLogger.addHandler(console) # allow for logger to output to console
    return mLogger
################################################

################################################
### (option b - DEPRECATED) Define a logging function manually (not using logging module)
import os, sys, re, json
from datetime import datetime as dt
import traceback

unixFormat = '%Y-%m-%d %H:%M:%S'

def log_error(excInfo,custMessage=None):
    currTime = dt.now().strftime(unixFormat)
    currTimeFormat = ""+currTime+': '
    exc_type, exc_value, exc_traceback = excInfo
    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    msg = ''.join(currTimeFormat + line for line in lines)
    f = open('./error_log', 'a+')
    f.write(msg)
    if custMessage is not None:
        f.write(currTimeFormat+custMessage+"\r\n-------------------\r\n")
################################################

### How to implement:
# Instantiate logging
mLogger = init_logger('loggint_tutorial')

# Use traceback to get the error msg and where the error occured.
# Use the logging function to write the log to file.
try:
    tuple()[0]
except:
    fm = traceback.format_exc().splitlines()
    # print fm[-1]+' '+fm[1]
    mLogger.error(fm[-1]+' '+fm[1])

# More on logging and traceback:
# https://docs.python.org/2/library/traceback.html
# https://docs.python.org/2/library/logging.html

###############################################################################
################################# TESTING #####################################
###############################################################################


