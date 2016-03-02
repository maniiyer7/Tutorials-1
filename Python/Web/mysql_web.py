#!/usr/local/bin/python
###############################################################################
# author: @amirkav based on @gbruer work
# email: amir.kavousian@sunrun.com
# created on: Nov 21, 2015
# summary: functions to interact with MySQL from Web
###############################################################################

###############################################################################
############################### LOAD MODULES ##################################
###############################################################################
### LOAD STANDARD MODULES
from datetime import datetime as dt
import ConfigParser, os, sys, traceback, MySQLdb, pytz
import numpy as np
import urllib, cgi
import imp  # for importing costume modules (other python scripts written by me)

import pCommon as com


### LOAD CUSTOM MODULES
# Detect interactive mode (iMode)
if not os.isatty(sys.stdout.fileno()):
    print 'Python is in interactive mode. __file__ and other OS parameters may not be available to Python.'
    __file__ = '/Users/amirkavousian/Documents/Py_Codes/Tutorials/Web/mysql_web.py'
    iMode = True
else:
    iMode = False

print __file__


# One way of importing a costume module is by adding the file directory to system path, then importing it via import command.
gpDir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(gpDir+'/BigData')
import Mysql_tut

# Another way to import a costume module is by using the imp module.
pWebUtils = imp.load_source('webutils', '/Users/amirkavousian/Documents/SQL_Codes/Tutorials/recipes/lib/cookbook_webutils.py')

sys.path.append('/Users/amirkavousian/Documents/SQL_Codes/Tutorials/recipes/lib')
import cookbook
import cookbook_utils
import cookbook_webutils


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
### ENCODING (Mysql Cookbook example from Ch. 18)
urllib.quote("""this is a < sign""")
cgi.escape("""this is a < sign""", quote=None)

import cgi
import urllib

sql = "SELECT phrase_val FROM phrase ORDER BY phrase_val"

db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
cursor = db.cursor()
cursor.execute(sql)

for (phrase,) in cursor:
    # make sure that the value is a string
    phrase = str(phrase)
    # URL-encode the phrase value for use in the URL
    url = "/cgi-bin/mysearch.py?phrase=" + urllib.quote(phrase)
    # HTML-encode the phrase value for use in the link label
    label = cgi.escape(phrase, 1)  # parameter 1 tells cgi.escape() to convert quotes to &quote; as well
    print(phrase)
    print('<a href="%s">%s</a><br />' % (url, label))

cursor.close()
db.close()
###############################################################################


###############################################################################
### PREPARING MySQL RESULTS FOR WEB DISPLAY (PARAGRAPH)
# Mysql Cookbook example from ch. 19

db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
cursor = db.cursor()

cursor.execute("SELECT NOW(), VERSION(), DATABASE()")
(now, version, db_name) = cursor.fetchone()

cursor.close()
db.close()

if db_name is None:
    db_name = 'NONE'

para = "Local time on the MySQL server is %s." % now
print("<p>%s</p>" % cgi.escape(para, 1))

para = "The server version is %s." % version
print("<p>%s</p>" % cgi.escape(para, 1))

para = "The default database is %s." % db
print("<p>%s</p>" % cgi.escape(para, 1))
###############################################################################


###############################################################################
### PREPARING MySQL RESULTS FOR WEB DISPLAY (LIST)
def make_ordered_list(items, encode=True):
    '''
    function to format query result set into a HTML list, ready for displaying in a HTML page
    '''
    result = ""
    for item in items:
        if item is None: # handle possibility of NULL item
            item = ""
        # make sure item is a string, then encode if necessary
        item = str(item)
        if encode:
            item = cgi.escape(item, 1)
        result += "<li>" + item + "</li>"
    return "<ol>" + result + "</ol>"



# fetch items for list
stmt = "SELECT item FROM ingredient ORDER BY id"

db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
cursor = db.cursor()
cursor.execute(stmt)
data = cursor.fetchall()
cursor.close()

# generate HTML list
print(make_ordered_list(data))
###############################################################################

###############################################################################
### PREPARING MySQL RESULTS FOR WEB DISPLAY (DEFINITION LIST)

def make_definition_list(items, encode=True):
    '''
    Function to format query result set into a definition list to be displayed as a HTML.
    It receives a list of tuples.
    '''
    result = ""
    for item in items:
        # make sure item is a string, then encode if necessary
        item = tuple([str(x) for x in list(item)])
        if encode:
            item = tuple([cgi.escape(x,1) for x in list(item)])
        result += "<dt>" + item[0] + "</dt>" + " <dd>" + item[1] + "</dd>"
    return "<dl>" + result + "</dl>"

sql = "SELECT note, mnemonic FROM doremi ORDER BY id"

db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
cursor = db.cursor()
cursor.execute(sql)
data = cursor.fetchall()
cursor.close()

# generate HTML list
print(make_definition_list(data))
###############################################################################

###############################################################################
### DISPLAYING QUERY RESULTS AS HTML TABLE

def create_table_cell(it):
    return "<td>" + it + "</td>"


def make_html_table(items, result=None, encode=True):
    '''
    Function to format query result set into a definition list to be displayed as a HTML.
    It receives a list of tuples.
    '''

    if result is None:
        result = ''

    for item in items:
        result += '<tr>'

        # make sure item is a string, then encode if necessary
        item = tuple([str(x) for x in list(item)])
        if encode:
            item = tuple([cgi.escape(x,1) for x in list(item)])
        result += ''.join(tuple([create_table_cell(x) for x in list(item)]))

        result += '</tr>'

    return "<dl>" + result + "</dl>"


result = """
<table border="1">
    <tr>
        <th>Year</th>
        <th>Artist</th>
        <th>Title</th>
    </tr>"""

sql = "SELECT note, mnemonic FROM doremi ORDER BY id"

db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
cursor = db.cursor()
cursor.execute(sql)
data = cursor.fetchall()
cursor.close()

# generate HTML list
print(make_html_table(data, result))
###############################################################################

###############################################################################
### DISPLAY URL VALUES AS HYPERLINKS
stmt = "SELECT name, website FROM book_vendor ORDER BY name"

db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
cursor = db.cursor()
cursor.execute(stmt)
data = cursor.fetchall()

def create_hyperlink(item):
    return '<a href="http://%s">%s</a>' % (urllib.quote(item[0]), cgi.escape(item[1], 1))

items = map(create_hyperlink, data)

cursor.close()
db.close()

# print items, but don't encode them; they're already encoded
print(items)
###############################################################################

###############################################################################
### STORING IMAGE FILES IN MySQL
# http://stackoverflow.com/questions/1294385/how-to-insert-retrieve-a-file-stored-as-a-blob-in-a-mysql-db-using-python
# http://stackoverflow.com/questions/22141718/inserting-and-retrieving-images-into-mysql-through-python
# http://stackoverflow.com/questions/5112072/saving-binary-data-to-mysqldb-python

from PIL import Image
import base64
import cStringIO
import PIL.Image


image = Image.open('/Users/amirkavousian/Documents/AMIR/Website/Appearence/Logo.png')
blob_value = open('/Users/amirkavousian/Documents/AMIR/Website/Appearence/Logo.png', 'rb').read()

db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
cursor=db.cursor()

# NOTE: because the image BLOB special characters, you need to pass the blob_value in a separate param variable to the cursor.execute() function.
# Ie, you cannot use the .format(blob=blob_value) that we normally use to create sql statement. You cannot create the sql statement as a stand-alone string and pass it to the cursor; instead,
# you need to pass the BLOB value as an argument to the cursor.execute() function.
sql = """INSERT INTO cookbook.image (name, type, data) VALUES(%s, %s, %s)"""
params = ('Logo.png', 'png', blob_value)
cursor.execute(sql, params)
db.commit()

cursor.close()
db.close()


#####################################
### DISPLAY AN IMAGE FROM A MySQL TABLE
# http://stackoverflow.com/questions/8317421/retrieve-and-displaying-blob-images-from-mysql-database-with-tkinter

sql1='select * from image'

db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
cursor=db.cursor()
cursor.execute(sql1)
data=cursor.fetchall()

# Note: change the indexes based on the table setup. You want to choose the column with the BLOB value.
print type(data[0][3])

file_like=cStringIO.StringIO(str(data[0][3]))
img=PIL.Image.open(file_like)
img.show()
###############################################################################


###############################################################################
### CREATE A SELECTION LIST
# NOTE: see chapter 20 of Cookbook for how to modify this code to create different types of selection lists.

stmt = "SELECT color FROM cow_color ORDER BY color"
db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
cursor = db.cursor()
cursor.execute(stmt)


# Print the selection list, one-by-one
print('<select name="color">')

for (color, ) in cursor:
    color = cgi.escape(color, 1)
    print('<option value="%s">%s</option>' % (color, color))

print('</select>')


cursor.close()
db.close()
###############################################################################

###############################################################################
### RADIO BUTTON
# <input type="radio" name="size" value="small" />small
# <input type="radio" name="size" value="medium" checked="checked" />medium
# <input type="radio" name="size" value="large" />large


### CHECKBOX
# <input type="checkbox" name="accessories" value="cow bell" checked="checked" />cow bell
# <input type="checkbox" name="accessories" value="horns" checked="checked" />horns
# <input type="checkbox" name="accessories" value="nose ring" />nose ring
# <input type="checkbox" name="accessories" value="tail ribbon" />tail ribbon


### POP-UP MENU
# <select name="color">
# <option value="Black">Black</option>
# <option value="Black &amp; White">Black &amp; White</option>
# <option value="Brown">Brown</option>
# <option value="Cream">Cream</option>
# <option value="Red">Red</option>
# <option value="Red &amp; White">Red &amp; White</option>
# <option value="See-Through">See-Through</option>
# </select>


### SCROLLING LISTS
# <select name="state" size="6">
# <option value="AL">Alabama</option>
# <option value="AK">Alaska</option>
# <option value="AZ">Arizona</option>
# <option value="AR">Arkansas</option>
# <option value="CA">California</option>
# ...
# <option value="WI">Wisconsin</option>
# <option value="WY">Wyoming</option>
# </select>


### MULTIPLE-SELECTION SCROLLING LIST
# <select name="accessories" size="3" multiple="multiple">
# <option value="cow bell" selected="selected">cow bell</option>
# <option value="horns" selected="selected">horns</option>
# <option value="nose ring">nose ring</option>
# <option value="tail ribbon">tail ribbon</option>
# </select>


### TABLE IN HTML
# <tr>
#   <th>rec_id</th>
#   <th>name</th>
#   <th>trav_date</th>
#   <th>miles</th>
# </tr>

###############################################################################


###############################################################################
### CREATE A ROLLING LIST

stmt = "SELECT abbrev, name FROM states ORDER BY name"
db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
cursor = db.cursor()
cursor.execute(stmt)


print('<select name="state">')

for (abbrev, name) in cursor:
    abbrev = cgi.escape(abbrev, 1)
    name = cgi.escape(name, 1)
    print('<option value="%s">%s</option>' % (abbrev, name))

print('</select>')


cursor.close()
db.close()
###############################################################################

###############################################################################
### UTILITY FUNCTIONS TO GET TABLE DATA
import re
# NOTE: functions are reproduced from /recipes/lib/cookbook_utils.py
def get_enumorset_info(conn, db_name, tbl_name, col_name):
  cursor = conn.cursor()
  stmt = '''
         SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT
         FROM INFORMATION_SCHEMA.COLUMNS
         WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s
         '''
  cursor.execute(stmt, (db_name, tbl_name, col_name))
  row = cursor.fetchone()
  cursor.close()
  if row is None: # no such column
    return None

  # create dictionary to hold column information
  info = {'name': row[0]}
  # get data type string; make sure it begins with ENUM or SET
  s = row[1]
  p = re.compile("(ENUM|SET)\((.*)\)$", re.IGNORECASE)
  match = p.match(s)
  if not match: # not ENUM or SET
    return None
  info['type'] = match.group(1)    # data type

  # get values by splitting list at commas, then applying a
  # quote-stripping function to each one
  s = match.group(2).split(',')
  f = lambda x: re.sub("^'(.*)'$", "\\1", x)
  info['values'] = map(f, s)

  # determine whether column can contain NULL values
  info['nullable'] = (row[2].upper() == 'YES')

  # get default value (None represents NULL)
  info['default'] = row[3]
  return info
###############################################################################


###############################################################################
### UTILITY FUNCTIONS TO CREATE WEB LISTS
# NOTE: functions are reproduced from /recipes/lib/cookbook_webutils.py
import types
import cgi

#######################################
def make_radio_group(name, values, labels, default, vertical):
  result = ''
  # make sure name and default are strings
  name = str(name)
  default = str(default)
  for i in range(len(values)):
    # make sure value and label are strings
    value = str(values[i])
    label = str(labels[i])
    # select the item if it corresponds to the default value
    if value == default:
      checked = ' checked="checked"'
    else:
      checked = ''
    result += '<input type="radio" name="%s" value="%s"%s />%s' % (
                cgi.escape(name, 1),
                cgi.escape(value, 1),
                checked,
                cgi.escape(label, 1))
    if vertical:
      result += '<br />'  # display items vertically
  return result
#######################################

#######################################
def make_checkbox_group(name, values, labels, default, vertical):
  if type(default) not in (types.ListType, types.TupleType):
    default = [default]   # convert scalar to list
  result = ''
  # make sure name is a string
  name = str(name)
  for i in range(len(values)):
    # make sure value and label are strings
    value = str(values[i])
    label = str(labels[i])
    checked = ''
    for d in default:
      d = str(d)
      if value == d:
        checked = ' checked="checked"'
        break
    result += '<input type="checkbox" name="%s" value="%s"%s />%s' % (
                cgi.escape(name, 1),
                cgi.escape(value, 1),
                checked,
                cgi.escape(label, 1))
    if vertical:
      result += '<br />'  # display items vertically
  return result
#######################################

#######################################
def make_popup_menu(name, values, labels, default):
  result = ''
  # make sure name and default are strings
  name = str(name)
  default = str(default)
  for i in range(len(values)):
    # make sure value and label are strings
    value = str(values[i])
    label = str(labels[i])
    # select the item if it corresponds to the default value
    if value == default:
      checked = ' selected="selected"'
    else:
      checked = ''
    result += '<option value="%s"%s>%s</option>' % (
                cgi.escape(value, 1),
                checked,
                cgi.escape(label, 1))

  result = '<select name="%s">%s</select>' % (
                 cgi.escape(name, 1), result)
  return result
#######################################

#######################################
def make_scrolling_list(name, values, labels, default, size, multiple):
  if type(default) not in (types.ListType, types.TupleType):
    default = [default]   # convert scalar to list
  result = ''
  # make sure name and size are strings
  name = str(name)
  size = str(size)
  for i in range(len(values)):
    # make sure value and label are strings
    value = str(values[i])
    label = str(labels[i])
    # select the item if it corresponds to one of the default values
    checked = ''
    for d in default:
      d = str(d)
      if value == d:
        checked = ' selected="selected"'
        break
    result += '<option value="%s"%s>%s</option>' % (
                cgi.escape(value, 1),
                checked,
                cgi.escape(label, 1))

  if multiple:
    multiple = ' multiple="multiple"'
  else:
    multiple = ''
  result = '<select name="%s" size="%s"%s>%s</select>' % (
             cgi.escape(name, 1),
             cgi.escape(size, 1),
             multiple,
             result)
  return result
#######################################

###############################################################################

###############################################################################
### USE THE LIST-MAKING UTILITY FUNCTIONS TO CREATE A RADIO BUTTON
values = []
stmt = "SELECT color FROM cow_color ORDER BY color"
db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
cursor = db.cursor()
cursor.execute(stmt)

values = cursor.fetchall()
# for (color, ) in cursor:
#     values.append(color)

cursor.close()
db.close()


# Convert the list to any select form you desire
print(make_radio_group('color', values, values, '', True))
print(make_popup_menu('color', values, values, ''))
print(make_scrolling_list('color', values, values, '', 3, False))
###############################################################################


###############################################################################
### GENERATE A LIST USING FEASIBLE VALUES OF A ENUM FIELD
db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
size_info = get_enumorset_info(db, 'cookbook', 'cow_order', 'size')

print(make_radio_group('size',
    size_info['values'],
    size_info['values'],
    size_info['default'],
    True)) # display items vertically

print(make_popup_menu('size',
    size_info['values'],
    size_info['values'],
    size_info['default']))
###############################################################################


###############################################################################
### CREATE A CHECKBOX FROM DATA IN A MySQL TABLE

db = Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
acc_info = cookbook_utils.get_enumorset_info(db, 'cookbook', 'cow_order', 'accessories')

if acc_info['default'] is None:
    acc_def = ""
else:
    acc_def = acc_info['default'].split(',')

print(cookbook_webutils.make_checkbox_group('accessories',
    acc_info['values'],
    acc_info['values'],
    acc_def,
    True)) # display items vertically

print(cookbook_webutils.make_scrolling_list('accessories',
    acc_info['values'],
    acc_info['values'],
    acc_def,
    3, # display 3 items at a time
    True)) # create multiple-pick list
###############################################################################


###############################################################################
### GET VARIABLES THAT ARE PASSED ON TO A URL
params = cgi.FieldStorage()
param_names = params.keys()
param_names.sort()

print("<p>Parameter names: %s</p>" % param_names)

items = []
for name in param_names:
    val = ','.join(params.getlist(name))
    items.append("name=" + name + ", value=" + val)

print(cookbook_webutils.make_unordered_list(items))
###############################################################################





###############################################################################
###############################################################################
###############################################################################
# Print header, blank line, and initial part of page
print('''Content-Type: text/html

<html>

<head><title>Printing some data ...</title></head>

<body>

''')

# print your stuff here....

# Print page trailer
print('''


</body>

</html>
''')
###############################################################################
###############################################################################
###############################################################################


