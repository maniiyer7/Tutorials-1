#!/usr/local/bin/python

import cookbook
import cgi
import os

title = 'State Name or Abbreviation Lookup'

title = "Search State Name"

# Print content type header and blank line that separates
# headers from page body

print("Content-Type: text/html")
print("")
print("<html>")
print("<head><title>%s</title></head>" % title)
print("<body>")


# If script is run from the command line, SCRIPT_NAME won't exist; fake
# it by using script name.
if not os.environ.has_key('SCRIPT_NAME'):
	  os.environ['SCRIPT_NAME'] = sys.argv[0]

print('''
	<form method="post" enctype="multipart/form-data" action="%s">
	State name:<br />
	<input type="text" name="state_name", size="60" />
	<br />
	Submit to Search:<br />
	<input type="submit" name="choice" value="Submit" />
	</form>
	''' % (os.environ['SCRIPT_NAME']))

form = cgi.FieldStorage()

if form.has_key('state_name') and form['state_name'].value != '':
	state_name = form['state_name'].value
else:
	state_name = None

print('<p>State abbrev.: %s</p>' %state_name)


conn = cookbook.connect()
conn.autocommit = True
cursor = conn.cursor()
stmt = """SELECT name FROM cookbook.states WHERE abbrev = '%s' """ % state_name
cursor.execute(stmt)
search_result = cursor.fetchone()
cursor.close()
conn.close()


print('<p>State full name: %s </p>' % search_result)

print("</body>")
print("</html>")



