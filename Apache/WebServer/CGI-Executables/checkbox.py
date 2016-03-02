#!/usr/local/bin/python

import os, sys
import cookbook

sys.path.append('./lib')
import cookbook_utils
import cookbook_webutils

db = cookbook.connect() # Mysql_tut.connect_to_db(db1Host, db1UName, db1Pwd, db1Name)
acc_info = cookbook_utils.get_enumorset_info(db, 'cookbook', 'cow_order', 'accessories')

if acc_info['default'] is None:
	acc_def = ""
else:
	acc_def = acc_info['default'].split(',')


# Print header, blank line, and initial part of page
print('''Content-Type: text/html

<html>

<head><title>HTML forms driven by Python</title></head>

<body>

<p>Trying different forms:</p>
''')


# put a checkbox form
print(cookbook_webutils.make_checkbox_group('accessories',
	acc_info['values'],
	acc_info['values'],
	acc_def,
	True)) # display items vertically


# put a multi select form
print(cookbook_webutils.make_scrolling_list('accessories',
	acc_info['values'],
	acc_info['values'],
	acc_def,
	3, # display 3 items at a time
	True)) # create multiple-pick list


print('''<p>

	</p>
	''')

# print a more involved upload form
print('''
		<form method="post" enctype="multipart/form-data" action="%s">
		Image name:<br />
		<input type="text" name="image_name", size="60" />
		<br />
		Image file:<br />
		<input type="file" name="upload_file", size="60" />
		<br /><br />
		<input type="submit" name="choice" value="Submit" />
		</form>
		''' % (os.environ['SCRIPT_NAME']))


import cgi
form = cgi.FieldStorage()
if form.has_key('upload_file') and form['upload_file'].filename != '':
	image_file = form['upload_file']
else:
	image_file = None


form = cgi.FieldStorage()
if form.has_key('upload_file') and form['upload_file'].filename:
	print("<p>A file was uploaded</p>")
else:
	print("<p>A file was not uploaded</p>")

print('''<p>
		''')
print(form)

print('''</p>''')

# Print page trailer
print('''


</body>

</html>
''')

