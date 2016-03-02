#!/usr/local/bin/python

import os
import cookbook
import cgi
import cookbook_webutils

params = cgi.FieldStorage()
param_names = params.keys()

title = 'Driver Log'

# Print content type header and blank line that separates
# headers from page body

print("Content-Type: text/html")
print("")
print("<html>")
print("<head><title>%s</title></head>" % title)
print("<body>")


# Handle parameters

if 'sort' in param_names:
	sort = str(params.getlist('sort')[0])
else:
	stmt = '''SELECT column_name FROM information_schema.columns 
	WHERE table_schema = '%s' AND table_name = '%s' AND ordinal_position = 1
	''' %('cookbook', 'driver_log')
	print('''<p>%s</p>''' %stmt)
	conn = cookbook.connect()
	cursor = conn.cursor()
	cursor.execute(stmt)
	sort = str(cursor.fetchone()[0])
	cursor.close()
	conn.close()

print('''<p>Sort by: %s</p>''' %sort)


stmt = '''SELECT * FROM %s.%s 
	  ORDER BY %s LIMIT 50
	''' %('cookbook', 'driver_log', sort)
conn = cookbook.connect()
cursor = conn.cursor()
cursor.execute(stmt)
results = cursor.fetchall()
cursor.close()
conn.close()

print('''<p>Results:
	''')
print(cookbook_webutils.make_unordered_list(results))
print('''
        </p>''')

stmt = '''SELECT column_name FROM information_schema.columns 
	  WHERE table_schema = '%s' AND table_name = '%s'
	  ''' %('cookbook', 'driver_log')
print('''<p>%s</p>''' %stmt)
conn = cookbook.connect()
cursor = conn.cursor()
cursor.execute(stmt)
colNames = cursor.fetchall()
cursor.close()
conn.close()


# Print header
print('<table border="1">')
print('<tr>')
for i in range(len(colNames)):
	colName = colNames[i][0]
	print('''<th><a href="%s?sort=%s">%s</a></th>
		''' %('http://localhost/cgi-bin/driver_log.py', colName, colName))
print('''</tr>''')


# Print rows
for i in range(len(results)):
	row = results[i]
	print('''<tr>''')
	for j in range(len(row)):
		print('''<td>%s</td>''' %str(row[j]))
	print('''</tr>''')

print('''</table>''')


print("""<p>Page View Count: %s</p>""" %(str(cookbook_webutils.get_hit_count('driver_log.py'))))


print("</body>")
print("</html>")

