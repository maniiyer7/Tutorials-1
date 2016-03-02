#!/usr/local/bin/python

import os
import cookbook
import cgi
import cookbook_webutils

params = cgi.FieldStorage()
param_names = params.keys()

title = 'Paged US State List'

# Print content type header and blank line that separates
# headers from page body

print("Content-Type: text/html")
print("")
print("<html>")
print("<head><title>%s</title></head>" % title)
print("<body>")


# Handle parameters
print("<p>Parameter names: %s</p>" % param_names)

if 'start' in param_names:
	start = int(params.getlist('start')[0])
else:
	start = 1

if 'count' in param_names:
	count = int(params.getlist('count')[0])
else:
	count = 5

end = start + count - 1

print('<p>Start: %s ; End: %s</p>' % (start,end))


# Read from database
stmt = """
SELECT name, abbrev, statehood, pop
FROM states
ORDER BY name LIMIT %d,%d
"""%(start-1, count+1)

print('<p>%s</p>' % stmt)

conn = cookbook.connect()
conn.autocommit = True
cursor = conn.cursor()
cursor.execute(stmt)
results = cursor.fetchall()
cursor.close()
conn.close()


# Links to previous and next pages
if start > 0:
	print('''<a href="http://localhost/cgi-bin/state_pager1.py?start=%s&count=%s">Previous Page</a>''' %(abs(start-count), count))
if len(results) > count:
	print('''<a href="http://localhost/cgi-bin/state_pager1.py?start=%s&count=%s">Next Page</a>''' %(start+count, count)) 


# Print results
print('''<p>Results:
	''')

print(cookbook_webutils.make_unordered_list(results[:-1]))

print('''
	</p>''')


print("</body>")
print("</html>")




