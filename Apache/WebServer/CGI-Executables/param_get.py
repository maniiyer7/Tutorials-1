#!/usr/local/bin/python
import cgi
import cookbook_webutils

params = cgi.FieldStorage()
param_names = params.keys()
param_names.sort()


print('''Content-Type: text/html

		<html>

		<head><title>Printing some data ...</title></head>

		<body>

		''')


print("<p>Parameter names: %s" % param_names)

items = []
for name in param_names:
	val = ','.join(params.getlist(name))
	items.append("name=" + name + ", value=" + val)

print(cookbook_webutils.make_unordered_list(items))

print('''
		</p>''')

print('''


		</body>

		</html>
		''')

