__author__ = 'amirkavousian'



###############################################################################
### CAPITAL-ONE Q1
import sys, math

m,n = sys.stdin.readline().split()
m,n = [int(x) for x in (m,n)]

try:
    # Using scipy if the package is installed (recommended)
    from scipy import misc
    print int(misc.comb(m,n))
except:
    # Using Python basic facilities (can get inefficient with large m,n values)
    print math.factorial(m) / (math.factorial(n)*math.factorial(m-n))
###############################################################################


###############################################################################
### CAPITAL-ONE Q2
import sys, re
inTxt = sys.stdin.readline().split()  # "I may opt for a top yam for Amy, May, and Tommy"

regex = re.compile('[^a-zA-Z]')

# Format individual words
def FormatText(txt):
    txt = regex.sub('', txt) # for alphanumeric: re.sub(r'\W+', '', txt)
    txt = txt.lower()
    txt = ''.join(sorted(txt))
    return txt

# Format the entire list and print
inTxt_alphanum = [FormatText(txt) for txt in inTxt]
print ' '.join(sorted(list(set(inTxt_alphanum))))
###############################################################################


###############################################################################
### CAPITAL-ONE Q3
import sys
m,k = sys.stdin.readline().split()
m,k = [int(x) for x in (m,k)]

result = m
for i in range(k):
    if (result % 2 == 0):
        result /= 2
    else:
        result = 3*result + 1

    print result

print result
###############################################################################


###############################################################################
### CAPITAL-ONE Q4
import sys
coefs = sys.stdin.readline().split()
coefs = [float(x) for x in coefs]

for i in range(1):
    # read model coefficients
    coefs = sys.stdin.readline().split()
    coefs = [float(x) for x in coefs]
    # read data points
    points = sys.stdin.readline().split()
    points = [float(x) for x in points]

rh = coefs[-1] + sum([b*x for b,x in zip(coefs[:-1], points)])
logEst = 1 / (1+math.exp(-rh))
print '%.3f' % logEst
###############################################################################

###############################################################################
###############################################################################
###############################################################################
### AGGREGATING ARRAYS USING itertools.groupby

# https://docs.python.org/2/library/itertools.html#itertools.groupby
# You can use itertools.groupby function to perform operations on groups of elements.
# groupby() returns an iterator; so you need to use it in a loop or similar structure if you want to use its results.

import itertools

# groupby() starts a new key whenever the value changes from previous value. It does not return a unique set of values. In that sense, it is different from groupby() function in pandas, which itself is modeled after GROUP BY command in SQL.
# In short, groupby() is an iterator for blocks of similar value in a list object. It is not a summary function, although it can be used as a summary function.
[k for k, g in itertools.groupby('AAAABBBCCDAABBB')]

# We can make it a summary function by sorting the array before passing it to the groupby() function.
[k for k, g in itertools.groupby(''.join(sorted('AAAABBBCCDAABBB')))]


### groupby() returns an iterator; so to use its values you need to convert it to an object (eg, a list)
arr = [1,2,3,3,3,4,4,5]
[{'key':k, 'value':list(g)} for k, g in itertools.groupby(arr)]

[{'key':k, 'key_count':len(list(g))} for k, g in itertools.groupby(arr)]


### You can define a function to operate on the array elements before they are evaluated for groupby() operation.
# This function is called 'key function'
[{'key':k, 'key_count':len(list(g))} for k, g in itertools.groupby(arr, lambda x: x%2)]


#######################################
### USE operator.itemgetter FUNCTION TO USE A SPECIFIC FIELD FOR GROUPING
# If the array is an ndarray with different fields, to use one specific field for grouping the values,
# use the function operator.itemgetter(fieldName) as key function.

import operator
import numpy as np

arr = np.array([(1,2,3),(1,3,4),(2,4,5)],
                 dtype=[('field1','O'), ('field2','O'), ('field3','O')])

fieldName = 'field1'
r = itertools.groupby(arr, key=operator.itemgetter(fieldName))  # key is the function that will be applied to each element of array before being sent to groupby

[{fieldName:e[0], 'data':np.array(map(tuple,list(e[1])), dtype=arr.dtype.descr)} for e in r]

###############################################################################


