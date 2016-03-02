###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: April 26, 2015
# summary: numpy playground
# Other resources:
# http://wiki.scipy.org/Tentative_NumPy_Tutorial
# http://www.sam.math.ethz.ch/~raoulb/teaching/PythonTutorial/intro_numpy.html
###############################################################################


import numpy as np
# from numpy import random
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
plt.ion()  # turn on interactive plotting
plt.style.use('ggplot')  # make matplotlib appearance similar to ggplot


###############################################################################
################################### NUMPY #####################################
###############################################################################
import numpy as np
###############################################################################
### DATA TYPES
# https://docs.scipy.org/doc/numpy-1.3.x/reference/arrays.dtypes.html
# numpy numerical types are instances of dtype objects
np.bool_
np.int_
np.intc
np.int16  # and similar
np.float64  # and similar
np.complex32  # and similar

# Examples:
x = np.float32(1.0)
y = np.int_([1,2,4])
z = np.arange(3, dtype=np.uint8)
# To convert the type of an array:
z.astype(float)
np.int8(z)
z.dtype

# Defining our own name for a specific data type
d = np.dtype(int)
np.issubdtype(d, int)

# Array scalars
# Numpy array scalars are scalars with an associated type.

###############################################################################
### ARRAY CREATION
# (option 1) lists and tuples (and any other array-like object) can be converted to np.array using array() function.
x = np.array([[1, 2.0],
              [0, 0],
              (1+1j, 3.)])

# (option 2) use pre-made functions such as zeros() and ones()
np.zeros((2,3))
np.ones((3,4))
np.arange(2, 10, 0.5, dtype=np.float)
np.linspace(1., 4., 6.)
np.indices((3, 3))

# (option 3) read from disk
# reading csv files: numpy, pylab, scipy.io packages all have csv reading functions

###############################################################################
### I/O with Numpy
from StringIO import StringIO
# The function genfromtxt() is used to import data into python.
# The delimiter keyword is used to define how to split each line into separate columns
data = "1, 2, 3\n4, 5, 6"
np.genfromtxt(StringIO(data), delimiter=",")

# If we have a fixed-width file, where columns have fixed width
data = " 1 2 3\n 4 5 67\n890123 4"
np.genfromtxt(StringIO(data), delimiter=3)

data = "123456789\n4 7 9\n4567 9"
np.genfromtxt(StringIO(data), delimiter=(4, 3, 2))

data = "1, abc , 2\n 3, xxx, 4"
np.genfromtxt(StringIO(data), dtype="|S5")
np.genfromtxt(StringIO(data), dtype="|S5", autostrip=True)

# Skipping file headers and footers
data = "\n".join(str(i) for i in range(10))
np.genfromtxt(StringIO(data),)
np.genfromtxt(StringIO(data), skip_header=3, skip_footer=5)

# Specifying columns
data = "1 2 3\n4 5 6"
np.genfromtxt(StringIO(data), usecols=(0, -1))

# Setting the names: to assign a name to each column, we can use an explicit structured dtype
data = StringIO("1 2 3\n 4 5 6")
np.genfromtxt(data, dtype=[(_, int) for _ in "abc"])
# or, just use the names keyword
np.genfromtxt(data, names="A, B, C")
# If we need to define the names from data itself, use names=True
data = StringIO("So it goes\n#a b c\n1 2 3\n 4 5 6")
np.genfromtxt(data, skip_header=1, names=True)

# Using converter function: we can define converter functions to convert elements of a column
convertfunc = lambda x: float(x.strip("%"))/100
data = "1, 2.3%, 45.\n6, 78.9%, 0"
names = ("i", "p", "n")
np.genfromtxt(StringIO(data), delimiter=",", names=names, converters={1: convertfunc})
# Converters can provide default for missing values
data = "1, , 3\n 4, 5, 6"
convert = lambda x: float(x.strip() or -999)
np.genfromtxt(StringIO(data), delimiter=",", converters={1: convert})


# Creating ndarray from text: ndfromtxt() function with usemask=False
# Creating masked array from text" mafromtxt() with usemask-True
# Creating recarray from text: recfromtxt(, usemask=False). Also recfromcsv(, usemask=False).
# Creating MaskedRecords from text: recfromtxt(, usemask=True). Also recfromcsv(, usemask=True).


###############################################################################
### ARRAYS
# Arrays are similar to lists in Python,
# except that every element of an array must be of the same type,
# typically a numeric type like float or int.
# np array is similar to R matrix

# array-creation routines:
# http://docs.scipy.org/doc/numpy/reference/routines.array-creation.html

# An array can be created from a list
a = np.array([1, 4, 5, 8], float)
type(a)
# The dtype property tells you what type of values are stored by the array:
a.dtype
type(a[1])
dt = np.dtype('>H')

### INDEXING FOR ARRAYS
# Array elements are accessed, sliced, and manipulated just like lists
a[:2]
a[3]
a[0] = 5.

# Arrays can be multidimensional. Unlike lists, different axes are accessed using commas inside bracket notation.
# Separate lists will each create one line of the resulting array.
a = np.array([[1, 2, 3], [4, 5, 6]], float)
a[0,0]
a[1,1]

# Negative index values start from the end
a[-1, -2]

# Use of a single ":" in a dimension indicates the use of everything along that dimension.
# Using : in arrays is the same as leaving the dimension blank in R.
a[-2,:]
a[-1:,-2:]  # last line, everything from 2 before last up to the last
a[-2,-1]


# The in statement can be used to test if values are present in an array.
2 in a
0 in a


### SHAPE
# The shape property of an array returns a tuple with the size of each array dimension.
a.shape
# When used with an array, the len function returns the length of the first axis (number of rows)
len(a)

# Arrays can be reshaped using tuples that specify new dimensions.
# About array shape: http://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r
a = np.array(range(30), float)
a = a.reshape((5, 6))
# One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
# I.e., if you use -1 for one of the shape dimensions, numpy will automatically calculate the correct value for you so that the resulting array dimensions are compatible with the original one.
b = a.reshape(3, -1)
c = a.reshape(-1, 3)
d = a.reshape(1, -1)  # Create a one-dim array from a matrix.


# Array reshaping and slicing example
y = np.arange(35).reshape(5, 7)
y[1:5:2, ::3]

### INDEX ARRAYS
# It is possible to index arrays with other arrays (or lists)
x = np.arange(10, 1, -1)
x[np.array([3, 3, 1, 8])]
# The result of index array operation is an array with the same shape and type of the index array (and not the original array)
x[np.array([[1,1],[2,3]])]
# Using index arrays on multidimensional arrays
y = np.arange(35).reshape(5, 7)
y[np.array([0, 2, 4]), np.array([0, 1, 2])]
y[np.array([0, 2, 4]), 1]
y[np.array([0, 2, 4])]
# Index arrays using boolean arrays
b = y > 20
y[b]
# If we index a two-dimensional array using a one-dimensional index array, the rows of the multi-dimensional array will be chosen.
y[b[:, 5]]
y[np.array([0,2,4]), 1:3]
# If the dimensions of the boolean index array is smaller than the dimensions of target array,
# the result will be indexed by the order of the target array, and any remaining dimension will be expanded.
x = np.arange(30).reshape(2,3,5)
b = np.array([[True, True, False], [False, True, True]])
# Here, the result will include all third-dimension elements because the index array only has two dimensions
x[b]


### OTHER ARRAY OPERATIONS
# Arrays can be converted to lists
a = np.array([1,2,3], float)
a.tolist()
list(a)

# Multi dimensional arrays will be converted to lists of lists.
a = np.array([[1, 2, 3], [4, 5, 6]], float)
a.tolist()
list(a)

# Arrays can be filled with one value.
# Note that this works on the array itself and the result cannot be passed on to another variable.
a.fill(0)

# Arrays can be transposed
# The result can be assigned to another variable and does not impact the original array.
a = np.array([[1, 2, 3], [4, 5, 6]], float)
b = a.transpose()

# Arrays can be flattened to a one-dimensional array.
# The result can be assigned to another variable and does not impact the original array.
a = np.array([[1, 2, 3], [4, 5, 6]], float)
b = a.flatten()

# Two or more arrays can be concatenated together using the concatenate function with a tuple of the arrays to be joined:
a = np.array([1,2], float)
b = np.array([3,4,5,6], float)
np.concatenate((a, b))
# If an array has more than one dimension, it is possible to specify the axis along which multiple arrays are concatenated
a = np.array([[1, 2], [3, 4]], float)
b = np.array([[5, 6], [7,8]], float)
np.concatenate((a,b))
np.concatenate((a,b), axis=0)
np.concatenate((a,b), axis=1)
###############################################################################

###############################################################################
### STRUCTURAL INDEXING
y = np.arange(35).reshape(5, 7)
y[:, np.newaxis, :].shape

x = np.arange(5)
x.shape
x[:, np.newaxis].shape
x[:, np.newaxis] + x[np.newaxis, :]

z = np.arange(81).reshape(3, 3, 3, 3)
z[1, ..., 2]
# this is equivalen to
z[1, :, :, 2]


# The dimensionality of an array can be increased using the newaxis constant in bracket notation
# np.newaxis creates one element from each single element of the given dimension
# and puts the new elements in a new dimension.
# The new array has two dimensions; the one created by newaxis has a length of one
a = np.array([1, 2, 3], float)
a[:,]
a[:,np.newaxis]
a[:,np.newaxis].shape

a[np.newaxis,:]
a[np.newaxis,:].shape

# np.newaxis always adds one dimension to the array.
# The new dimension is added along
c = a[:,np.newaxis]
c[np.newaxis,:]
c[:,np.newaxis].shape

### BROADCASTING
x = np.arange(4)
xx = x.reshape(4, 1)
y = np.ones(5)
z = np.ones((3, 4))
x.shape
y.shape
x + y
xx.shape
y.shape
(xx + y).shape
x + z

# Boradcasting is related to outer product of two arrays
a = np.array([0., 10., 20., 30.])
b = np.array([1., 2., 3.])
a[:, np.newaxis] + b
###############################################################################

###############################################################################
### STRUCTURED ARRAYS
x = np.zeros((2,), dtype=('i4, f4, a10'))
x[:] = [(1, 2., 'Hello'), (2, 3., 'World')]
x
# Here we have created a one-dimensional array of length 2.
# Each element of this array is a record that contains three items,
# a 32-bit integer, a 32-bit float, and a string of length 10 or less.
# We can access array elements using names of the fields within each record.
y = x['f1']
y[:] = 2 * y
# Note that y is a view, rather than an independent copy of x elements. Therefore, any operation in y translates to x.
x

# numpy arrays are different from conventional matrices. numpy arrays are better understood as an array of records, with each records having its own data structure.
# Therefore, within one record, one can have different data types and even separate matrices within the record
# As a result, numpy arrays cannot be referenced using [row, column] convention.
x[0]  # the first record
x[0][1]  # the second item in first record
x[0, 1]  # this will throw an error

# We can specify record structure (i.e., dtype) using an argument.
# The argument can be a string, tuple, list, or dictionary.
# The argument is supplied to a dtype function keyword or a dtype object constructor.
# The argument can be set in one of four alternative ways:
#   (a) string argument: a comma-separated list of type specifiers, optionally with extra shape information. Examples: b1, i1, u4, f4, c16, a<n>, int8, uint8, float16, complex16
#   (b) tuple: when a structure is mapped to an existing data type. This is done by pairing in a tuple, the existing data type with a matching dtype definition.
#   (c) list: each tuple has 2 or 3 elements specifying (a) the name of the field and (2) the type of the field and (3) the shape.
#   (d) dictionary: a dictionary with two required fields (names and formats). The names must be strings. The format list contains type/shape specifier. Two optional keys of 'offsets' and 'titles' can be used.
#   (d*) another dictionary form is a dictionary of name keys with tuple values specifying type, offset, and an optional title.


# The result set as a whole is a list, because rows are functionally equivalent (homogeneous);
# the individual rows are tuples, because rows are coherent, record-like groupings of (heterogeneous) column data.



## example of defining the structure using tuple
# The shape of dtype determines the shape of each record.
x = np.zeros(3, dtype='3int8, float32, (2,3)float64')
x.shape

## If the names of dtypes is not important, use a single string to specify dtype.
x = np.zeros(3, dtype='int8, float32, float64')
x.shape

## If you want to specify the name of each dtype, pass them as a list of tuples.
x = np.zeros(3, dtype=[('one', 'int8'), ('two', 'float32'), ('three', 'float64')])
x.shape
x.dtype.names

x = np.zeros(3, dtype=('i4', [('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')]))
x.dtype  # it did not capture 'i4'
x.shape
x.dtype.names
x['r']
x['g']
x['b']
x['a']

## example of defining the structure using lists (need to pass the names)
x = np.zeros(3, dtype=[('x', 'f4'), ('y', np.float32), ('value', 'f4', (2, 2))])

## example of defining the structure using dictionary
x = np.zeros(3, dtype={'col1':('i1', 0), 'col2':('f4', 1)})


### FIELD NAMES
# field names are an attribute of the dtype object defining the record structure
x.dtype.names
x.dtype.names = ('x', 'y')

### FIELD TITLES
x = np.zeros(3, dtype={'col1':('i1', 0, 'title 1'), 'col2':('f4', 1, 'title 2')})
# Field titles can be used to put associated info for fields. They do not have to be strings.
x.dtype.fields['title 1'][2]

# To access multiple fields, use a list of field names
x = np.array([(1.5, 2.5, (1.0,2.0)), (3., 4., (4.,5.)), (1., 3., (2.,6.))],
             dtype=[('x', 'f4'), ('y', np.float32), ('value', 'f4', (2,2))])
x.shape
x[['x', 'y']]
x[['x', 'value']]

arr = np.zeros((5,), dtype=[('var1','f8'),('var2','f8')])
arr['var1'] = np.arange(5)
arr[0] = (10,20)
###############################################################################

###############################################################################
### MISSING VALUES
# NaN can be used
myarr = np.array([1., 0., np.nan, 3.])
# Similar to R, you cannot use equality to test NaN
np.where(myarr == np.nan)
myarr[np.isnan(myarr)]
np.isinf(myarr)  # True if value is inf
np.isfinite(myarr)  # True if not nan or inf
np.nan_to_num(myarr)  # Map nan to 0, inf to max float, -inf to min float
# To ignore NaN from common summary functions
np.nansum(myarr)
np.nanmax(myarr)
np.nanmin(myarr)
np.nanargmax(myarr)
np.nanargmin(myarr)


### missing_values() function
# a string or a comma-separated string: the marker for missing data for all the columns
# a sequence of strings: each item is associated with a column
# a dictionary: keys are column indices or names. The special key "None" can be used to define a default applicable to all columns

### filling_values() function
# a single value: the default for all columns
# a sequence of values: the default for the corresponding column
# a dictionary: default values specific to each column by using column index or name as the key. use None to refer to the entire array.
data = "N/A, 2, 3\n4, ,???"
kwargs = dict(delimiter=",",
              dtype=int,
              names="a,b,c",
              missing_values={0:"N/A", 'b':" ", 2:"???"},
              filling_values={0:0,  'b':0, 2:-999})
np.genfromtxt(StringIO(data), **kwargs)
###############################################################################


###############################################################################
# 'ignore' : Take no action when the exception occurs.
# 'warn' : Print a RuntimeWarning (via the Python warnings module).
# 'raise' : Raise a FloatingPointError.
# 'call' : Call a function specified using the seterrcall function.
# 'print' : Print a warning directly to stdout.
# 'log' : Record error in a Log object specified by seterrcall.
oldsettings = np.seterr(all='warn')
np.zeros(5,dtype=np.float32)/0.
j = np.seterr(under='ignore')
np.array([1.e-100])**10
j = np.seterr(invalid='raise')
np.sqrt(np.array([-1.]))

def errorhandler(errstr, errflag):
    print "saw stupid error!"

np.seterrcall(errorhandler)
j = np.seterr(all='call')
np.zeros(5, dtype=np.int32)/0
j = np.seterr(**oldsettings) # restore previous error-handling settings
###############################################################################


###############################################################################
# Note that transpose does not change the dimensionality of single dimension arays.
b = a.transpose()

# The arange function is similar to the range function but returns an array.
np.arange(5, dtype=float)
np.arange(1, 6, 2, dtype=int)
np.arange(1, 12, 3, dtype=int)

# The functions zeros and ones create new arrays of specified dimensions filled with these values.
np.ones((2,3), dtype=float)
np.zeros(7, dtype=int)

# The zeros_like and ones_like functions create a new array with the same dimensions and type of an existing one.
a = np.array([[1, 2, 3], [4, 5, 6]], float)
np.zeros_like(a)
np.ones_like(a)

# Matrices are 2D arrays. Matrices also have special functions for fast specification.
np.identity(4, dtype=float)
# The eye function returns matrices with ones along the kth diagonal.
np.eye(4, k=1, dtype=float)

# When standard mathematical operations are used with arrays,
# they are applied on an element-by-element basis.
# This is the same for 2D arrays (matrices).
a = np.array([1,2,3], float)
b = np.array([5,2,6], float)
a + b
a - b
a * b
b / a
a % b
b ** a

# Size is different from dimension.
a = np.array([[1, 2], [3, 4], [5, 6]], float)
# Size is the total number of elements in the array.
a.size
# Dimension is the shape of the array.
a.shape

# If the size of individual dimensions of the two arras are not the same, error will be thrown.
a = np.array([1,2,3], float)
b = np.array([4,5], float)
a + b
# If the arrays are different in dimension, but the size of each dimension is the same, the smaller array will be broadcasted
a = np.array([[1, 2], [3, 4], [5, 6]], float)
b = np.array([-1, 3], float)
a + b

# We can use np.newaxis to specify how to broadcast.
a = np.zeros((2,2), float)
b = np.array([-1., 3.], float)
a + b[np.newaxis,:]
a + b[:,np.newaxis]

# It is possible to loop over the arrays
a = np.array([1, 4, 5], int)
for x in a:
    print x
# For multi dimensional arrays, the iteration returns one full element at a time.
a = np.array([[1, 2], [3, 4], [5, 6]], float)
for x in a:
    print x
# Multiple assignment
a = np.array([[1, 2], [3, 4], [5, 6]], float)
for (x, y) in a:
    print x * y

# Certain statistics of the array elements can be calculated.
a = np.array([2, 4, 3], float)
a.sum()
a.prod()
a.mean()
a.var()
a.std()
a.min()
a.max()
a.argmin()
a.argmax()
# For multidimensional arrays, the summary functions can take an optional argument axis
# that will perform an operation along only the specified axis,
# placing the results in a return array.
a = np.array([[0, 2], [3, -1], [3, 8]], float)
a.mean(axis=0)
a.mean(axis=1)

# Arrays can be sorted
a = np.array([6, 2, 5, -1, 0], float)
b = sorted(a)  # This returns a list, containing sorted elements of a
b
a.sort()  # This operates on the array itself, and changes the array. Does not return anything.
a

# Clipping values: Values in an array can be "clipped" to be within a prespecified range.
# This is the same as applying min(max(x, minval), maxval) to each element x in an array.
b = a.clip(0, 5)
b

# To find unique elements in an array
a = np.array([1, 1, 4, 5, 5, 5, 7], float)
np.unique(a)

# The any and all operators can be used to determine whether or not any or all elements of a Boolean array are true.
c = np.array([ True, False, False], bool)
any(c)
all(c)

# Compound Boolean expressions can be applied to arrays on an element-by-element basis
# using special functions logical_and, logical_or, and logical_not.
a = np.array([1, 3, 0], float)
np.logical_and(a > 0, a < 3)
b = np.array([True, False, True], bool)
np.logical_not(b)
c = np.array([False, True, False], bool)
np.logical_or(b, c)

# The where function forms a new array from two arrays of equivalent size
# using a Boolean filter to choose between elements of the two.
# Where() is similar to ifelse() functin in R.
a = np.array([1, 3, 0], float)
np.where(a != 0, 1/a, a)
np.where(a > 0, 3, 2)

# The nonzero function gives a tuple of indices of the nonzero values in an array.
a = np.array([[0, 1], [3, 0]], float)
a.nonzero()

# To assess whether elements are NaN
a = np.array([1, np.NaN, np.Inf], float)
np.isnan(a)
np.isfinite(a)


### ARRAY ITEM SELECTION
# (a) using item index
# (b) using boolean variables
# (c) selecting full dimensions using colon :
# (d) using integer arrays or lists which can also contain repeat items
# (e) use special function take() which works exactly the same as bracket.
# arrays also permit selection using other arrays. This is similar to R behavior.
a = np.array([[6, 4], [5, 9]], float)
a >= 6
a[a >= 6]
# Or we could save the selector array in a variable.
sel = (a >= 6)
a[sel]
a[np.logical_and(a > 5, a < 9)]


# For multi-dimensional arrays, multiple selection arrays are needed, one for each dimension.
# The selection arrays will be paired on an element-by-element basis.
a = np.array([[1, 4], [9, 16]], float)
b = np.array([0, 0, 1, 1, 0], int)
c = np.array([0, 1, 1, 1, 1], int)
a[b,c]


### take() can be used to select array elements in specific dimensions.
a = np.array([[0, 1], [2, 3]], float)
b = np.array([0, 0, 1], int)
a.take(b, axis=0)
a.take(b, axis=1)


### choose() selects elements from an array based on a set of indexes
choices = [[0, 1, 2, 3], [10, 11, 12, 13],
           [20, 21, 22, 23], [30, 31, 32, 33]]
np.choose([2, 3, 1, 0], choices)
# the first element of the result will be the first element of the
# third (2+1) "array" in choices, namely, 20; the second element
# will be the second element of the fourth (3+1) choice array, i.e.,
# 31, etc.
# because there are 4 choice arrays, an index of greater than 3 has to either be clipped or wrapped around back to the beginning of the array.
np.choose([2, 4, 1, 0], choices, mode='clip') # 4 goes to 3 (4-1)
np.choose([2, 4, 1, 0], choices, mode='wrap') # 4 goes to (4 mod 4); i.e., 0
# The final outcome will have the shape of the index matrix.

a = np.array([0, 1]).reshape((2,1,1))
c1 = np.array([1, 2, 3]).reshape((1,3,1))
c2 = np.array([-1, -2, -3, -4, -5]).reshape((1,1,5))
np.choose(a, (c1, c2)) # result is 2x3x5, res[0,:,:]=c1, res[1,:,:]=c2

### put() functions takes values from a source array and places them at specific indices  in the array.
a = np.array([0, 1, 2, 3, 4, 5], float)
b = np.array([9, 8, 7], float)
a.put([0, 3], b)
a
a.put([0, 3], 5)
a


### VECTOR AND MATRIX MATH
# dot product
a = np.array([1, 2, 3], float)
b = np.array([0, 1, 1], float)
np.dot(a, b)

a = np.array([[0, 1], [2, 3]], float)
b = np.array([2, 3], float)
c = np.array([[1, 1], [4, 0]], float)
np.dot(b, a)

# Inner, Outer, Cross product of matrices
a = np.array([1, 4, 0], float)
b = np.array([2, 2, 1], float)

# Outer product: the first vector becomes a column matrix and the second vector becomes a row matrix.
# This way, the middle dimension (right dim of the left matrix and left dim of the right matrix) is always 1.
# As a result, the sizes of the vectors do not have to be equal.
np.outer(a, b)
np.outer(b ,a)

# Inner product is the sum-product of vector items.
np.inner(a, b)

# Cross product: see http://mathworld.wolfram.com/CrossProduct.html
np.cross(a, b)



# Matrix is a 2-dimensional ndarray that preserves its two-dimensional nature throughout operations. It has certain special operations, such as * (matrix multiplication) and ** (matrix power), defined:
x = np.mat([[1, 2], [3, 4]])
x
x**2

### MORE LINEAR ALGENBRA FUNCTIONS
# Matrix determinant
a = np.array([[4, 2, 0], [9, 3, 7], [1, 2, 1]], float)
np.linalg.det(a)

# eigenvalues and eigenvectors of a matrix
vals, vecs = np.linalg.eig(a)
vals
vecs

# Inverse matrix
b = np.linalg.inv(a)
b

# Singular value decomposition
a = np.array([[1, 3, 4], [5, 2, 3]], float)
U, s, Vh = np.linalg.svd(a)
U
s
Vh


### POLYNOMIAL MATH
# Given a set of roots, it is possible to show the polynomial coefficients:
np.poly([-1, 1, 1, 10])
# Given a set of coefficients, the root function returns all of the polynomial roots:
np.roots([1, -11, 9, 11, -10])

# Integrating polynomials
np.polyint([1,1,1,1])
# Taking derivatives
np.polyder([1./4., 1./3., 1./2., 1., 0.])

# Addition, subtraction, multplication, division of polynomial coefficients
# functions polyadd, polysub, polymul, and polydiv

# Evaluating a polynomial at a point
np.polyval([1, -2, 0, 2], 4)

# Fit a polynomial of specified order to a set of data using a least-squares approach
# More sophisticated interpolation routines can be found in the SciPy package.
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [0, 2, 1, 3, 7, 10, 11, 19]
np.polyfit(x, y, 2)
np.polyfit(x, y, 3)


#######################################
### SUMMARY STATISTICS
### Median
a = np.array([1, 4, 3, 8, 9, 2, 3], float)
np.median(a)

### Covariance
# covariance is not normalized. The values on the diagonal are variance of individual variables.
np.cov(a)


### Correlation coefficient
# corr coeff is normalized covarianve matrix. Values range between -1 and 1. Diagonal values are always 1.
a = np.array([[1, 2, 1, 3], [5, 3, 1, 8]], float)
c = np.corrcoef(a)
c


#######################################
### RANDOM NUMBERS
# Set the seed
np.random.seed(293423)
# An array of random numbers in the half-open interval [0.0, 1.0)
np.random.rand(5)
# two-dimensional random arrays
np.random.rand(2, 3)
np.random.rand(6).reshape((2,3))
# a single random number in [0.0, 1.0)
np.random.random()
# random integers in the range [min, max)
np.random.randint(5, 10)
# Functions for drawing random numbers from other distributions are also available.
np.random.poisson(6.)
np.random.normal(1.5, 4.)
# draw from a standard normal distribution
np.random.normal()
# draw multiple values
np.random.normal(size=5)
# randomly shuffle the order of items in a list (it alters the original list in place)
l = range(10)
np.random.shuffle(l)


### MORE FUNCTIONS
# include Fourier transforms, more complex linear algebra operations,
# size / shape / type testing of arrays, splitting and joining arrays,
# histograms, creating arrays of numbers spaced in various ways,
# creating and evaluating functions on grid arrays,
# treating arrays with special (NaN, Inf) values, set operations,
# creating various kinds of special matrices,
# evaluating special mathematical functions (e.g., Bessel functions).
###############################################################################

###############################################################################
### numpy DATA TYPES
# a custom record type where it has two fields associated for each variable
# A 16-bit string with a field named name
# A 2-element tuple of floating point numbers that are 64-bits each, with a field named grades
dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
type(dt)
# Now we can use the data type as a template for new variables
x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
type(x)
# the list itself is a numpy.array, but not the individual elements.
type(x[1])  # a void is a generic data type that can hold elements of diff types
type(x[1]['name'])  # this gives us the type of the individual values
# Summary: each element in this list is of type numpy.void.
# However, the individual fields for each element in our list is either a tuple of numbers, or a string
###############################################################################


###############################################################################
### FURTHER RESOURCES
# http://docs.scipy.org/doc/numpy/glossary.html

###############################################################################
################### IMPORTANT ND ARRAY SHAPE MANIPULATIONS ####################
###############################################################################

###############################################################################
### meshgrid
# Creates a vector-based mesh grid.
# x values will be repeated along the first axis, while y values will be repeated along the second axis.
# As a result, multiplying the resulting vectors gives all possible combinations of the two original vectors; ie, a mesh.

import numpy as np
nx, ny = (3, 4)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)
xv
yv


###############################################################################
### METHODS TO COMBINE, SPLIT, RESHAPE, AND MANIPULATE ARRAYS

### c_[]
# NOTE that c_ is an object, not a function (see the [] instead of paranthesese)
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html
# Translates slice objects to concatenation along the second axis.
# The number of rows will stay the same, while column values will be concatenated.
np.c_[np.array([[1,2,3], [20,21,22]]), np.array([[4,5,6], [10,11,12]])]


### r_[]
# Appends rows after each other
np.r_[np.array([[1,2,3], [20,21,22]]), np.array([[4,5,6], [10,11,12]])]


### concatenate()
# Appends the arrays one after the other.
np.concatenate((np.array([[1,2,3], [20,21,22]]), np.array([[4,5,6], [10,11,12]])))


## reshape()
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
# reshape() works the exact opposite of c_() and ravel().
# It puts the elements in the desired shape. The order is such that the
# last element changes the fastest. For example, in a 2D array, it
# populates by row.
np.arange(6).reshape((3, 2))


### ravel()
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html
# flattens the array. By default, it flattens the array by changing the last index fastest and so on.
# For example, for a 2D array, ravel goes row-by-row.
np.array([[1, 2, 3], [4, 5, 6]]).ravel()


### hstack() & vstack()
# Stack arrays in sequence horizontally (column wise).
np.hstack(([1,2,3], [20,21,22]))

# It works similarly to np.c_[]
np.hstack((np.array([[1,2,3], [20,21,22]]), np.array([[4,5,6], [10,11,12]])))

a = np.array([[1], [2], [3]])
b = np.array([[2], [3], [4]])
np.vstack((a,b))


### split()
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.split.html#numpy.split
# The split() function can accept an integer or an array to specify where the cuts would happen.
# If indices_or_sections is an integer, N, the array will be divided into N equal arrays along axis. If such a split is not possible, an error is raised.
# If indices_or_sections is a 1-D array of sorted integers, the entries indicate where along axis the array is split. For example, [2, 3] would, for axis=0, result in arr[:2], arr[2:3], arr[3:]
# If an index exceeds the dimension of the array along axis, an empty sub-array is returned correspondingly.

# split() only accepts integers if the array dimension is divisible by that integer.
# array_split() allows indices_or_sections to be an integer that does not equally divide the axis.
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html#numpy.array_split

### vsplit()
# Split an array into multiple sub-arrays vertically (row-wise).
x = np.arange(16.0).reshape(4, 4)

np.vsplit(x, 2)

np.vsplit(x, np.array([3, 6]))
np.vsplit(x, np.array([1,3]))

# vsplit() is the same as split(axis=0)


### dstack()
# Stack arrays in sequence depth wise (along third axis).
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html#numpy.dstack
# This is a simple way to stack 2D arrays (images) into a single 3D array for processing.
# Similar to what zip() does with tuples.
a = np.array((1,2,3))
b = np.array((2,3,4))
np.dstack((a,b))

a = np.array([[1],[2],[3]])
b = np.array([[2],[3],[4]])
np.dstack((a,b))

###############################################################################
