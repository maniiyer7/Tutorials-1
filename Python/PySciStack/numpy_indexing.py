
"""
Source:
http://docs.scipy.org/doc/numpy-1.10.1/user/basics.indexing.html
"""

import numpy as np



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

###############################################################################

# http://docs.scipy.org/doc/numpy-1.10.1/user/basics.indexing.html

x = np.arange(10)
x.shape = (2,5) # now x is 2-dimensional
x[1,3]

# if one indexes a multidimensional array with fewer indices than dimensions, one gets a subdimensional array
x[0]

# The returned array is not a copy of the original, but points to the same values in memory as does the original array.
# note that x[0,2] = x[0][2] though the second case is more inefficient as a new temporary array is created after the first index that is subsequently indexed by 2.
x[0][2]


#######################################
### Slicing and striding arrays
# It is possible to slice and stride arrays to extract arrays of the same number of dimensions,
# but of different sizes than the original.
# Slices of arrays do not copy the internal array data but also produce new views of the original data.

x = np.arange(10)
x[2:5]

x[:-7]

x[1:7:2]

y = np.arange(35).reshape(5,7)
y[1:5:2,::3]


#######################################
### Indexing arrays with other arrays
# There are two different ways of accomplishing this.
# (a) Use one or more arrays of index values.
# (b) Provide a boolean array of the proper shape to indicate the values to be selected.
# Index arrays are a very powerful tool that allow one to avoid looping over individual elements in arrays and
# thus greatly improve performance.

# For all cases of index arrays, what is returned is a copy of the original data, not a view as one gets for slices.
# Slices ==> view of the original data
# Index arrays ==> copy of the original data
# Boolean index arrays ==> copy of the original data

x = np.arange(10,1,-1)
x[np.array([3, 3, 1, 8])]
x[np.array([3, 3, 1, 8])] = -5
x

# The output has the same shape as the index array,
# with the type and values of the array being indexed.
# Therefore, the shape of the index array does not have to be compatible with the original array.
x[np.array([[1,1],[2,3]])]


#######################################
### Indexing multidimensional arrays
# if the index arrays have a matching shape,
# and there is an index array for each dimension of the array being indexed,
# the resultant array has the same shape as the index arrays,
# and the values correspond to the index set for each position in the index arrays.
y[np.array([0,2,4]), np.array([0,1,2])]


# If the index arrays do not have the same shape, there is an attempt to broadcast them to the same shape.
# If they cannot be broadcast to the same shape, an exception is raised:
y[np.array([0,2,4]), np.array([0,1])]


# The broadcasting mechanism permits index arrays to be combined with scalars for other indices.
# The effect is that the scalar value is used for all the corresponding values of the index arrays
y[np.array([0,2,4]), 1]


# It is possible to only partially index an array with index arrays
y[np.array([0,2,4])]

# When the index array has less dimensions than the original array,
# each value of the index array selects one row from the array being indexed
# and the resultant array has the resulting shape (size of row, number index elements).


#######################################
### BOOLEAN (MASK) INDEX ARRAYS
# Boolean arrays used as indices are treated in a different manner entirely than index arrays.
# Boolean arrays must be of the same shape as the initial dimensions of the array being indexed.
# As with index arrays, what is returned is a copy of the data, not a view as one gets with slices.

b = y>20
y[b]

# In general, when the boolean array has fewer dimensions than the array being indexed, this is equivalent to y[b, ...],
# which means y is indexed by b followed by as many : as are needed to fill out the rank of y.
b[:,5] # use a 1-D boolean whose first dim agrees with the first dim of y
y[b[:,5]]

# For example, using a 2-D boolean array of shape (2,3) with four True elements
# to select rows from a 3-D array of shape (2,3,5)
# results in a 2-D result of shape (4,5):
x = np.arange(30).reshape(2,3,5)
b = np.array([[True, True, False], [False, True, True]])
x[b]


#######################################
### COMBINING INDEXES AND SLICES
# Index arrays may be combined with slices.
y[np.array([0,2,4]),1:3]

# slicing can be combined with broadcasted boolean indices
b = y>20
y[b[:,5],1:3]


###############################################################################
### STRUCTURAL INDEXING
# To facilitate easy matching of array shapes with expressions and in assignments,
# the np.newaxis object can be used within array indices to add new dimensions with a size of 1
y = np.arange(35).reshape(5, 7)
y.shape
y2 = y[:,np.newaxis,:]
y2.shape


# This could be useful for performing operations on arrays that otherwise would needed a reshaping
x = np.arange(5)
x.shape
x[np.newaxis, :].shape
x[:, np.newaxis].shape
x[:, np.newaxis] + x[np.newaxis, :]


# The ellipsis syntax maybe used to indicate selecting in full any remaining unspecified dimensions. For example:
z = np.arange(81).reshape(3, 3, 3, 3)
z[1, ..., 2]
# this is equivalent to
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


### ASSIGNING VALUES TO INDEXED ARRAYS
x = np.arange(10)
x[2:7] = 1

# Note that assignments may result in data type changes
# if assigning higher types to lower types (like floats to ints)
# or even exceptions (assigning complex to floats or ints):
x[1] = 1.2
x[1]

x[1] = 1.2j

# Unlike some of the references (such as array and mask indices)
# assignments are always made to the original data in the array
x = np.arange(0, 50, 10)
x[np.array([1, 1, 3, 1])] += 1


### Indexing with list vs indexing with tuple
# lists are automatically converted to an array. So a multi-element list can be used to index a multi-dimension array.
# However, tuples are not automatically converted to an array. So a multi-element tuple will be used as a single-dimensional index.
z[[1,1,1,1]] # produces a large array equal to repeating z[1] four times
z[(1,1,1,1)] # returns a single value, which is the same as z[1,1,1,1]

# For this reason, tuples are a better data type to create index with,
# because they prevent the chance of repeating an index.
indices = (1,1,1,1)
z[indices]

# We can even use special proverbs to create indices
indices = (1,1,1,slice(0,2)) # same as [1,1,1,0:2]
z[indices]

indices = (1, Ellipsis, 1) # same as [1,...,1]
z[indices]

