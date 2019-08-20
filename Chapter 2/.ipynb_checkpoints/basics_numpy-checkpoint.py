# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 07:58:31 2019

@author: Agus Richard Lubis
"""

# NumPy array attributes
import numpy as np
np.random.seed(0) # Seed for reproductibility

x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(3, 4))
x3 = np.random.randint(10, size=(3, 4, 5))

# Each array has these three attributes, plus 3 again!!!
print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
print("dtype:", x3.dtype)
print("itemsize:", x3.itemsize, "bytes") # Size of single element
print("nbytes:", x3.nbytes, "bytes") # Total size

#------------------------------------------------------------------------------
# Array indexing: Accessing Single Elements

# Element accessing still the same like python list
# but becareful when trying to reassign value and changing
# its whole elements in an array

#------------------------------------------------------------------------------
# Array Slicing: Accessing Subarrays

# General syntax for x[start:stop:step]
x = np.arange(10)
print(x[:5]) # printing elements from beginning to 5 exclusive
print(x[5:]) # printing elements from elements #5 to the end
print(x[::2]) # step 2 accessing

# Multidimensional subarrrays
print(x2[:2, :3]) # two rows, three columns
print(x2[1:, 2:]) # 2x2 matrix... probably
print(x2[::-1, ::-1]) # reversing

# Accessing array rows and columns
print(x2[:, 0])
print(x2[0]) # for rows accessing

# Subarrays as no-copy views
# array slices return views rather than copy
x2_sub = x2[:2, :2]
print(x2_sub)
x2_sub[0, 0] = 99 # This line changes the x2_sub and x2
print(x2_sub)
print(x2)

# Creating copies of arrays
x2_sub_copy = x2[:2, :2].copy()
print(x2_sub_copy)
x2_sub_copy[0, 0] = 42 # This line changes only the x2_sub_copy, not the prior matrix
print(x2_sub_copy)
print(x2)
print(x2_sub)

#------------------------------------------------------------------------------
# Reshaping of arrays
grid = np.arange(1, 10).reshape((3,3))
print(grid)
x = np.array([1, 2, 3])
# row vector via reshape
x.reshape((1, 3))
x[np.newaxis, :]
x.reshape((3, 1))
x[:, np.newaxis]
# Testing
sekar = np.array([1, 2, 3, 4, 5])
sekar[:, np.newaxis] # Changing it into column vector


#------------------------------------------------------------------------------
# Array concatenation and splitting

# Concatenation of arrays
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y]) # This method works like appending one list to another
z = [99, 99, 99]
print(np.concatenate([x, y, z]))
print(np.concatenate(x, x))
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
np.concatenate([grid, grid], axis=0) # concatenate aling the first axis (to the bottom)
np.concatenate([grid, grid], axis=1) # concatenate along the second axis (to the right)
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7], 
                [6, 5, 4]])
# vertically stack the arrays
np.vstack([x, grid])
# horizontally stack the arrays
y = np.array([[99],
[99]])
np.hstack([grid, y])
# testing
sekar = np.eye(3,3, dtype='int')
print(sekar)
np.vstack([sekar, [0, 0, 0]])
np.hstack([sekar, [[0], [0], [0]]])

# Splitting of arrays
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)
grid = np.arange(16).reshape((4, 4))
print(grid)
upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)
left, right = np.hsplit(grid, [2])
print(left)
print(right)


# Testing some knowledge
x = np.array([1, 2, 3])
x.reshape(1, 3)
print(x)
x.reshape(3, 1)
print(x)
x[np.newaxis, :]
x[:, np.newaxis]
x = np.linspace(1, 10, 12).reshape(3, 4)
x = x[:, :, np.newaxis]
x = x.reshape(2, 3, 2)
y = np.linspace(10, 20, 12).reshape(3, 4)
np.concatenate([x, y])
np.concatenate([x, y], axis=1)
np.hstack([x, y])
np.vstack([x, y])
np.dstack([x, y])
line = np.linspace(0, 10, 100)
print(line)
np.split(line, list(range(0, 100, 5)))
lst = [x for x in range(1, 5)]
print(lst)
