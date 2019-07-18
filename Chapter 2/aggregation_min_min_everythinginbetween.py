# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 05:57:31 2019

@author: Agus Richard Lubis
"""
#------------------------------------------------------------------------------

# Aggregations: Min, Max, and Everything in between

#------------------------------------------------------------------------------

import numpy as np

#------------------------------------------------------------------------------
# Summing the values in an array

big_array = np.random.rand(1000000)
%timeit sum(big_array)
%timeit np.sum(big_array) # by using numpy function, certainly the result will be faster


#-----------------------------------------------------------------------------
# Minimum and Maximum
print(min(big_array), max(big_array))
print(np.min(big_array), np.max(big_array))
print(big_array.min(), big_array.max(), big_array.sum())

# Multidimensional aggregates
M = np.random.random((3, 4))
print(M)
print(M.sum())
print(M.min(axis=0)) # along the column axis
print(M.max(axis=1)) # along the row axis
# The axis keyword specifies the dimension of the array that will be collapsed

# Other aggregation functions
# Nan-safe counterpart calculating without bothering the missing value
print(M.argmin(axis=0))


#------------------------------------------------------------------------------
# Examples: What is the average height of US presidents

