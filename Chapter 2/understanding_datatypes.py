# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 05:56:02 2019

@author: Agus Richard Lubis
"""

# Basic of all information, including images and sounds
# can be represented by numerical numbers in an array

import numpy as np

# integer array
np.array([1, 4, 2, 5, 3])
np.array([3.14, 4, 2, 3])
np.array([1, 2, 3, 4], dtype='float32')

# Nested lists result in mutlidimensional arrays
np.array([range(i, i+3) for i in [2, 4, 6]])

# Create a length-10 integer array filled with zeros
np.zeros(10, dtype='int')
# Create a 3x3 zeros array 
np.zeros((3, 3), dtype='int')
# Create a 3x5 array filled with ones
np.ones((3, 5), dtype='int')
# Create a 3x5 array filled with 3.14
np.full((3, 5), 3.14)
# Create a 3x3 array filled with character 'a'
np.full((3, 5), 'a', dtype='str')

# Create an array filled with a linear sequence
np.arange(0, 20, 2) # First inclusive, last exclusive, step 2

# Create an array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)

# Create a 3x3 array of uniformly distributed 
# random values between 0 and 1
np.random.random((3, 3))

# Create a 3x3 array of normally distributed 
# random values with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))

# Create a 3x3 array on random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))
np.random.randint(low=0, high=10, size=(3, 3))

# Create a 3x3 identity matrix
np.eye(3, dtype='int')

# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that
# memory location
np.empty(3)

