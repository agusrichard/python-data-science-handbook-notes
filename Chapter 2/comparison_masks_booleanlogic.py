# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 05:54:46 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
# Comparison, Masks, and Boolean Logic
#------------------------------------------------------------------------------

import numpy as np

#------------------------------------------------------------------------------
# Comparison Operators as ufuncs

x = np.array([1, 2, 3, 4, 5])
print(x < 3) # returning list of boolean values
print(x != 3)       
print((2 * x) == (x ** 2)) # element by element comparison of two arrays

rng = np.random.RandomState(0)
print(rng)
x = rng.randint(10, size=(3, 4))
print(x)


#------------------------------------------------------------------------------
# Working with boolean arrays

# Counting entries
print(np.count_nonzero(x < 6)) # how many values less than 6
print(np.count_nonzero([1, 2, 3, 4, 5]))
print(np.sum(x < 6)) # this condition part, is returning a list

print(np.sum(x < 6, axis=0)) # return list where value is less than six in each column
print(np.any(x > 8)) # works like existential quantifier
print(np.all(x > 1))
print(np.all(x < 10))
print(np.any(x >= 7, axis=0))

# Boolean opertors
x = rng.randint(50, size=(10, 10))
y = rng.randint(50, size=10)
print((y > 10) & (y < 40))
print((x > 10) & (x < 40))
inbetween = (x > 10) & (x < 40)
outer = (x <= 10) | (x >= 40)

print(np.sum(inbetween))
print(np.sum(outer))


#------------------------------------------------------------------------------
# Boolean arrays as masks
x = rng.randint(10, size=(3, 4))
print(x)
print(x < 5)
print(x[x < 5]) # returning an array satisfiying the condition
                # masking operation
print(x > 5)


#------------------------------------------------------------------------------
# Knowledge testing

import numpy as np
X = np.random.normal(size=(100, 5))
print(X.mean(axis=0))
print((X[0, :] > 0) & (X[0, :] < 0.5))
print((X[1, :] > 0) & (X[1, :] < 0.5))
print((X > 0) & (X < 0.5))
print(np.count_nonzero((X > 0) & (X < 0.5)))
print(np.count_nonzero(~(((X <= 0) | (X >= 0.5)))))
print(np.sum((X > 0) & (X < 0.5), axis=0))
print(np.sum((X > 0) & (X < 0.5), axis=0).sum())
Y = np.random.randint(-10, 10, size=(10, 5))
print(Y[Y >= 0])
print(Y[[1, 2, 3], [1, 2, 3]])
row = np.array([1, 2, 3])
print(Y[row[:, np.newaxis], [True, False, True, False, True]])
