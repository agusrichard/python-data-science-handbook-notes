# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:38:30 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
# Fancy Indexing
#------------------------------------------------------------------------------

# Fancy indexing is conceptually means passing an array of indices to access
# multiple array elements at once

import numpy as np

#------------------------------------------------------------------------------
# Exploring fancy indices

rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
print(x)
print([x[3], x[7], x[2]])

ind = [3, 7, 4]
print(x[ind]) # returning values of x with index passed by ind

ind = np.array([[3, 7],
                [4, 5]])
print(x[ind]) # returning values of the specified index, with shape of ind

X = np.arange(12).reshape((3, 4))
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
print(X[row, col]) # remember remember row then column, pairing index for row and colum
print(X[col, row]) # will resulting an error
print(row)
print(row[:, np.newaxis])
print(X[row[:, np.newaxis], col])


#------------------------------------------------------------------------------
# Combined indexing

print(X)
print(X[2, [2, 0, 1]]) # Combine fancy indexing with simple indexing
print(X[1:, [2, 0, 1]]) # Combine fancy indexing with slicing
mask = np.array([1, 0, 1, 0], dtype=bool)
print(mask)
print(row[:, np.newaxis])
print(X[row[:, np.newaxis]])
print(X[row[:, np.newaxis], mask])


#------------------------------------------------------------------------------
# Example: Selecting random points

mean = [0, 0]
cov = [[1, 2],
       [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)

import matplotlib.pyplot as plt
import seaborn; seaborn.set()

plt.scatter(X[:, 0], X[:, 1]) # normally distributed points

indices = np.random.choice(X.shape[0], 20, replace=False)
print(indices)
selection = X[indices]
print(selection)

plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1],
            facecolor='none', s=200);
            
            
# Modifying values with fancy indexing

x = np.arange(10)
i = np.array([2, 1, 8, 4])
print(x)
print(i)
x[i] = 99
print(x)

x[i] -= 10
print(x)

x = np.zeros(10)
print(x)
print(x[[0, 0]])
x[[0, 0]] = [4, 6]
print(x)

i = [2, 3, 3, 4, 4, 4, 4]
x[i] += 1 # the assignment happens in multiple times, but not the augmentation
print(x)

x = np.zeros(10)
np.add.at(x, i, 1)
print(x)


#------------------------------------------------------------------------------
# Example: Bining data

np.random.seed(42)
x = np.random.randn(100)

# compute histogram by hand
bins = np.linspace(-5, 5, 20)
print(bins)
counts = np.zeros_like(bins)
print(counts)

# find appropriate bin for each x
i = np.searchsorted(bins, x)
