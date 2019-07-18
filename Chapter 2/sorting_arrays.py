# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 03:26:12 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
# Sorting arrays
#------------------------------------------------------------------------------

import numpy as np

#------------------------------------------------------------------------------
# Fast sorting in numpy: np.sort and np.argsort
# np.sort uses an big-O(N logN), quicksort

x = np.array([2, 1, 4, 3, 5])
print(x)
np.sort(x) # returns void, because it changes the value of x
print(np.sort(x))

x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x) # returns the indices of sorted array
print(i)
print(x[i]) # printing the sorted array

#Sorting along rows and columns
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
print(X)

# keep in mind that the relationships between columns and rows will be lost
print(np.sort(X, axis=0)) # sort each column
print(np.sort(X, axis=1)) # sort each row


#------------------------------------------------------------------------------
# Partial sorts: Partitioning

x = np.array([7, 2, 3, 1, 6, 5, 4])
print(np.partition(x, 3)) # resulting an array 3 smallest value, and put it in on the left
print(X)
print(np.partition(X, 2, axis=1))


#------------------------------------------------------------------------------
# Example: k-Nearest Neighbors

X = rand.rand(10, 2)
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.scatter(X[:, 0], X[:, 1], s=100)

differences = X[:, np.newaxis, :] - X[np.newaxis, :, :] # make a 3-dimensional array...
print(differences.shape)
sq_differences = differences ** 2
print(sq_differences.shape)
dist_sq = sq_differences.sum(-1)
dist_sq.shape
nearest = np.argsort(dist_sq, axis=1)
print(nearest)
K = 2
nearest_partition = np.argpartition(dist_sq, K+1, axis=1)
plt.scatter(X[:, 0], X[:, 1], s=100)
for i in range(X.shape[0]):
    for j in nearest_partition[i, :K+1]:
        # plot a line from X[i] to X[j]
        # use some zip magic to make it happen:
        plt.plot(*zip(X[j], X[i]), color='black')


#------------------------------------------------------------------------------
# Structured data: Numpy's structured arrays
        
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

x = np.zeros(4, dtype=int)
# use compound data type for structured arrays
data = np.empty(4, dtype={'names' : ('name', 'age', 'weight'),
                          'formats' : ('U10', 'i4', 'f8')})
print(data)
print(data.dtype)

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)

# get all names
print(data['name'])
print(data[0])
print(data[-1]['name'])
print(data[data['age'] < 30]['name'])

# We will cover the more advanced one, in pandas package


#------------------------------------------------------------------------------
# Creating Structured Arrays

np.dtype({'names':('name', 'age', 'weight'),
          'formats':('U10', 'i4', 'f8')})
np.dtype({'names':('name', 'age', 'weight'),
          'formats':((np.str_, 10), int, np.float32)})
np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
np.dtype('S10,i4,f8')


#------------------------------------------------------------------------------
# More advanced compound types

tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
print(tp)
X = np.zeros(1, dtype=tp)
print(X[0])
print(X['mat'][0]) 


#------------------------------------------------------------------------------
# RecordArrays: Structured arrays with a twist

print(data['age'])
data_rec = data.view(np.recarray) # it works like turning str to some attributes
print(data_rec.age)
%timeit data['age']
%timeit data_rec['age']
%timeit data_rec.age


#------------------------------------------------------------------------------
# Knowledge testing

data = np.zeros(4, dtype={'names': ('name', 'age', 'weight'), 
                          'formats' : ('U10', 'i4', 'f8')})
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)
