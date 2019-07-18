# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 23:17:21 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
# Introduction: Data Manipulation with Pandas
#------------------------------------------------------------------------------

# Pandas is a newer package built on top of Numpy
# DataFrames are esssentially multidimensional arrays with attached row and column labes

import numpy as np
import pandas as pd # our import convention

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Introducing pandas object

# There are three fundamental data structures on Pandas:
# Series, DataFrame, Index


#------------------------------------------------------------------------------
# The pandas series object
# One dimensinal array of indexed data

data = pd.Series([0.25, 0.5, 0.75, 1])
print(data)
print(data.values) # pandas wrap data as indices and values
print(data.index) # array-like object of type pd.Index
# data can be accessed by the associated index
print(data[1])
print(data[1:3]) # pandas series is more general than one-dimensional numpy's array

# Series as generalized numpy array
# what makes a difference from numpy array and pandas series is
# on numpy the indices are given implicitly
# and on pandas the indices are given explicitly

data = pd.Series([0.25, 0.5, 0.75, 1], index=['a', 'b', 'c', 'd'])
print(data)
print(data.index)
print(data['b'])
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 5, 3, 7])    # the index need not to be sequential
print(data)

# Series as specialized dictionary
# pandas series becomes more efficient than traditional dictionaries for certain cases

population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict) # we can convert dictionary into pandas series
print(population)
print(population['California'])
print(population['California' : 'Illinois']) # pandas series support slicing data
print(population['Texas' : 'Florida']) # The last is inclusive

#Contructing series object
print(pd.Series([2, 4, 6])) # the values come from list
print(pd.Series(5, index=[100, 200, 300])) # values or index can be fromlist or numpy array
print(pd.Series({2:'a', 1:'b', 3:'c'})) # we can pass dictionary to the Series contructor
print(pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])) # it works like accessing the specified values by index keyword


#------------------------------------------------------------------------------
# The Pandas dataframe object
# DataFrame can be thought as the generalized numpy array
# or specialized python dictionary

# DataFrame as a generalized Numpy array
# share the same index

area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
print(area)
states = pd.DataFrame({'population' : population, 'area' : area }) # joining to specified data, with the same index
print(states.index)
print(states.values)
print(states.columns) # give us the column lables

# DataFrame as specialized dictionary
# it is better to think DataFrame as generalized dictionary than generalized numpy array
print(states['area'])
print(states[0])

# Contructing DataFrame object

# from a single series object
print(pd.DataFrame(population, columns=['population']))
print(pd.DataFrame(area, columns=['area']))

# from a list of dict, by list comprehension
data = [{'a' : i, 'b' : 2*i} for i in range(3)]
print(data)
print(pd.DataFrame(data))
print(pd.DataFrame([{'a' : 1, 'b' : 2}, {'b' : 3, 'c' : 4 }])) # pandas fill the missing one

# from a dictionary of series objects
print(pd.DataFrame({'population':population, 'area':area}))

# from two dimensional numpy array
pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'], # so we specified the label for column and row
             index=['a', 'b', 'c'])

# from a numpy structured array
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
print(A)
print(pd.DataFrame(A))


#------------------------------------------------------------------------------
# The pandas index object
# immutable array or ordered set or multiset

ind = pd.Index([2, 3, 5, 7, 11])
print(ind)

# Index as immutable array
print(ind[1]) # retrieve one element
print(ind[::2]) # slicing with step 2
print(ind.size, ind.shape, ind.ndim, ind.dtype)
ind[0] = 1 # Index is immutable object, cannot mutate it

# Index as ordered set
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
print(indA & indB) # intersection
print(indA | indB) # union
print(indA ^ indB) # symmetric difference


