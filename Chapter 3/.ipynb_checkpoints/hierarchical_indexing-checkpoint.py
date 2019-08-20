# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:02:08 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
# Hierarchical Indexing
#------------------------------------------------------------------------------
# also known as multi-indexing

import numpy as np
import pandas as pd

#------------------------------------------------------------------------------
# A multiple indexed Series

# The bad way
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
print(pop)
print(pop[('California', 2010):('Texas', 2000)]) # ordinary indexing and slicing
# messing munging
print([i for i in pop.index if i[1] == 2010])
print(pop[[i for i in pop.index if i[1] == 2010]])

# The better way: Pandas MultiIndex
index = pd.MultiIndex.from_tuples(index)
print(index)
pop = pop.reindex(index) # it doesn't change its representation on variable explorer
print(pop)
# to access all data for which the second index is 2010
print(pop[:, 2010]) # accessing data point where its index is 2010
print(pop[:, 2000])
print(pop[2010]) # error, why?

# MultiIndex as extra dimension
# _.unstack() method will convert MultiIndex to conventional DataFrame
pop_df = pop.unstack() # the second index became column's label
print(pop_df)
print(pop_df.stack()) # undo the effect of unstack()
# MultiIndexing represent two-dimensional data within a one dimensional Series
# extra level in a multi-index represents extra dimension of data
pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})
print(pop_df)
f_u18 = pop_df['under18'] / pop_df['total'] # compute fraction of people under 18 by year
print(f_u18.unstack())


#------------------------------------------------------------------------------
# Methods of MultiIndex creation
index2 = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], # if i pass index2 into here, the effect is the same
                  columns=['data1', 'data2'])
# the work of creating the Multiindex is done in the background
print(df)
data = {('California', 2000): 33871648,
       ('California', 2010): 37253956,
       ('Texas', 2000): 20851820,
       ('Texas', 2010): 25145561,
       ('New York', 2000): 18976457,
       ('New York', 2010): 19378102}
pd.Series(data) # MultiIndex created in the background

# Explicit MultiIndex constructors
# from list, giving the index values within each level
print(pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]]))
# from tuples, giving the multiple index values of each point
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
# form cartesian product of single indices
pd.MultiIndex.from_product([['a', 'b'], [1, 2]]) # write it as list
# creating it by its internal encoding
pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
              labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
# we can pass these into Series or DataFrame's index keyword

# MultiIndex level names
pop.index.names = ['state', 'year']
print(pop)

# MultiIndex for columns
# on before we know that rows can have levels but, columns too
# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

# mock some data
data = np.round(np.random.rand(4, 6), 1)
data[:, ::2] *= 10
data += 37
health_data = pd.DataFrame(data, index=index, columns=columns)
print(health_data)
# this is fundamentally four-dimensional data
print(health_data['Guido']) # we are accessing the data with column label's GUIDO


#------------------------------------------------------------------------------
# Indexing and Slicing a MultiIndex

# Multiply indexed Series
print(pop)
print(pop['California', 2000]) # accessing single element
# partial indexing, it means that the lower level is still maintained
print(pop['California'])
print(pop.loc['California':'New York']) # slicing the data, as long as it is sorted
# perform partialindexing on lower levels
print(pop[:, 2000])
print(pop[pop > 22000000]) # selection based on Boolean mask
print(pop[['California','Texas']]) # fancy indexing mutufaka

# Multiple indexed DataFrames
print(health_data)
print(health_data['Guido', 'HR']) # rememberthat columns are primary
print(health_data.iloc[:2, :2]) # print bob the builder
print(health_data.loc[:, ('Sue', 'HR')])
print(health_data.loc[(:, 1), (:, 'HR')]) # will results on error, why? because we hybrid iloc and loc
idx = pd.IndexSlice # it lookslike a constructor
print(health_data.loc[(2013, 1), ('Bob', 'HR')])
print(health_data.loc[idx[:, 1], idx[:, 'HR']])


#------------------------------------------------------------------------------
# Rearranging Multi-indices

# Sorted and unsorted indices
# many MutliIndex slicing operations will fail if the index is not sorted
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
print(data)
# try, will it resulting an error
try:
    data['a':'b']
except KeyError as e:
    print(type(e))
    print(e)
# this is because MultiIndex is not sorted
# sorting it
data = data.sort_index() # will return a sorted data, not void!
print(data)
try:
    print(data['a':'b'])
except KeyError as e:
    print(type(e))
    print(e)

# Stacking and unstacking indices
print(pop)
print(pop.unstack(level=0)) # it means it unstack the state
print(pop.unstack(level=1)) # unstack the year
print(pop.unstack().stack()) # didn't do anything just like me

# Index setting and resetting
# from Series to DataFrame
# keyword name is for specifying column label of values
pop_flat = pop.reset_index(name='population') # resulting of DataFrame
print(pop_flat)
print(type(pop_flat))
print(pop_flat.set_index(['state', 'year'])) # reverse the process of reset index
print(type(pop_flat.set_index(['state', 'year'])))


#------------------------------------------------------------------------------
# Data Aggregations on Multi-indices

print(health_data)
data_mean = health_data.mean(level='year')
print(data_mean) # the mean computed on level
print(health_data.mean(level='visit'))
print(data_mean.mean(axis=1, level='type')) # axis means it is considered by columns, and level is how it is computed















