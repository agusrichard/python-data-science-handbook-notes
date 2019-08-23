# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 01:04:38 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
# Data Indexing and Selection
#------------------------------------------------------------------------------

import pandas as pd

#------------------------------------------------------------------------------
# Data selection in Series

# Series as dictionary
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print(data)
print(data['b'])
print('a' in data)
print(data.keys())
print(list(data.items()))
data['e'] = 1.25 # assigning a new value to data
print(data)

# Series as one-dimensional array
print(data['a':'d']) # slicing using explicit index
print(data[0:2]) # slicing using implicit index
print(data[(data > 0.3) & (data < 0.8)]) # masking
# Note!!! when we use masking the condition will return a boolean array/list/series
print(data[['a', 'e']]) # fancy indexing
print(data[['b', 'd']])

# Indexers: loc, iloc and ix
# indexing operation data[1] will use the explicit indices
# meanwhile, data[1:2] will us the implicit indices

data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
print(data)
print(data[1]) # explicit indexing
print(data[2:3]) # implicit indexing

print(data.loc[1]) # explicit indexing
print(data.loc[1:3]) # explicit indexing

print(data.iloc[1]) # implicit indexing
print(data.iloc[1:3]) #implicit indexing


#------------------------------------------------------------------------------
# Data selection in DataFrame
# DataFrame works in many ways like two-dimensional or structured array

# DataFrame as a dictionary
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
print(data)
print(data[1:3])
print(data['area']) # accessing certain column with dictionary accessing
print(data.area) # similar as before
print(data.area is data['area'])
# this shorthand form only works if it is a string and no-space allowed or conflict the method of dataframe
print(data.pop is data['pop']) # if there is a conflicting column label with method, the method wins
data['density'] = data['pop'] / data['area'] # creating a new column for density
print(data['density'])

# DataFrame as two-dimensional array
print(data.values)
print(data.T) # transposing the data
print(data.values[0])
print(data['area'])
# using loc, iloc, ix for making dataframe looks like numpy array
print(data.iloc[:3, :2]) # DataFrame index and column labels are maintained
print(data.loc[:'Illinois', :'pop'])
# ix indexer, is fucking hybrid of loc and iloc
print(data.ix[:3, :'pop'])
# masking and fancy indexing
print(data.loc[data.density > 100], ['pop', 'density'])
print(data.loc[data.density > 100])
# using the standard way to modifying values
data.iloc[0, 2] = 90
print(data)

# Additional indexing conventions
print(data['Florida':'Illinois']) # by explicit indexing
print(data[1:3]) # by implicit indexing
print(data[data.density > 100]) # masking operations