# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 13:20:01 2019

@author: Agus Richard Lubis
"""
#------------------------------------------------------------------------------
# Handling Missing Data
#------------------------------------------------------------------------------

# dataset in real world rarely in a clean and homogeneous form.
# there must exist missing data

import numpy as np
import pandas as pd

#------------------------------------------------------------------------------
# Trade-Offs in Missing Data Conventions
# two strategies: mask or sentinel value
# use of seperat =e mask array requires allocation of additional boolean array
# which adds overhead in both storage and computation
# sentinel value reduces the range of valid values 
# and require extra logic in cpu and gpu arithmetic

#------------------------------------------------------------------------------
# Missing Data in Pandas

# None: Pythonic missing data
vals1 = np.array([1, None, 3, 4]) # this none will make the array type become object
print(vals1)
print(vals1.dtype)

# test for performannce
for dtype in ['object', 'int']:
    print("dtype = ", dtype)
    %timeit np.arange(1E6, dtype=dtype).sum()
    print()
# surely, datatype int performs faster calculation than type object

# if we perform aggregations like sum and min across an array with None value
# we will get an error
print(vals1.sum()) # error... addition between None and integer is undefined

# NaN: Missing numerical data
# NaN is special floating point value
vals2 = np.array([1, np.nan, 3, 4])
print(vals2.dtype) # you'll float too
# support fast operations becomes we are using native data types
# aware tha NaN works like a fucking virus. it infects data point it touches
print(1 + np.nan) # results nan
print(0 * np.nan) # results nan
# aggregates over array wont give you error
print(vals2.sum(), vals2.min(), vals2.max())
print(np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)) # ignores the missing value
# keep in mind that NaN is a floating-point representation
# there is no nan for string, boolean, or anything

# Nan and None in pandas
# pandas operates on Nan or non interchangebly
print(pd.Series([1, np.nan, 2, None])) # None changes to NaN
x = pd.Series(range(2), dtype=int)
print(x)
x[0] = None
print(x)
y = pd.Series([False, True, None], dtype=bool) # None becomes False
print(y)


#------------------------------------------------------------------------------
# Operating on Null Values
# essential method for null values
# isnull() : generate a boolean mask indicating missing values
# notnull() : opposite on isnull()
# dropna() : return a filtered version of the data
# fillna() : return copy of the data with missing values filled or imputed

# Detecting null values
data = pd.Series([1, np.nan, 'hello', None]) 
print(data.isnull()) # series object has instance method isnull()
print(data[data.notnull()]) # just returns the not null values

# Droping null values
print(data.dropna()) # removes NA values
# for DataFrame
df = pd.DataFrame([[1, np.nan, 2],
                   [2, 3, 5],
                   [np.nan, 4, 6]])
print(df)
# we can only drop full rows or full columns
# by default dropna() will drop all rows in which any null value is present
print(df.dropna()) # since rows without nan is 1, then it return row 1
print(df.dropna(axis=1)) # since columns without nan is 2, then it returns column 2

df[3] = np.nan
print(df)
print(df.dropna(axis='columns', how='all')) # will drop columns only when nan is on all rows in that colum
print(df.dropna(axis='rows', thresh=3)) # thresh keyword represents the minimum number of non-null values

# Filling null values
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
print(data)
print(data.fillna(0)) # fill the nan by float 0.0
print(data.fillna(method='ffill')) # forward-fill: propagates the previous value to the next, which a nan
print(data.fillna(method='bfill')) # back-fill: propagates the next value to the nan value, before
print(df)
print(df.fillna(0, axis=1))
print(df.fillna(0))