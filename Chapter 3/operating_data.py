# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 09:44:10 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
# Operating on Data in Pandas
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd

# Ufuncs: Index Preservation
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
print(ser)
df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
                  columns=['A', 'B', 'C', 'D'])
print(df)
# the result will be another pandas object with indices preserved
print(np.exp(ser))
print(np.sin(df * np.pi / 4))

#------------------------------------------------------------------------------
#Ufuncs: Index Alignment

# Index allignment in Series
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')
print(population/area) # they align values with the same indices
print(area.index | population.index)
print(population|area) # pandas using Nan to mark missing data

A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
print(A + B)
# another way to represents the same thing as the above
print(A.add(B, fill_value=0)) # the missing data will get replaced by 0 (fill_value)

# Index allignment in DataFrame
A = pd.DataFrame(rng.randint(0, 20, (2, 2)), columns=list('AB'))
print(A)
B = pd.DataFrame(rng.randint(0, 10, (3, 3)), columns=list('BAC'))
print(B)
print(A+B)
print(A.add(B, fill_value=0))

fill = A.stack().mean() # stack method put column 1 to the bottom of column 0
print(A.add(B, fill_value=fill))


# Ufuncs: Operations between DataFrame and Series
A = rng.randint(10, size=(3, 4))
print(A)
df = pd.DataFrame(A, columns=list('QRST'))
print(df - df.iloc[0]) # subtracts df by its first row
print(df.subtract(df['R'], axis=0)) # operation by column... df are subtracted by its column of label 'R'
halfrow = df.iloc[0, ::2]
print(halfrow)
print(df - halfrow)     # the operation happens just for columns with label Q and S, since
                        # halfrow just has label Q and S, so values for R and T are empty
                        # preservation of labels mofo!!!


















