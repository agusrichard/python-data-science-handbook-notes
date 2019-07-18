# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 00:06:04 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
# Aggregation and Grouping
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns

#------------------------------------------------------------------------------
# Planets Data

planets = sns.load_dataset('planets')
print(planets.shape)
print(planets.head())


#------------------------------------------------------------------------------
# Simple Aggregation in Pandas
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
print(ser)
print(ser.sum())
print(ser.mean())
df = pd.DataFrame({'A' : rng.rand(5), 'B' : rng.rand(5)})
print(df)
print(df.mean()) # by default mean is computed by column
print(df.mean(axis=1)) # axis=0 is column, axis=1 is row.. WTF?
print(df.mean(axis='rows'))
print(df.mean(axis='columns'))

# get back to planets
print(planets.dropna().describe())


#------------------------------------------------------------------------------
# GroupBy: Split, Apply, Combine

# Split, apply, combine
# apply is a summation aggregtion
# split: breaking up and grouping a DataFrame depending on the value of the key
# appy: computing some function, usually aggragates, transformation, or filtering
# combine: merges the result
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data': range(6)}, columns=['key', 'data'])
print(df)
print(df.groupby('key')) # passing the name of the desired key column
print(df.groupby('key').sum()) # to produce the result we have to apply the aggregate
print(df.groupby('key').mean())

# The groupby object
# Column indexing
planets.groupby('method')
planets.groupby('method')['orbital_period']
# above, we've selected Series group from the original DataFrame group 
# by reference by its column name
planets.groupby('method')['orbital_period'].median()
# this gives an idea of the general scale of orbital periods

# Iterative over groups
for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method, group.shape))

# Dismatch methods
planets.groupby('method')['year'].describe().unstack()

# Aggregate, filter, transform, apply
rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
    columns = ['key', 'data1', 'data2'])
print(df)
df.groupby('key').aggregate(['min', np.median, max])
df.groupby('key').aggregate({'data1' : 'min', 
                             'data2' : 'max'})

# Filtering
# drop data based on the group properties
def filter_func(x):
    return x['data2'].std() > 4

print(df); print(df.groupby('key').std());
print(df.groupby('key').filter(filter_func))
df.groupby('key').transform(lambda x: x - x.mean())
def norm_by_data2(x):
    # x is a DataFrame of group values
    x['data1'] /= x['data2'].sum()
    return x
print(df); print(df.groupby('key').apply(norm_by_data2))
L = [0, 1, 0, 1, 2, 0]
print(df); print(df.groupby(L).sum())
print(df); print(df.groupby(df['key']).sum())
df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
print(df2); print(df2.groupby(mapping).sum())
print(df2); print(df2.groupby(str.lower).mean())
df2.groupby([str.lower, mapping]).mean()
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)