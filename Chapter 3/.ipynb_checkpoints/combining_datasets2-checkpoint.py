# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:48:39 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
# Combining Datasets: Merge and Join
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd

# Relational algebra is a formal set of rules for manipulating relationaldata, and forms
# conceptual foundation of operations available in most databases


#------------------------------------------------------------------------------
# Categories of Joins

# One-to-one joins
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
print(df1); print(df2)
df3 = pd.merge(df1, df2)
print(df3)
# pd.merge() function recognized that each DataFrame has an 'employee' colomn,
# and automaticalu joins using this column as a key.
# this function adjusting the column of employees and hire group so it represents the right thing
# in general, discards the index, except in the special case of merges by index

# Many-to-one joins
# are joins in which one of the key columns contains duplicate entries
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
print(df3); print(df4); print(pd.merge(df3, df4))
# note hat in column group there are duplicate entries, therefore pd.merge()
# adjusting and discards natural index

# Many-to-many joins
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
print(df1); print(df5); print(pd.merge(df1, df5))
# this above method creates new row to corresponding the necessity of good DataFrame
# for example, Bob is an acounting which has skills at math and spreadsheets
# then this merge method creates new row of bob and accounting to accomodate its corresponding two skills


#------------------------------------------------------------------------------
# Specification of the Merge Keys
# it looks for one or more matching column names between two inputs, and uses
# this as the key

# The on keyword
# specity the name of the key column
print(df1); print(df2); print(pd.merge(df1, df2, on='employee'))

# The left_on and right_on keywords
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
print(df1); print(df3)
print(pd.merge(df1, df3, left_on='employee', right_on='name'))
# note that this result has redundant column
# below, we are dropping the redundant column
print(pd.merge(df1, df3, left_on='employee', right_on='name').drop('name', axis=1))

# The left_index and right_index keywords
df1a = df1.set_index('employee') # this makes the column name 'employee' becomes the index
df2a = df2.set_index('employee')
print(df1a); print(df2a)
print(type(df1a))
print(pd.merge(df1a, df2a, left_index=True, right_index=True)) 
# the keywords right_index and left_index must both set to be True 
# below, it shortened form using instance method of DataFrame object
print(df1a); print(df2a); print(df1a.join(df2a))
print(df1a); print(df3)
print(pd.merge(df1a, df3, left_index=True, right_on='name'))
# index put to the left, and the column with same keys, one of these columns are used
# which is the 'name' one


#------------------------------------------------------------------------------
# Specifying Set Arithmetic for Joins

df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                    columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                    columns=['name', 'drink'])
print(df6); print(df7); 
print(pd.merge(df6, df7)) # the result is the intersection of the two sets of inputs
# note that column with the same name is 'name' and the common key is only Mary, then
# the result by default only will resulting intersection on mary
# this is what is known as inner join
print(pd.merge(df6, df7, how='inner')) # same as before, just for emphasizing
print(pd.merge(df6, df7, how='outer')) # now is the union of two dataset, and fill the empty cell by nan
print(pd.merge(df6, df7, how='left')) # using column name of the left dataset
print(pd.merge(df6, df7, how='right'))


#------------------------------------------------------------------------------
# Overlapping Column Names: The suffix keyword
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
print(df8); print(df9)
print(pd.merge(df8, df9, on='name'))
# note: because the output would have two conflicting column names, 
# the merge function automaticaly append a suffix_x or _y to make the output columns unique
print(df8); print(df9)
print(pd.merge(df8, df9,on='name', suffixes=['_L', '_R']))


#------------------------------------------------------------------------------
# Example: US States Data

pop = pd.read_csv('state-population.csv')
areas = pd.read_csv('state-areas.csv')
abbrevs = pd.read_csv('state-abbrevs.csv')
print(pop.head()); print(areas.head()); print(abbrevs.head())
# above, printing the head five from dataset
merged = pd.merge(pop, abbrevs, how='outer', 
                  left_on='state/region', right_on='abbreviation')
merged = merged.drop('abbreviation', axis=1)
print(merged.head())
# check for any mismatches, and looking for rows of null
print(merged.isnull().any(axis=0))
print(merged[merged['population'].isnull()].head())
# above, print head of population with null value
print(merged.loc[merged['state'].isnull(), 'state/region'].unique())
# our population data includes entries for Puerto Rico (PR) and 
# United State as a whole (USA), while keys do not appear in the abbreviation key
merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()
final = pd.merge(merged, areas, on='state', how='left')
print(final.isnull().any())
print(final['state'][final['area (sq. mi)'].isnull()].unique())
final.dropna(inplace=True)
print(final.head())
data2010 = final.query("year == 2010 & ages == 'total'")
print(data2010.head())
data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']
density.sort_values(ascending=False, inplace=True)
density.head()
density.tail()























