# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 11:48:19 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
#-------------------------!EXPERIMENTATION!------------------------------------
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd

rng = np.random.RandomState(21)

#------------------------------------------------------------------------------
# Introduction
#------------------------------------------------------------------------------
# Series object

arr = rng.randint(0, 10, size=10)
x = pd.Series(arr, index=list('abcdefghij'))
print(x)
print(x.values)
print(x.index)

area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})



#------------------------------------------------------------------------------
# DataFrame object

# From series
df = pd.DataFrame(x, columns=['numbers'])
print(df)
# From list of dictionary 1
lod = [{'a' : 1, 'b' : 1}, {'a' : 2, 'b' : 2}, {'a' : 3, 'b' : 3},
       {'a' : 4, 'b' : 5}]
df1 = pd.DataFrame(lod)
print(df1)
# From list of dictionary 2
lod1 = [{'a' : i, 'b' : i**2} for i in range(10)]
df2 = pd.DataFrame(lod1)
print(df2)
print(df2.index)
print(df2.values)
data = df2.values
print(data)
# From a dictionary of series objects
ser1 = pd.Series(arr)
print(ser1)
ser2 = pd.Series(arr)
print(ser2)
df3 = pd.DataFrame({'ser1' : ser1, 'ser2' : ser2})
print(df3)
# From a two dimensional numpy array
df4 = pd.DataFrame(rng.randint(0, 10, size=(3, 3)),
                   columns=['col1', 'col2', 'col3'])
print(df4)
print(df4.values)
# From a numpy structured array
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
print(A) # A is a list of tuple object 
df5 = pd.DataFrame(A)
print(df5)



#------------------------------------------------------------------------------
# Data Indexing and Selection
#------------------------------------------------------------------------------

# Data selection in Series

a = pd.Series(rng.normal(size=10))
print(a)
print(a[0])
print(a[0:5])
a.index = list('abcdefghij')
print(a)
print(a['a'])
print(a['c' : 'h'])
print(list(a.items()))
print(a[a  > 0])
print(np.count_nonzero(a > 0))
print(a)
print(a.loc['a':'b'])
print(a.iloc[1:3])


# Data selection in DataFrame
arr = np.array([[1, 2, 3],
                [2, 3, 4],
                [3, 4, 5]])
x = pd.DataFrame(arr, columns=['col1', 'col2', 'col3'])
x.iloc[:, 2] = [1, 1, 1] # changing the third column with iloc
x.ix[:, 'col1'] = [1, 1, 1] # using ix, hybrid of iloc and loc
x['col4'] = [1, 1, 1]
arr1 = rng.normal(size=(10, 5))
X = pd.DataFrame(arr1, columns=['col1', 'col2', 'col3', 'col4', 'col5'],
                 index=list(range(10)))
X_values = X.values
X_values[0, 0] = 0  # so this numpy object also change the pandas DataFrame,
                    # since, this numpy object is a view of DataFrame
print(x.loc[:1, ['col1', 'col2']]) # index 1 is included because loc is using explicit indexing
print(x.loc[x['col3'] > 3, ['col1', 'col2']])
print(X.loc[X['col3'] > 0, ['col2', 'col5']])
print(np.count_nonzero(X['col3'] > 0)) # four data in col3 where its value is greater than 0



#------------------------------------------------------------------------------
# Operating on Data in Pandas
#------------------------------------------------------------------------------

# Index allignment in Series
a = pd.Series(rng.randint(0, 10, 3), index=[0, 1, 2])
b = pd.Series(rng.randint(0, 10, 3), index=[1, 2, 3])
print(a + b) # missing values on array treated as NaN... resulting a Union
print(a.add(b, fill_value=0))
print(a.multiply(b, fill_value=0))

# Index allignment in DataFrame
x = pd.DataFrame(rng.randint(0, 10, (2, 2)), columns=['A', 'B'])
y = pd.DataFrame(rng.randint(0, 10, (3, 3)), columns=['B', 'A', 'C'])
print(x); print(y)
print(x + y) # the order of column got adjusted relative to x and the missing values are NaN
print(x.add(y, fill_value=0))   # the process is adjusting the shape and fill the missing values by zero
                                # then do add
                                
# Ufuncs: Operations between DataFrame and Series
A = pd.DataFrame(rng.randint(0, 10, size=(10, 3)), columns=list('ABC'))
print(A)
print(np.exp(A))
print(np.sin(A)) # in radians
print(A - A.iloc[0])
print(A.subtract(A['A'], axis=0)) # weird result if the axis is 1
# every operation with a NaN, the result will also be NaN



#------------------------------------------------------------------------------
# Handling Missing Data
#------------------------------------------------------------------------------

#! everything touched by nan will become nan... Becareful!

A = pd.DataFrame(rng.randint(0, 10, size=(10, 3)), columns=list('ABC'))
A.iloc[[2, 5, 7], [0, 2]] = np.nan  # assing nan to several points
print(A)
print(np.nanmean(A, axis=0))
print(np.nanmean(A.iloc[2, :]))

# Detecting null values
print(A['A'].isnull()) # checking null values in column 'A'
print(A[A['A'].notnull()]) # eliminate the nan values in column 'A'

# Droping null values
A.iloc[3, 1] = np.nan
print(A.dropna(axis='rows')) # droping rows with nan values
print(A.dropna(axis='rows', how='all')) # will drop rows with values all nan, won't have any effect in this case
print(A.dropna(axis='columns', thresh=8)) # thresh specify the minimum number of non-null values

# Filling null values
avg = np.nanmean(A, axis=0)
print(avg)
print(A.isnull())
A.iloc[:, 0] = A.iloc[:, 0].fillna(avg[0]) # filling each nan of each column by its mean
A.iloc[:, 1] = A.iloc[:, 1].fillna(avg[1])
A.iloc[:, 2] = A.iloc[:, 2].fillna(avg[2])
new_avg = np.mean(A, axis=0) # new average is the same as the previous average



#------------------------------------------------------------------------------
# Hierarchical Indexing
#------------------------------------------------------------------------------

index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
print(pop)

index1 = [('a', 'x', 0), ('a', 'x', 1), ('a', 'y', 0), ('a', 'y', 1),
          ('b', 'x', 0), ('b', 'x', 1), ('b', 'y', 0), ('b', 'y', 1),
          ('c', 'x', 0), ('c', 'x', 1), ('c', 'y', 0), ('c', 'y', 1)]
numbers = rng.randint(0, 10, 12)


 # reset old index with a new one (MultiIndex)
print(pop)
print(pop['California'])

index1 = pd.MultiIndex.from_tuples(index1) # three level index
numbers = pd.Series(numbers, index=index1) 
print(numbers)
print(numbers[:, :, 0]) # the comma-separated list specify the intended level of accessing

# MultiIndex as extra dimension
pop_df = pop.unstack() # change into DataFrame
print(pop_df)
print(pop_df.stack())

pop_df = pd.DataFrame({'total': pop,                    # creating DataFrame of Series and list
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})
print(pop_df)
print(pop_df['under18'] / pop_df['total'])
f_u18 = pop_df['under18'] / pop_df['total']
print(f_u18.unstack())

# Methods of MultiIndex creation
df = pd.DataFrame(rng.randint(0, 1000000, (4, 3)),
                  index=[['a', 'a', 'b', 'b'], [0, 1, 0, 1]],
                  columns=['col1', 'col2', 'col3'])
print(df)

# MultiIndex level names and for columns
pop.index.names = ['state', 'year']

index = pd.MultiIndex.from_product([['ind1', 'ind2'], ['a', 'b']],
                                   names=['index1', 'index2'])
column = pd.MultiIndex.from_product([['col1', 'col2'], ['A', 'B']],
                                    names=['column1', 'column2'])
df1 = pd.DataFrame(rng.randint(0, 1000000,(4, 4)),
                   index=index, columns=column)
print(df1)

#------------------------------------------------------------------------------
# Indexing and Slicing a MultiIndex

# Series
index = pd.MultiIndex.from_tuples(index)
pop = pd.Series(populations, index=index)
print(pop)
print(pop['California'])
print(pop[:, 2010])
print(pop[pop > 25000000])

# DataFrame
print(df1.values)
print(df1['col1'])
print(df1.loc['col1', 'A']) # surely is an error, because there is no index label 'col1'
print(df1['col1', 'A']) 
print(df1.iloc[1:3, 1:3]) # it works, because indexing with iloc is less error-prone
print(df1.loc[('ind1', 'b'), ('col1', 'B')]) # it works precisely like if the index is the ordinary index
print(df1.loc[('ind1', ['a', 'b']), :])
print(df1.loc[['ind1', ['a', 'b']], :]) # error, because accessing rows and columns must enclosed between tuples
print(df1.loc[(:, 'a'), :]) # error
idx = pd.IndexSlice     # now the other way to solve the error problem
print(df1.loc[idx[:, 'a'], idx[:, 'A']])
print(df1.loc[idx[:, 'b'], idx[:, 'B']])
print(df1.iloc[1:3, 1])

#------------------------------------------------------------------------------
# Rearranging Multi-Indices

# Sorted and unsorted indices
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
df2 = pd.Series(rng.randint(0, 1000000, 6), index=index)
print(df2)
df2 = df2.sort_index()
print(df2)
index = index.sort() # a good rule of thumbs is sort the index first then use it
print(index)
df2.index.names = ['char', 'nums']

# Stacking and unstacking indices
print(df2.unstack(level=0))
print(df2.unstack(level=1))

# Index setting and resetting
df2_flat = df2.reset_index(name='millions') # the type is DataFrame, and the common format of raw data
print(df2_flat)
df2 = df2_flat.set_index(['char', 'nums'])  # mostly raw data will be like this flat
                                            # and the convenient way is to turn it into MultiIndex Series
print(df2)

# Data aggregations
print(df1)
print(df1.mean(level='index1')) # the default axis is on index
print(df1.mean(level='index2'))
print(df1.mean(axis='columns', level='column1'))



#------------------------------------------------------------------------------
# Combining Datasets: Concat and Append
#------------------------------------------------------------------------------

def make_df(cols, ind):
    """Quickly make a DataFrame"""
    data = {c : [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)

print(make_df('ABC', range(3)))

# Recall concatenate in numpy
arr = rng.randint(0, 1000000, size=(3, 3))
arr1 = rng.randint(0, 1000000, size=(3, 3))
arr_cont = np.concatenate([arr, arr1], axis=1)
print(arr_cont)
arr_cont1 = np.concatenate([arr, arr1], axis=0)
print(arr_cont1)

# concate on Series
ser0 = pd.Series(rng.randint(0, 10, size=3), index=[1, 2, 3])
ser1 = pd.Series(rng.randint(0, 10, size=3), index=[4, 5, 6])
print(pd.concat([ser0, ser1]))

# concate on DataFrame
df0 = pd.DataFrame(rng.randint(0, 10, size=(3, 2)), index=[1, 2, 3], 
                columns=['col1', 'col2'])
df1 = pd.DataFrame(rng.randint(0, 10, size=(3, 2)), index=[4, 5, 6],
                   columns=['col2', 'col3'])
print(pd.concat([df0, df1]))    # the concatenation will allign the axis,
                                # therefore there will be some NaN data points
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index
print(pd.concat([x, y])) # preserves indices

try:
    pd.concat([x, y], verify_integrity=True)
except ValueError as e:
    print(e)
    
try:
    print(pd.concat([x, y], ignore_index=True)) # so we will be using the default index, the implicit one 
except ValueError as e:
    print(e)
    
print(x); print(y)
print(pd.concat([x, y], keys=['x', 'y'])) # assigning MultiIndex as its index

df2 = make_df('ABC', [1, 2])
df3 = make_df('BCD', [2, 3])
print(pd.concat([df2, df3], join='inner'))
print(pd.concat([df2, df3], join='outer'))
print(pd.concat([df2, df3], join_axes=[df2.columns]))



#------------------------------------------------------------------------------
# Combining Datasets: Merge and Join
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Categories of Joins

# One-to-one joins
# from book
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
print(df1); print(df2)
df3 = pd.merge(df1, df2)
print(df3) # merging process will allign the corresponding 'employee' column

# made up by myself
x = pd.DataFrame({'names' : ['sekar', 'saskia', 'arifa', 'afiffah'], 'numbers1' : rng.randint(0, 1000000, 4)})
print(x)
y = pd.DataFrame({'names' : ['arifa', 'sekar', 'saskia', 'afiffah'], 'numbers2' : rng.normal(size=4)})
print(y)
z = pd.DataFrame({'names' : ['arifa', 'sekar', 'saskia', 'afiffah'], 'type' : ['HS', 'College', 'College', 'HS']})
xz = pd.merge(z, x)
print(xz)
xy = pd.merge(x, y)
print(xy)
xyz = pd.merge(xz, xy)
print(xyz)
a = pd.DataFrame({'type' : ['HS', 'HS', 'College', 'College'], 'skills' : ['Dance', 'Sing', 'Analysis', 'Coding']})
print(a)
az = pd.merge(z, a)
print(az)
axyz = pd.merge(xyz, az)
print(axyz)

# Many-to-one joins
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
print(pd.merge(df3, df4))
# notice that the column group in df3 has duplicate data, 
# so the merging process is considering this duplicate data

# Many-to-many joins
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
print(df1); print(df5)
print(pd.merge(df1, df5))
# notice that in df5 column 'group' there are duplicate data points
# then the merging process will duplicate df1 column 'employee' to 
# its correspoding df5 based on column employee


#------------------------------------------------------------------------------
# Specification of the Merge Key

# tke on keyword
print(df1); print(df2); print(pd.merge(df1, df2, on='employee')) # matching with key 'employee'

# the left_on and right_on keywords
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
print(df3)
print(df1); print(df3); print(pd.merge(df1, df3, left_on='employee', right_on='name'))
# notice that the merge dataset is redundant because has two columns with the same datapoints
print(pd.merge(df1, df3, left_on='employee', right_on='name').drop('name', axis='columns'))
# the drop method will remove the intended column

# the lef_index and right_index keywords
df1a = df1.set_index('employee')    # index for df1 is employee
df2a = df2.set_index('employee')    # -------------------------
print(df1a); print(df2a)
print(pd.merge(df1a, df2a, left_index=True, right_index=True))
print(pd.merge(df1a, df2a, left_index=True, right_index=True)) # must pass both left_index and right_index keywords, otherwise will be error
print(pd.merge(df1a, df3, left_index=True, right_on='name'))
print(pd.merge(df3, df1a, right_index=True, left_on='name'))
print(pd.merge(df3, df1a, right_index=True, right_on='name')) # error
print(pd.merge(df3, df1a, left_index=False, right_on='name')) # error


#------------------------------------------------------------------------------
# Specifying Set Arithmetic for Joins

df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                    columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                    columns=['name', 'drink'])
print(df6); print(df7)
print(pd.merge(df6, df7)) # the default settings is find the intersection
print(pd.merge(df6, df7, how='inner')) # specifying the how keyword
print(pd.merge(df6, df7, how='outer')) # fing the union between two data
print(pd.merge(df6, df7, how='left')) # A-B
print(pd.merge(df6, df7, how='right')) # B-A
print(pd.merge(df7, df6, how='left'))


#------------------------------------------------------------------------------
# Overlapping Column Names: The suffixes Keyword

df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
print(df8); print(df9)
print(pd.merge(df8, df9, on='name'))
# notice that in df8 and df9 there are two columns with the same name
# therefore the merge process will reserved the columns by give it suffixes
print(pd.merge(df8, df9, right_on='name', left_on='name')) # redundant code
print(pd.merge(df8, df9, on='name', suffixes=['_L', '_R'])) # speciying the suffixes with list


#------------------------------------------------------------------------------
# Example: US States Data

pop = pd.read_csv('state-population.csv')
areas = pd.read_csv('state-areas.csv')
abbrevs= pd.read_csv('state-abbrevs.csv')

print(pop.head())
print(areas.head())
print(abbrevs.head())
print(len(pop[pop['state/region'].isnull() == True])) # no null
print(len(pop[pop['ages'].isnull() == True])) # no null
print(len(pop[pop['year'].isnull() == True])) # no null
print(len(pop[pop['population'].isnull() == True])) # no null

# creating function to check any null values
def check_null(data):
    col = list(data.columns)
    lst = []
    for i in col:
        lst.append(len(data[data[i].isnull() == True]))
    return lst

print(check_null(pop)) # there are null values in fourth column, which is population
print(check_null(areas)) # no nulls
print(check_null(abbrevs)) # no nulls

data1 = pd.merge(abbrevs, pop, left_index=True, left_on='abbreviation', right_on='state/region', how='outer')
print(data1.head())
data1 = data1.drop('state/region', axis=1)
print(check_null(data1))
data = pd.merge(data1, areas, left_on='state', right_on='state', how='outer')
print(check_null(data))
print(data[data.isnull()].head())

data = data.dropna(axis=0)
print(check_null(data))
# will lose several data because of false merging


#------------------------------------------------------------------------------
# Aggregation and Grouping
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Planets data
import seaborn as sns
planets = sns.load_dataset('planets')
print(planets[planets['method'] == 'Radial Velocity'].mean())
print(planets['method'].unique())
print(len(planets['method'].unique())) # there are ten methods of finding extrasolar planets in this dataset


#------------------------------------------------------------------------------
# Simple Aggregation in Pandas

print(planets.dropna().describe())
%timeit len(planets.dropna()) # faster than below method, why?
%timeit np.sum(planets.dropna().notnull())
print(planets.dropna().notnull().sum())

#------------------------------------------------------------------------------
# GroupBy: Split, Apply, Combine

# Split, apply, combine
df = pd.DataFrame({'key' : ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data' : rng.randint(0, 10, 6)})
print(df)
print(df.groupby('key'))    # returning a dataframe groupby object
print(df.groupby('key').mean())
print(df.groupby('key').sum())
print(df.groupby('key').median())

# Column indexing
print(planets.groupby('method')['orbital_period'].median())
print(planets.groupby('method')['orbital_period'].mean())
print(planets.groupby('method')['distance'].mean()) # note, still contain NaN
print(planets.dropna().groupby('method')['distance'].mean())    # weird result

# Iteration over groups
# groupby object is iterable
for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method, group.shape))

for (method, group) in planets.groupby('method'):
    print(method, group)

# Dispatch methods
print(planets.groupby('method')['year'].describe())
print(planets.groupby('method')['year'].describe().unstack())


# Aggregate, filter, transform, apply
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                    'data1': range(6),
                    'data2': rng.randint(0, 10, 6)},
                    columns = ['key', 'data1', 'data2'])
print(df)

# Aggregation
print(df.groupby('key').aggregate([min, np.median, max])) # this parameter in aggregate can take function or 
print(df.groupby('key').aggregate(['min', 'median', 'max'])) # string that represents the function
print(df.groupby('key').aggregate([np.mean, np.std, np.median]))
print(df.groupby('key').aggregate([np.sum, np.sin, np.log])) # error, must produce aggregated value
# abouve pass the parameter as list of aggregate operations
print(df.groupby('key').aggregate({'data1':np.mean, 'data2':np.median}))
# aboce, pass the parameter as dictionary

# Filtering
def filter_func(data):
    return data['data2'].std() > 4

print(df); print(df.groupby('key').std())
print(df.groupby('key').filter(filter_func))

# Transformation
print(df.groupby('key').transform(lambda x: x - x.mean()))
print(df.groupby('key').transform(lambda x: x - np.sin(x)))
print(df.groupby('key').transform(lambda x: x + x.std())) 
# center the data around the mean
# passing lamda expression

# The apply() method
# apply arbitrary function
def norm_by_data2(x):
    # x is a DataFrame of group values
    x['data1'] /= x['data2'].sum()
    return x

print(df); print(df.groupby('key').apply(norm_by_data2))

#----create new dataset for exploration
data = pd.DataFrame({'keys' : ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'], 
                     'million' : rng.randint(0, 1000000, 15),
                     'normal' : rng.normal(size=15),
                     'negative' : rng.randint(-1000000, 0, 15)})
print(data)

# Specifying split key
# A list, array, series or index providing the grouping keys
L = [0, 1, 0, 1, 2, 0]
print(df); print(df.groupby(L).sum()) # we are using split key 
print(df.groupby('key').sum())

# A dictionary or series mapping index to group
df2 = df.set_index('key') # setting colum key as new index, the explicit one
mapping = {'A':'vowel', 'B':'consonant', 'C':'consonant'}
print(df2.groupby(mapping).sum())

# Any python function
print(df2); print(df2.groupby(str.lower).sum())

# A list of valid keys
print(df2.groupby([str.lower, mapping]).mean())
print(df2.groupby([str.lower, mapping]).mean().index)

# Grouping example
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
print(planets.groupby(['method', decade])['number'].sum().unstack().fillna(0))
