# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 01:15:56 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
# Combining Datasets: Concat and Append
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd

# defining a function which creates DataFrame
def make_df(cols, ind):
    """Quickly make a DataFrame"""
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)

# example DataFrame
print(make_df('ABC', range(3)))

#------------------------------------------------------------------------------
# Recall: Concatenation of Numpy Arrays

x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
print(np.concatenate([x, y, z])) # remember it is concatenate it to the right as a whole list
x = [[1, 2],
     [3, 4]]
print(np.concatenate([x, x], axis=1)) # concatenate it by column
print(np.concatenate([x, x], axis=0))

#------------------------------------------------------------------------------
# Simple concatenation with pd.concat()

ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
print(ser1)
print(ser2)
print(pd.concat([ser1, ser2])) # axis=0 (default)
print(pd.concat([ser1, ser2], axis=1))
df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
print(df1); print(df2); print(pd.concat([df1, df2]))
df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
print(df3); print(df4); print(pd.concat([df3, df4], axis=1))

# Duplicate indices
# concatenation preserved indices
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
print(pd.concat([x, y], axis=1))
print(pd.concat([x, y], axis=0))
y.index = x.index # make dulplicate indices
print(pd.concat([x, y]))

# Catching the repeats as an error
# verify_integrity flag if sets to True will check for repeated indices and throw an error if it happens
try:
    pd.concat([x, y], verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)
    
# Ignoring the index
# the result of concatenation will ignore the real index, and use the make up one
# which is ordered integers
print(x); print(y); print(pd.concat([x, y], ignore_index=True))

# Adding MultiIndex keys
print(x); print(y); print(pd.concat([x, y], keys=['x', 'y']))

# Concatenation with joins
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
print(df5); print(df6); 
print(pd.concat([df5, df6]))
print(df5); print(df6)
print(pd.concat([df5, df6], join='inner')) # will print the intersection
print(pd.concat([df5, df6], join='outer')) # print the union
print(df5); print(df6);
print(pd.concat([df5, df6], join_axes=[df5.columns])) # join axes means that we are using columns of df5 and union the intersection

# The append method
# works the same way as pd.concate, we can use _.append() to
# works more efficiently
print(df1); print(df2); 
print(df1.append(df2)) # rather than pd.concate, we use df1.append(df2)
# append does not modify the original object
# it creates a new object (not a void method)
# if we want to append many DataFrames, it is better using pd.concate()
# and pass all object into it
