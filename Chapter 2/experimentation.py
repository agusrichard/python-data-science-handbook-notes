# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:44:49 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
# Experimentation
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.RandomState(21)

#-----------------------------------------------------------------------------
# Experiment on several distribution

x1 = rng.normal(loc=0, scale=5, size=1000)
x2 = rng.binomial(1000, 0.5, size=1000)
x3 = rng.chisquare(21, size=1000) 
x4 = rng.gamma(0.5, size=1000) 
x5 = rng.poisson(5, size=1000)
x6 = rng.pareto(21, size=1000)

# Testing without loop
f, ax = plt.subplots(2, 3, figsize=(10, 10))
sns.distplot(x1,color='blue', ax=ax[0, 0], label='normal')
sns.distplot(x2,color='red', ax=ax[0, 1], label='binomial')
sns.distplot(x3,color='green', ax=ax[0, 2], label='chisquare')
sns.distplot(x4,color='magenta', ax=ax[1, 0], label='gamma')
sns.distplot(x5, color='black', ax=ax[1, 1], label='poisson')
sns.distplot(x6, color='yellow', ax=ax[1, 2], label='pareto')
plt.legend()
plt.show()


#------------------------------------------------------------------------------
# Experimentation on sampling distribution

X = rng.randint(-100, 100, size=(1000, 30))
col = np.hsplit(X, list(range(1, 30)))
f, ax = plt.subplots(5, 6, figsize=(10, 10))
for i in range(5):
    for j in range(6):
        sns.distplot(col[i+j], bins=30, ax=ax[i, j])
plt.show()

# Now take the mean from all row
mean_array = X.mean(axis=1)
sns.distplot(mean_array, color='blue', bins=30)




