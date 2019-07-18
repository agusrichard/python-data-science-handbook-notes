# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 06:23:42 2019

@author: Agus Richard Lubis
"""

#------------------------------------------------------------------------------
# Computation on Arrays: Broadcasting
#------------------------------------------------------------------------------

# Broadcasting is simply a set of rules for applying binary ufuncs on arrays
# of different sizes

import numpy as np

#------------------------------------------------------------------------------
# Introducing Broadcasting
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
print(a + b)
print(a + 5)
x = np.arange(9).reshape((3,3))
print(x + 5)

M = np.ones((3, 3))
print(M + a)

a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a)
print(b)
print(a + b) # it looks like the vector getting streched to accomodate the following operation


#------------------------------------------------------------------------------
# Rules of broadcasting

M = np.ones((2, 3))
a = np.arange(3)
print(M)
print(a)
print(M + a)

a = np.arange(3).reshape((3, 1))
b = np.arange(3)
print(a)
print(b)
print(a + b)

M = np.ones((3, 2))
a = np.arange(3)
print(M)
print(a)
print(M + a) # will raise a ValueError because of the incompatible operations


#------------------------------------------------------------------------------
# Broadcasting in Practice

# Centering an array

X = np.random.random((10, 3))
Xmean = X.mean(0)
print(Xmean)
X_centered = X - Xmean # all data points in X is subtracted by Xmean
print(X_centered)
print(X_centered.mean(0))


#------------------------------------------------------------------------------
# Plotting a two-dimensional function

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

%matplotlib inline
import matplotlib.pyplot as plt

plt.imshow(z, origin='lower', extent=[0, 5, 0, 5],
           cmap='viridis')
plt.colorbar();


# Knowledge testing
a = np.arange(3).reshape((3, 1))
b = np.arange(3)
X = np.random.random((10, 3 ))
print(X)
print(X.mean(axis=0))
print(X.mean(axis=1))
