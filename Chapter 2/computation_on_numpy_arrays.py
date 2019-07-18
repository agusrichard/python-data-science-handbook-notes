# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 05:00:24 2019

@author: Agus Richard Lubis
"""
#------------------------------------------------------------------------------
# Computation on Numpy Arrays:Universal Functions
#------------------------------------------------------------------------------

import numpy as np

#------------------------------------------------------------------------------
# The slowness of loops
np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty((len(values)))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output

values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)

big_array = np.random.randint(1, 100, size=100000)
%timeit compute_reciprocals(big_array)


#------------------------------------------------------------------------------
# Introducing UFuncs

%timeit (1.0/ big_array)
# vector operations on numpy are using Ufuncs, to make it more flexible and fast

# ufuncs on one-dimensional array
np.arange(5) / np.arange(1, 6)
np.arange(1, 6) / np.arange(6, 11) 
print(1/6)

# ufuncs on multidimensional array
x = np.arange(9).reshape((3, 3))
print(2**x)


#------------------------------------------------------------------------------
# Exploring Numpy's UFuncs

# Array aritmetic
x = np.arange(4)
print("x =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2) # floor division
print("-x = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2 = ", x % 2)
print(-(0.5*x + 1) ** 2)
print(np.add(x, 2)) # we are using

# Absolute value
x = np.array([-2, -1, 0, 1, 2])
print(abs(x))
print(np.abs(x))
x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
np.abs(x) # returning the magnitude of complex number

# Trigonometric functions
theta = np.linspace(0, np.pi, 3)
print("theta = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))
x = [-1, 0, 1]
print("x = ", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))

# Exponents and logarithms
x = [1, 2, 3]
print("x =", x)
print("e^x =", np.exp(x))
print("2^x =", np.exp2(x))
print("3^x =", np.power(3, x))
x = [1, 2, 4, 10]
print("x =", x)
print("ln(x) =", np.log(x)) # base euler number
print("log2(x) =", np.log2(x))
print("log10(x) =", np.log10(x))
x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))

# Specialized ufuncs
from scipy import special
# Gamma functions(generalized factorials) and related functions
x = [1, 5, 10]
print("gamma(x) =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2) =", special.beta(x, 2))
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x) =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))


#------------------------------------------------------------------------------
# Advanced UFunc Features
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y) # after doing calculation, the result stored in y
y = np.zeros(10)
np.power(2, x, out=y[::2]) # Changing the y, because we are using the array view
print(y)
# if we are using the normal way, for small data it doens't make any difference
# but for large number of data points, it is the inverse.

# Aggregates 
x = np.arange(1, 6)
np.add.reduce(x) # works like sumation 
np.multiply.reduce(x) # works like factorial 
np.add.accumulate(x) # accumulation, surely
np.multiply.accumulate(x)

# Outer products
x = np.arange(1, 6)
np.multiply.outer(x, x) # making multiplication table
