# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:16:40 2019

@author: Andreas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import norm

###########################################################
# Basic variables
n = 10000
p = 0.2
U = np.random.rand(n)

###########################################################
#Geometric distribution:

def f_n(n,p):
    r = np.power((1-p), range(n) - np.ones(n))*p
    return r

X1 = np.floor(np.log(U)*(1.0/np.log(1-p))) + np.ones(n)

plt.hist(X1, 20)

###########################################################
# Simulation of a 6-point distribution:

P_i = np.array([7.0/48, 5.0/48, 1.0/8.0, 1.0/16, 1.0/4, 5.0/16])

# Crude method:
X2 = np.zeros(n)
cum_dist = np.cumsum(P_i)

for i in range(n):
    t1 = U[i]
    for j in range(len(cum_dist)):
        if (cum_dist[j] >= U[i]):
            X2[i] = j+1
            break
plt.hist(X2)

# Rejection sampling. Note that not 10000 samples are made.:
c= 0.33
U2 = np.random.rand(n)
X3 = np.zeros(0)

I_list=1+np.floor(6*U)
for i in range(len(U)):
    I = int(I_list[i])
    p = P_i[I-1]
    if (U2[i]< p/c):
        X3 = np.append(X3,I)
        
plt.hist(X3)

# Alias sampling:
U2 = np.random.rand(n)
X3 = np.zeros(n)
I_list=1+np.floor(6*U)


### To be continued ######