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

def setup_Alias(F):
    smaller = []
    larger = []
    for i, prob in enumerate(F):
        if prob < 1.0:
            smaller.append(i)
        else:
            larger.append(i)
    return smaller, larger

###########################################################
#Geometric distribution:

def f_n(n,p):
    r = np.power((1-p), range(n) - np.ones(n))*p
    return r

X1 = np.floor(np.log(U)*(1.0/np.log(1-p))) + np.ones(n)

#plt.hist(X1, 20)

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

#plt.hist(X3)
# Alias sampling:
U2 = np.random.rand(n)
X3 = np.zeros(n)
I_list=np.floor(6*U)
L = list(range(1,7))
F = [p * 6 for p in P_i]
S, G = setup_Alias(F)
while len(S)!=0:
    k = G[0]
    j = S[0]
    L[j] = k
    F[k] = F[k] - (1-F[j])
    del S[0]
    if F[k] < 1:
        del G[0]
        S.append(k)

print(F)
print(L)
print(I_list)
for i, u in enumerate(U2):
    if u <= F[int(I_list[i])]:
        X3 = int(I_list[i])
    else:
        X3 = L[int(I_list[i])]
plt.hist(X3)
plt.show()



### To be continued ######
