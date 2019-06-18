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

def probabilities_geometric(array, n):
    histogram = [0]*26
    sum = 0
    for i, k in enumerate(array):
        if k>25:
            sum = sum + 1
        else:
            histogram[int(k)] = histogram[int(k)] + 1
    for k, prob in enumerate(histogram):
        histogram[int(k)] = histogram[int(k)]/n
    histogram.append(sum/n)
    return histogram

def probabilities_6point(array, n):
    histogram = [0]*6
    sum = 0
    for i, k in enumerate(array):
        histogram[int(k)-1] = histogram[int(k)-1] + 1
    for k, prob in enumerate(histogram):
        histogram[int(k)] = histogram[int(k)]/n
    return histogram

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
CorrectX1 = [0]
X1 = probabilities_geometric(X1,n)
sum = 0.0
for i in range(1, 26):
    value = (1-p)**(i-1)*p
    CorrectX1.append(value)
    sum = sum + value
CorrectX1.append(1-sum)

# plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], X1)
# plt.show()
# plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], CorrectX1)
# plt.show()
chi2sum = 0
zipped = zip(X1, CorrectX1)
for element in zipped:
    if element[1] != 0:
        chi2sum = chi2sum + (element[0]-element[1])**2/element[1]

print("P value for geometric test: " + str(1.0-chi2.cdf(chi2sum, 25)))



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
X2 = probabilities_6point(X2,n)
# plt.bar([1-.125, 2-.125, 3-.125, 4-.125, 5-.125, 6-.125], X2, width=0.25, label="generated")
# plt.bar([1+.125, 2+.125, 3+.125, 4+.125, 5+.125, 6+.125], P_i, color="r", width=0.25, label="theoretical")
# plt.xlabel("Point class")
# plt.ylabel("Probability")
# plt.legend()
# plt.show()

chi2sum = 0
zipped = zip(X2, P_i)
for element in zipped:
    if element[1] != 0:
        chi2sum = chi2sum + (element[0]-element[1])**2/element[1]

print("P value for rejection method: " + str(1.0-chi2.cdf(chi2sum, 5)))

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

X3 = probabilities_6point(X3,len(X3))
plt.bar([1-.125, 2-.125, 3-.125, 4-.125, 5-.125, 6-.125], X3, width=0.25, label="generated")
plt.bar([1+.125, 2+.125, 3+.125, 4+.125, 5+.125, 6+.125], P_i, color="r", width=0.25, label="theoretical")
plt.xlabel("Point class")
plt.ylabel("Probability")
plt.legend()
plt.show()

chi2sum = 0
zipped = zip(X3, P_i)
for element in zipped:
    if element[1] != 0:
        chi2sum = chi2sum + (element[0]-element[1])**2/element[1]

print("P value for crude method: " + str(1.0-chi2.cdf(chi2sum, 5)))

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

for i, u in enumerate(U2):
    if u <= F[int(I_list[i])]:
        X3 = int(I_list[i])
    else:
        X3 = L[int(I_list[i])]
plt.hist(X3)
plt.show()



### To be continued ######
