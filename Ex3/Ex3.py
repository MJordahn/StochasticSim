# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:36:03 2019

@author: Andreas
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import t


###########################################################
# Basic variables
n = 10000
U1 = np.random.rand(n)
U2 = np.random.rand(n)

###########################################################
# We generate values from distributions distribution:

# exponential:
lamb = 1
X1 = np.log(U1)*(-1.0/lamb)
plt.hist(X1, 20)

# standard normal:
Z1 = np.sqrt(-2*np.log(U1)) * np.cos(2*pi*U2)
plt.hist(Z1, 20)

# Pareto with beta=1 and k=2.05, 2.5, 3, 4.
def Pareto(beta, k, U1):
    P1 = beta*(np.power(U1, -(1.0/k)))
    plt.hist(P1, 40)
    return(P1)

P1=Pareto(1, 2.05, U1)
P2=Pareto(1, 2.5, U1)
P3=Pareto(1, 3.0, U1)
P4=Pareto(1, 4.0, U1)

#Theoretical distribution
P1_T = np.random.pareto(2.05, n)+np.ones(n)
P2_T = np.random.pareto(2.5, n)+np.ones(n)
P3_T = np.random.pareto(3.0, n)+np.ones(n)
P4_T = np.random.pareto(4.0, n)+np.ones(n)

# Theoretical mean and variance:

def MeanVarTheo(beta, k):
    m = beta*(k/(k-1.0))
    v = beta**2*(k/((k-1)**2*(k-2)))
    return(m, v)


MeanVarTheo(1.0, 2.05)
MeanVarTheo(1.0, 4.0)

np.mean(P4) # Empirical mean fits well with theoretical mean.
np.std(P4)**2 # Empirical variance of pareto distribution.

#############################################################
# We generate   100 95% confidence intervals:

U = np.random.rand(100*10)
U_helper = np.random.rand(100*10)
Z = np.sqrt(-2*np.log(U)) * np.cos(2*pi*U_helper)

plt.hist(Z)
# mean estimate using 10 observations:

mu_list = np.zeros((100, 3))
var_list = np.zeros((100, 3))

for i in range(100):
    Z_list = Z[10*i:(10*i)+10]
    mu = 0.1*sum(Z_list)
    var = 1.0/(10-1)*sum((Z_list-mu*np.ones(10))**2)
    s = math.sqrt(var/10)
    
    mu_lower = mu - 1.96*s
    mu_upper = mu + 1.96*s
    
    var_upper= ((10.0-1)*var)/(chi2.ppf(0.025, 9))
    var_lower= ((10.0-1)*var)/(chi2.ppf(0.975, 9))
    
    mu_list[i, :] = [mu, mu_lower, mu_upper]
    var_list[i, :] = [var, var_lower, var_upper]
    
    #print("Mean confidence interval: [" + str(mu_lower) + ", " + str(mu_upper) + "]")
    #print("Var confidence interval: [" + str(var_lower) + ", " + str(var_upper) + "]")

# Mean estimates with lower and upper bound;
mu_list
# Variance esstiamte with lower and upper bound:
var_list

# The mean estimates varies a lot.
# So does the variance estimates.
# This is due to the fact, that only 10 observations are used.
# Note that:
print("Average Mean estimate: " + str(sum(mu_list[:,0])/100))
print("Average Variance estimate: " + str(sum(var_list[:,0])/100))
# are both close to the true value.

