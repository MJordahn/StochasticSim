# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:36:03 2019

@author: Andreas
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import chi2
from scipy.stats import norm


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
P1_T = np.random.pareto(2.05, n)
P2_T = np.random.pareto(2.5, n)
P2_T = np.random.pareto(3.0, n)
P3_T = np.random.pareto(4.0, n)

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

