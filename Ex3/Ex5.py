# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:36:57 2019

@author: Andreas
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import t

###############################################################

# We estimate the integral showed using crude monte carlo simulation:
n=100
U = np.random.rand(n)

X = math.e**U
X_mu = np.average(X)
X_var = 1.0/(n-1)*sum((X-X_mu)**2)

s = math.sqrt(X_var/n)
mu_upper = X_mu + t.ppf(0.975, n-1)*s
mu_lower = X_mu + t.ppf(0.025, n-1)*s

print("-----------Using crude monte carlo-----------")
print("The mean is estimated to: " + str(X_mu))
print("The var is estimated to: " + str(X_mu))
print("Confidence interval: [" + str(mu_lower) + ", " + str(mu_upper) +"]")
print("Theoretical value: " + str(math.e-1))
print(" ")

# We estimate the integral showed using the antithetic variable simulation:
Y = (math.e**U + math.e**(1-U))/2.0
Y_mu = np.average(Y)
Y_var = 1.0/(n-1)*sum((Y-Y_mu)**2)

s = math.sqrt(Y_var/n)
mu_upper = Y_mu + t.ppf(0.975, n-1)*s
mu_lower = Y_mu + t.ppf(0.025, n-1)*s

print("-----------Using antithetic variable-----------")
print("The mean is estimated to: " + str(Y_mu))
print("The var is estimated to: " + str(Y_var))
print("Confidence interval: [" + str(mu_lower) + ", " + str(mu_upper) +"]")
print("Theoretical value: " + str(math.e-1))
print(" ")

# We estimate the integral using  a control variable:
c_opt = -0.14086*12 #see slides.
X = math.e**U
Z = X + c_opt*(U-0.5)
Z_mu = np.average(Z)
Z_var = 0.0039 #see slides

s = math.sqrt(Z_var/n)
mu_upper = Z_mu + t.ppf(0.975, n-1)*s
mu_lower = Z_mu + t.ppf(0.025, n-1)*s

print("-----------Using control variates-----------")
print("The mean is estimated to: " + str(Z_mu))
print("The var is estimated to: " + str(Z_var))
print("Confidence interval: [" + str(mu_lower) + ", " + str(mu_upper) +"]")
print("Theoretical value: " + str(math.e-1))
print(" ")

# We estimate the integral using stratified sampling:
U1 = U/10.0
strata = 10.0
W = np.zeros(int(strata))
for i in range(int(strata)):
    for j in range(int(n/strata)):
        W[i] += math.e**(i/10.0 + U1[i*10+j])/10.0
        
W_mu = np.average(W)
W_var = 1.0/((n/strata)-1)*sum((W-W_mu)**2)

s = math.sqrt(W_var/(n/strata))
mu_upper = W_mu + t.ppf(0.975, (n/strata)-1)*s
mu_lower = W_mu + t.ppf(0.025, (n/strata)-1)*s

print("-----------Using stratified sampling-----------")
print("The mean is estimated to: " + str(W_mu))
print("The var is estimated to: " + str(W_var))
print("Confidence interval: [" + str(mu_lower) + ", " + str(mu_upper) +"]")
print("Theoretical value: " + str(math.e-1))
print(" ")








