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
X_var = 1.0/(n-1)*(sum(X**2)-n*X_mu**2)

s = math.sqrt(X_var/n)
mu_upper = X_mu + t.ppf(0.975, n-1)*s
mu_lower = X_mu + t.ppf(0.025, n-1)*s

print("-----------Using crude monte carlo-----------")
print("The mean is estimated to: " + str(X_mu))
print("The var is estimated to: " + str(X_var))
print("Confidence interval: [" + str(mu_lower) + ", " + str(mu_upper) +"]")
print("Theoretical value: " + str(math.e-1))
print(" ")

# We estimate the integral showed using the antithetic variable simulation:
Y = (math.e**U + math.e**(1-U))/2.0
Y_mu = np.average(Y)
Y_var = 1.0/(n-1)*(sum(Y**2)-n*Y_mu**2)

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
strata = 10.0
n=10
m=int(n*strata)
U = np.random.rand(m)
U1 = U/strata

W = np.zeros(n)
for i in range(n):
    for j in range(int(strata)):
        W[i] += math.e**(j/strata + U1[int(i*strata+j)])/strata
        
W_mu = np.average(W)
W_var = 1.0/(n-1)*(sum(W**2)-n*W_mu**2)
#W_var = 1.0/(n-1)*sum((W-np.ones(n)*W_mu)**2)

s = math.sqrt(W_var/n)
mu_upper = W_mu + t.ppf(0.975, n-1)*s
mu_lower = W_mu + t.ppf(0.025, n-1)*s

print("-----------Using stratified sampling-----------")
print("The mean is estimated to: " + str(W_mu))
print("The var is estimated to: " + str(W_var))
print("Confidence interval: [" + str(mu_lower) + ", " + str(mu_upper) +"]")
print("Theoretical value: " + str(math.e-1))
print(" ")








