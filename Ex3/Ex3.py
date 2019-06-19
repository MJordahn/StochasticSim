# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:36:03 2019

@author: Andreas
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
from scipy.stats import expon
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import pareto


###########################################################
# Basic variables
n = 10000
U1 = np.random.rand(n)
U2 = np.random.rand(n)


def countValues(arr, l):
    hist = np.zeros(l+1)
    for i in range(len(arr)):
        id = int(arr[i])
        if id >l:
            hist[l] += 1
        else:
            hist[id] += 1
    return hist

###########################################################
# We generate values from different distribution:

# exponential:
lamb = 1
X1 = np.log(U1)*(-1.0/lamb)
X1_t = np.random.exponential(scale=1, size = 10000)

plt.hist(X1)
plt.show()
plt.hist(X1_t)
plt.show()

# Kolomogorov test:
emp_dist_x = np.sort(X1)
emp_dist = np.array((range(10001)))[1:]/10000
true_dist = []
for i, item in enumerate(emp_dist_x):
    true_dist.append(expon.cdf(item))

#plt.plot(emp_dist_x, emp_dist)
#plt.plot(emp_dist_x, true_dist)
#plt.show()

D_n = np.max(np.abs(emp_dist-true_dist))
Adjust_D_n = (math.sqrt(10000.0) + 0.12 + 0.11*math.sqrt(10000.0))*D_n
print("-------- Exponential dist tests ---------")
print("Kolomogorov statistic for exponential distribution test: " + str(Adjust_D_n))
# Hypothesis is accepted.

#A chisquare test is performed:
#v1_s = countValues(X1, 7)/10000
#v1_t = [expon.cdf(1, scale=1)]
#for i in range(6):
#    prev = v1_t[i]
#    new = expon.cdf(i+2, scale=1)
#    v1_t.append(new)

#chi2sum = 0
#zipped = zip(v1_s, v1_t)
#for element in zipped:
#    if element[1] != 0:
#        chi2sum = chi2sum + (element[0]-element[1])**2/element[1]

#print("P value for exponential chi2 test: " + str(1.0-chi2.cdf(chi2sum, 7)))

n = 10000
U1 = np.random.rand(n)
U2 = np.random.rand(n)

# standard normal with Box Muller:
Z1 = np.sqrt(-2*np.log(U1)) * np.cos(2*pi*U2)
Z1_t = np.random.randn(10000)

plt.hist(Z1)
plt.show()
plt.hist(Z1_t)
plt.show()

#Kolomogoro test:
emp_dist_x = np.sort(Z1)
emp_dist = np.array((range(10001)))[1:]/10000
true_dist = []
for i, item in enumerate(emp_dist_x):
    true_dist.append(norm.cdf(item))

#plt.plot(emp_dist_x, emp_dist)
#plt.plot(emp_dist_x, true_dist)
#plt.show()

D_n = np.max(np.abs(emp_dist-true_dist))
Adjust_D_n = (math.sqrt(10000.0) + 0.12 + 0.11*math.sqrt(10000.0))*D_n
print("-------- Normal dist tests ---------")
print("Kolomogorov statistic for normal distribution test: " + str(Adjust_D_n))

# Pareto with beta=1 and k=2.05, 2.5, 3, 4.
n = 10000
U1 = np.random.rand(n)
U2 = np.random.rand(n)
U3 = np.random.rand(n)
U4 = np.random.rand(n)

def Pareto(beta, k, U1):
    P1 = beta*(np.power(U1, -(1.0/k)))
    return(P1)

P1=Pareto(1, 2.05, U1)
P2=Pareto(1, 2.5, U2)
P3=Pareto(1, 3.0, U3)
P4=Pareto(1, 4.0, U4)


#Theoretical distribution
P1_T = np.random.pareto(2.05, n)+np.ones(n)
P2_T = np.random.pareto(2.5, n)+np.ones(n)
P3_T = np.random.pareto(3.0, n)+np.ones(n)
P4_T = np.random.pareto(4.0, n)+np.ones(n)

plt.hist(P4, 20)
plt.show()
plt.hist(P4_T, 20)
plt.show()

#Kolomogoro test:
def kolTestPareto(P, k):
    emp_dist_x = np.sort(P)
    emp_dist = np.array((range(10001)))[1:]/10000
    true_dist = []
    for i, item in enumerate(emp_dist_x):
        true_dist.append(pareto.cdf(item, b=k, loc=0))

    #plt.plot(emp_dist_x, emp_dist)
    #plt.plot(emp_dist_x, true_dist)
    #plt.show()

    D_n = np.max(np.abs(emp_dist-true_dist))
    Adjust_D_n = (math.sqrt(10000.0) + 0.12 + 0.11*math.sqrt(10000.0))*D_n
    return Adjust_D_n

print("-------- Pareto tests ---------")
print("Kolomogorov statistic for pareto distribution k=2.05 test: " + str(kolTestPareto(P1, 2.05)))
print("Kolomogorov statistic for pareto distribution k=2.5 test: " + str(kolTestPareto(P2, 2.5)))
print("Kolomogorov statistic for pareto distribution k=3.0 test: " + str(kolTestPareto(P3, 3.0)))
print("Kolomogorov statistic for pareto distribution k=4.0 test: " + str(kolTestPareto(P4, 4.0)))
print("-------------------------------")


# Theoretical mean and variance:

def MeanVarTheo(beta, k):
    m = beta*(k/(k-1.0))
    v = beta**2*(k/((k-1)**2*(k-2)))
    return(m, v)

print("Theoretical mean and variance, k=2.05: " + str(MeanVarTheo(1.0, 2.05)))
print("Theoretical mean and variance, k=2.5:  " + str(MeanVarTheo(1.0, 2.5)))
print("Theoretical mean and variance, k=3.0:  " + str(MeanVarTheo(1.0, 3.0)))
print("Theoretical mean and variance, k=4.0:  " + str(MeanVarTheo(1.0, 4.0)))
print("-------------------------------")
print("Empirical mean and variance, k=2.05: " + str((np.mean(P1), np.std(P1)**2)))
print("Empirical mean and variance, k=2.5:  " + str((np.mean(P2), np.std(P2)**2)))
print("Empirical mean and variance, k=3.0:  " + str((np.mean(P3), np.std(P3)**2)))
print("Empirical mean and variance, k=4.0:  " + str((np.mean(P4), np.std(P4)**2)))

#############################################################
# We generate   100 95% confidence intervals:

U = np.random.rand(100*10)
U_helper = np.random.rand(100*10)
Z = np.sqrt(-2*np.log(U)) * np.cos(2*pi*U_helper)

#plt.hist(Z)
#plt.show()
# mean estimate using 10 observations:

mu_list = np.zeros((100, 3))
var_list = np.zeros((100, 3))

for i in range(100):
    Z_list = Z[10*i:(10*i)+10]
    mu = 0.1*sum(Z_list)
    var = 1.0/(10-1)*sum((Z_list-mu*np.ones(10))**2)
    s = math.sqrt(var/10)
    
    mu_lower = mu + t.ppf(0.025, 9)*s
    mu_upper = mu - t.ppf(0.025, 9)*s
    
    var_upper= ((10.0-1)*var)/(chi2.ppf(0.025, 9))
    var_lower= ((10.0-1)*var)/(chi2.ppf(0.975, 9))
    
    mu_list[i, :] = [mu, mu_lower, mu_upper]
    var_list[i, :] = [var, var_lower, var_upper]
    
    #print("Mean confidence interval: [" + str(mu_lower) + ", " + str(mu_upper) + "]")
    #print("Var confidence interval: [" + str(var_lower) + ", " + str(var_upper) + "]")

#print(mu_list)
#print(var_list)

c_var = 0
c_mu = 0
for i in range(len(mu_list)):
    mu = mu_list[i]
    var = var_list[i]
    if mu[1] <= 0 and 0 <= mu[2]:
        c_mu += 1
    if var[1] <= 1 and 1 <= var[2]:
        c_var += 1
print("----------100 confidence intervals------------")
print("Count of mean outside confidence interval: " + str(float(c_var)))
print("Count of var outside confidence interval: " + str(float(c_mu)))
print("Min mean estimate: " + str(min(mu_list[:,0])))
print("Max mean estimate: " + str(max(mu_list[:,0])))
# The mean estimates varies a lot.
# So does the variance estimates.
# This is due to the fact, that only 10 observations are used.
# Note that:
print("Average Mean estimate: " + str(sum(mu_list[:,0])/100))
print("Average Variance estimate: " + str(sum(var_list[:,0])/100))
# are both close to the true value.

