import sys
import numpy as np
import math
from scipy.stats import t
import random
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import scipy.io as sio

#####################################################3
# BOOTSTAP EXERCISE #
######################################################

#Ex 13 of "The bootstrapping technique for estimatin mean squared errors".

a = -5
b = 5
X = [56, 101, 78, 67, 93, 87, 64, 72, 80, 69]
n = len(X)
X.sort()
cum_sum = []
last = -1
id_cum_sum = -1

for i in range(n):
    if X[i] == last:
        cum_sum[id_cum_sum] += 1
    else:
        cum_sum.append(1)
        id_cum_sum += 1
    last = X[i]
cum_sum = np.cumsum(np.array(cum_sum)/float(n))

# Empirical distribution:
#plt.plot(X, cum_sum)
#plt.show()
#print(X)
#print(cum_sum)

##################################################
# Do bootstrapping:
N = 1000
mu = np.average(X)
ind_mat = np.random.choice(a=range(n), size=(N,n))
bs_X = np.zeros((N,n))
for i in range(N):
    for j in range(n):
        id = ind_mat[i, j]
        bs_X[i,j] = X[id]

p = 0
for i in range(N):
    Z = np.average(bs_X[i]) - mu
    if a < Z and Z < b:
        p +=1

print("Bootstrap number " + str(N))
print("Number of accepts: " + str(p))
print("Estimate of p: " + str(p/N))
print("-------------------------------------")

##########################################################
# Part 2:

def medCalculation(x_obs, r):
    n=len(x_obs)
    med = np.median(x_obs)
    ind_mat = np.random.choice(a=range(n), size=(r, n))
    bs_x = np.zeros((r, n))
    for i in range(r):
        for j in range(n):
            id = ind_mat[i, j]
            bs_x[i, j] = x_obs[id]
    s = sum((np.median(bs_x[i,:])-med*np.ones(n))**2)
    mse = s/float(r)
    return med, mse

r=100
med, mse = medCalculation(X, r)
print("r is set to: " + str(r))
print("The sample media is: " + str(med))
print("The sample variance is: " + str(mse))
print("-------------------------------------")

###############################################################
# We simulate N=200 Pareto distributed random variables:
r=100
n=200
beta = 1
k = 1.05
P = (np.random.pareto(k, n)+np.ones(n))*beta

med, mse = medCalculation(P, r)
mu = np.average(P)
mu_var = 1.0/(n-1)*sum((P-mu*np.ones(n))**2)
print("Median estimate and bootstrap variance of median: " + str((med, mse)))
print("Mean estimate and estiamted variance of mean: " + str((mu, mu_var)))
print("Precision of median: " + str(1.0/mse))
print("Precision of mean: " + str(1.0/mu_var))
# Observations: We see, that the precision of bootstrap variance of the median is much greater than the one of the variance.
# This is due to the fact, that the Pareto distribution does not have second moments for k<2.
# The sample mean is therefor not at very robust estimator for the mean, when working with the Pareto distribution.
