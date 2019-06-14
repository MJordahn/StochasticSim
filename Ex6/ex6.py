import sys
import numpy as np
import math
from scipy.stats import t
import random
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import bisect

def probabilities(array, n):
    histogram = [0]*11
    print(array)
    for i, k in enumerate(array):
        histogram[k] = histogram[k] + 1
    for k, prob in enumerate(histogram):
        histogram[k] = histogram[k]/n
    print(histogram)
    return histogram


def main():
    _lambda = 1
    _mean_service_time = 8
    A = 1*8
    n = 10000
    states = [1]
    for i in range(0, n-1):
        accept = False
        sample = np.random.choice(a=[-1, 0, 1])
        if states[-1] + sample > 10:
            y = 0
        elif states[-1] + sample < 0:
            y = 10
        else:
            y = states[-1] + sample
        g_x = A**states[-1]/math.factorial(states[-1])
        g_y = A**y/math.factorial(y)
        rnd = random.random()
        if g_y/g_x > rnd:
            accept = True
        if accept:
            states.append(y)
        else:
            states.append(states[-1])

    plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], probabilities(states,n))
    plt.show()
    sampled_walk = [1]
    #for i in range(0, n-1):
    #    sampled_walk.append(sampled_walk[-1] + np.random.poisson(lam=_lambda))
    print(states)
    #print(sampled_walk)
    print(chisquare(states, f_exp=sampled_walk))

if __name__ == "__main__":
    main()
