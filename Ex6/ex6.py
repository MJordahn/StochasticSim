import sys
import numpy as np
import math
from scipy.stats import t
import random
import bisect

def main():
    _lambda = 1
    _mean_service_time = 8
    A = 1*8
    n = 5
    states = [0]
    for i in range(0, n):
        accept = False
        sample = np.random.beta(a=0, b=1)
        y = states[-1] + sample
        g_x = A**states[-1]/math.factorial(states[-1])
        g_y = A**y/math.factorial(math.floor(y))
        if g_y >= g_x:
            accept = True
        else:
            rnd = random.random()
            if g_y/g_x > rnd:
                accept = True
        if accept:
            states.append(y)
        else:
            states.append(x)
    true_samples = np.random.poisson(lam=_lambda)
    print(states)

if __name__ == "__main__":
    main()
