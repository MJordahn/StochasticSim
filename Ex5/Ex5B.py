# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:58:32 2019

@author: Andreas
"""

import sys
import numpy as np
import math
from scipy.stats import t
import random
import bisect

def simulation(_lambda, _service_time, runtimes, arrival, service):
    theta_hats = []
    _service_time_estimates = []
    for i in range(0, runtimes):
        theta_hat = 0
        offered_traffic = int(sys.argv[1])
        service_unit = [False]*int(sys.argv[2])
        event_list = []
        new_sample = 0
        if arrival == "poisson":
            new_sample = np.random.exponential(scale=_lambda)
        elif arrival == "erlang":
            shape = 3
            new_sample = np.random.gamma(shape = shape, scale=1/shape)
        elif arrival == "hyperexponential":
            rand = random.random()
            if rand < 0.8:
                new_sample = np.random.exponential(scale=1/0.8333)
            else:
                new_sample = np.random.exponential(scale=1/5)

        arrival_time = new_sample
        event_list.append((arrival_time, 'a', None))

        customers = 1
        blocked = 0
        service_times = []

        while customers <= offered_traffic:
            next_event = event_list.pop(0)
            time = next_event[0]
            if next_event[1] == "a":

                for i, occupied in enumerate(service_unit):
                    if occupied == False:
                        if service == "exponential":
                            r = np.random.exponential(scale=_service_time)
                            service_times.append(r)
                            new_end_time = time + r
                            bisect.insort_left(event_list,(new_end_time, 'd', i))
                            service_unit[i] = True
                            break
                        elif service == "constant":
                            new_end_time = time + _service_time
                            bisect.insort_left(event_list,(new_end_time, 'd', i))
                            service_unit[i] = True
                            break
                    elif i == len(service_unit)-1:
                        blocked = blocked + 1
                new_sample = 0
                if arrival == "poisson":
                    new_sample = np.random.exponential(scale=_lambda)
                elif arrival == "erlang":
                    shape = 3
                    new_sample = np.random.gamma(shape = shape, scale=1/shape)
                elif arrival == "hyperexponential":
                    rand = random.random()
                    if rand < 0.8:
                        new_sample = np.random.exponential(scale=1/0.8333)
                    else:
                        new_sample = np.random.exponential(scale=1/5)
                new_arrival_time = time + new_sample
                bisect.insort_right(event_list, (new_arrival_time, 'a', None))
                customers=customers+1
            elif next_event[1] == 'd':
                service_unit[next_event[2]] = False

        avg = np.average(service_times)
        _service_time_estimates.append(avg)

        theta_hat = float(blocked)/float(offered_traffic)
        theta_hats.append(theta_hat)
        
    theta_bar = np.average(theta_hats)
    
    sum_hats = 0
    for theta in theta_hats:
        sum_hats = sum_hats + theta**2
    std = math.sqrt(1/(runtimes-1)*(sum_hats-theta_bar**2*runtimes))
    sqrt_n = math.sqrt(int(runtimes))
    lower_bound_confidence = theta_bar + std/sqrt_n*t.ppf(0.025, df=runtimes-1)*(runtimes-1)
    upper_bound_confidence = theta_bar + std/sqrt_n*t.ppf(0.975, df=runtimes-1)*(runtimes-1)
    sum_theoretical = 0
    A = _lambda*_service_time
    for i in range (0, int(sys.argv[2])+1):
        sum_theoretical = sum_theoretical + A**i/math.factorial(i)
    theoretical_fraction = A**int(sys.argv[2])/math.factorial(int(sys.argv[2]))/sum_theoretical
    
    # Utilizing control variates:
    n=runtimes
    m=offered_traffic
    U=_service_time_estimates
    U_mu = _service_time
    U_var = float(_service_time)**2/m

    cov = (1.0/(n-1))*np.matmul(theta_hats-theta_bar*np.ones(n), U-np.average(U)*np.ones(n))
    c_opt = -cov/U_var

    theta_hats_c = theta_hats + c_opt*(U-U_mu*np.ones(n))
    theta_bar_c = np.average(theta_hats_c)

    std_c = math.sqrt((1.0/(n-1)) * (sum(np.array(theta_hats_c)**2) - n*theta_bar_c**2))

    sqrt_n = math.sqrt(int(n))
    lower_bound_c = theta_bar_c + std_c/sqrt_n*t.ppf(0.025, df=n-1)
    upper_bound_c = theta_bar_c + std_c/sqrt_n*t.ppf(0.975, df=n-1)

    #U_var = (1.0/(n-1)) * sum(np.array(U)**2)-n*U_mu**2
    # U_var = (1.0/10000)*(8.0**2)
    #
    # cov = (1.0/(n-1))*(np.matmul(theta_hats-theta_bar*np.ones(n), U-U_mu*np.ones(n)))
    # c_opt = -cov/(U_var)
    # theta_hats_c = theta_hats + c_opt*(U-U_mu*np.ones(n))
    # theta_bar_c = np.average(theta_hats_c)
    # sqrt_n = math.sqrt(int(n))
    #
    # std_c = math.sqrt(1.0/(n-1.0) * (sum(theta_hats_c**2.0)-n*theta_bar_c**2))
    # lower_bound_c = theta_bar_c + std_c/sqrt_n*t.ppf(0.025, df=n-1)*(n-1)
    # upper_bound_c = theta_bar_c + std_c/sqrt_n*t.ppf(0.975, df=n-1)*(n-1)
    
    return theta_bar_c, theoretical_fraction, std_c, std, lower_bound_c, upper_bound_c

def main():
    _lambda = 1
    _service_time = 8
    runtimes = 10

    theta_bar, theoretical_fraction, std_c, std, lower_bound_confidence, upper_bound_confidence =  simulation(_lambda, _service_time, runtimes, "poisson", "exponential")

    print("-------------------Poisson arrival and exponential service-------------------")
    print("Estimator: " + str(theta_bar))
    print("Theoretical fraction: " + str(theoretical_fraction))
    print("Standard deviation control: " + str(std_c))
    print("Standard deviation: " + str(std))
    print("Lower bound: " + str(lower_bound_confidence))
    print("Upper bound: " + str(upper_bound_confidence))
    print("-----------------------------------------------------------------------------\n\n")

    #Variance 1 and mean 1

    theta_bar, theoretical_fraction, std_c, std, lower_bound_confidence, upper_bound_confidence =  simulation(_lambda, _service_time, runtimes, "erlang", "exponential")

    print("-------------------Erlang arrival and exponential service--------------------")
    print("Estimator: " + str(theta_bar))
    print("Standard deviation control: " + str(std_c))
    print("Standard deviation: " + str(std))
    print("Lower bound: " + str(lower_bound_confidence))
    print("Upper bound: " + str(upper_bound_confidence))
    print("-----------------------------------------------------------------------------\n\n")

    #Variance 1/5 and mean 1

    theta_bar, theoretical_fraction, std_c, std, lower_bound_confidence, upper_bound_confidence =  simulation(_lambda, _service_time, runtimes, "hyperexponential", "exponential")

    print("-------------------Hyperexponential arrival and exponential service--------------------")
    print("Estimator: " + str(theta_bar))
    print("Standard deviation control: " + str(std_c))
    print("Standard deviation: " + str(std))
    print("Lower bound: " + str(lower_bound_confidence))
    print("Upper bound: " + str(upper_bound_confidence))
    print("---------------------------------------------------------------------------------------\n\n")

    #Variance 1.32 and mean 1

if __name__ == "__main__":
    main()