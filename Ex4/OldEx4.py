import sys
import numpy as np
import math
from scipy.stats import t
import random

def simulation(_lambda, _service_time, runtimes, arrival, service):
    theta_hats = []
    for i in range(0, runtimes):
        offered_traffic = int(sys.argv[1])
        service_unit = [0]*int(sys.argv[2])

        arrival_list = []
        actual_samples = 0

        while actual_samples < offered_traffic:
            if arrival == "poisson":
                new_sample = np.random.poisson(lam=_lambda)
            elif arrival == "erlang":
                shape = 5
                new_sample = math.floor(np.random.gamma(shape = shape, scale=1/shape))
            elif arrival == "hyperexponential":
                rand = random.random()
                if rand < 0.8:
                    new_sample = math.floor(np.random.exponential(scale=0.8333))
                else:
                    new_sample = math.floor(np.random.exponential(scale=5))
            if actual_samples + new_sample > int(offered_traffic):
                new_sample = int(offered_traffic)-actual_samples
            actual_samples = actual_samples + new_sample
            arrival_list.append(new_sample)

        blocked_total = 0
        blocked_list = []
        for current_time, arriving_customers in enumerate(arrival_list):

            attempted_allocated = 0
            blocked = 0
            while attempted_allocated < arriving_customers:
                for k, end_time in enumerate(service_unit):
                    if end_time <= current_time:
                        if service == "exponential":
                            new_end_time = current_time + np.random.exponential(scale=_service_time)
                        service_unit[k] = new_end_time
                        break
                    elif k == len(service_unit)-1:
                        blocked = blocked + 1
                attempted_allocated = attempted_allocated + 1
            blocked_total = blocked_total+blocked
        theta_hat = float(blocked_total)/float(offered_traffic)
        theta_hats.append(theta_hat)
    theta_bar = 0
    for theta in theta_hats:
        theta_bar = theta + theta_bar
    theta_bar = theta_bar/runtimes
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
    return theta_bar, theoretical_fraction, std, lower_bound_confidence, upper_bound_confidence, len(arrival_list)

def main():
    _lambda = 1
    _service_time = 8
    runtimes = 10

    theta_bar, theoretical_fraction, std, lower_bound_confidence, upper_bound_confidence, arrival_list =  simulation(_lambda, _service_time, runtimes, "poisson", "exponential")

    print("-------------------Poisson arrival and exponential service-------------------")
    print("Estimator: " + str(theta_bar))
    print("Theoretical fraction: " + str(theoretical_fraction))
    print("Standard deviation: " + str(std))
    print("Lower bound: " + str(lower_bound_confidence))
    print("Upper bound: " + str(upper_bound_confidence))
    print("Time elapsed in simulation: " + str(arrival_list))
    print("-----------------------------------------------------------------------------\n\n")

    theta_bar, theoretical_fraction, std, lower_bound_confidence, upper_bound_confidence, arrival_list =  simulation(_lambda, _service_time, runtimes, "erlang", "exponential")

    print("-------------------Erlang arrival and exponential service--------------------")
    print("Estimator: " + str(theta_bar))
    print("Standard deviation: " + str(std))
    print("Lower bound: " + str(lower_bound_confidence))
    print("Upper bound: " + str(upper_bound_confidence))
    print("Time elapsed in simulation: " + str(arrival_list))
    print("-----------------------------------------------------------------------------\n\n")

    theta_bar, theoretical_fraction, std, lower_bound_confidence, upper_bound_confidence, arrival_list =  simulation(_lambda, _service_time, runtimes, "hyperexponential", "exponential")

    print("-------------------Hyperexponential arrival and exponential service--------------------")
    print("Estimator: " + str(theta_bar))
    print("Standard deviation: " + str(std))
    print("Lower bound: " + str(lower_bound_confidence))
    print("Upper bound: " + str(upper_bound_confidence))
    print("Time elapsed in simulation: " + str(arrival_list))
    print("---------------------------------------------------------------------------------------\n\n")



if __name__ == "__main__":
    main()
