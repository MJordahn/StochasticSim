import sys
import numpy as np
import math
from scipy.stats import t
import random
import bisect

def simulation(_lambda, _service_time, runtimes, arrival, service):
    theta_hats = []
    for i in range(0, runtimes):
        theta_hat = 0
        offered_traffic = int(sys.argv[1])
        service_unit = [False]*int(sys.argv[2])
        event_list = []
        new_sample = 0
        if arrival == "poisson":
            new_sample = np.random.poisson(lam=_lambda)
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

        while customers <= offered_traffic:
            next_event = event_list.pop(0)
            time = next_event[0]
            if next_event[1] == "a":

                for i, occupied in enumerate(service_unit):
                    if occupied == False:
                        if service == "exponential":
                            new_end_time = time + np.random.exponential(scale=_service_time)
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
                    new_sample = np.random.poisson(lam=_lambda)
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

        theta_hat = float(blocked)/float(offered_traffic)
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
    return theta_bar, theoretical_fraction, std, lower_bound_confidence, upper_bound_confidence

def main():
    _lambda = 1
    _service_time = 8
    runtimes = 10

    theta_bar, theoretical_fraction, std, lower_bound_confidence, upper_bound_confidence =  simulation(_lambda, _service_time, runtimes, "poisson", "exponential")

    print("-------------------Poisson arrival and exponential service-------------------")
    print("Estimator: " + str(theta_bar))
    print("Theoretical fraction: " + str(theoretical_fraction))
    print("Standard deviation: " + str(std))
    print("Lower bound: " + str(lower_bound_confidence))
    print("Upper bound: " + str(upper_bound_confidence))
    print("-----------------------------------------------------------------------------\n\n")

    theta_bar, theoretical_fraction, std, lower_bound_confidence, upper_bound_confidence =  simulation(_lambda, _service_time, runtimes, "poisson", "constant")

    print("-------------------Poisson arrival and constant service-------------------")
    print("Estimator: " + str(theta_bar))
    print("Theoretical fraction: " + str(theoretical_fraction))
    print("Standard deviation: " + str(std))
    print("Lower bound: " + str(lower_bound_confidence))
    print("Upper bound: " + str(upper_bound_confidence))
    print("-----------------------------------------------------------------------------\n\n")

    #Variance 1 and mean 1

    theta_bar, theoretical_fraction, std, lower_bound_confidence, upper_bound_confidence =  simulation(_lambda, _service_time, runtimes, "erlang", "exponential")

    print("-------------------Erlang arrival and exponential service--------------------")
    print("Estimator: " + str(theta_bar))
    print("Standard deviation: " + str(std))
    print("Lower bound: " + str(lower_bound_confidence))
    print("Upper bound: " + str(upper_bound_confidence))
    print("-----------------------------------------------------------------------------\n\n")

    #Variance 1/5 and mean 1

    theta_bar, theoretical_fraction, std, lower_bound_confidence, upper_bound_confidence =  simulation(_lambda, _service_time, runtimes, "erlang", "constant")

    print("-------------------Erlang arrival and constant service-------------------")
    print("Estimator: " + str(theta_bar))
    #print("Theoretical fraction: " + str(theoretical_fraction))
    print("Standard deviation: " + str(std))
    print("Lower bound: " + str(lower_bound_confidence))
    print("Upper bound: " + str(upper_bound_confidence))
    print("-----------------------------------------------------------------------------\n\n")

    theta_bar, theoretical_fraction, std, lower_bound_confidence, upper_bound_confidence =  simulation(_lambda, _service_time, runtimes, "hyperexponential", "exponential")

    print("-------------------Hyperexponential arrival and exponential service--------------------")
    print("Estimator: " + str(theta_bar))
    print("Standard deviation: " + str(std))
    print("Lower bound: " + str(lower_bound_confidence))
    print("Upper bound: " + str(upper_bound_confidence))
    print("---------------------------------------------------------------------------------------\n\n")

    #Variance 1.32 and mean 1

    theta_bar, theoretical_fraction, std, lower_bound_confidence, upper_bound_confidence =  simulation(_lambda, _service_time, runtimes, "hyperexponential", "constant")

    print("-------------------Hyperexponential arrival and constant service-------------------")
    print("Estimator: " + str(theta_bar))
    #print("Theoretical fraction: " + str(theoretical_fraction))
    print("Standard deviation: " + str(std))
    print("Lower bound: " + str(lower_bound_confidence))
    print("Upper bound: " + str(upper_bound_confidence))
    print("-----------------------------------------------------------------------------\n\n")


if __name__ == "__main__":
    main()
