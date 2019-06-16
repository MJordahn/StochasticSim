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
    for i, k in enumerate(array):
        print(k)
        histogram[k] = histogram[k] + 1
    for k, prob in enumerate(histogram):
        histogram[k] = histogram[k]/n
    return histogram

def probabilities_2d(array1, array2, n):
    histogram = [[0]*11]*11
    for i, k in enumerate(array1):
        histogram[array1[i]][array2[i]] = histogram[array1[i]][array2[i]] + 1
    histogram[array1[i]][array2[i]] /= n
    return histogram


def main():
    # _lambda = 1
    # _mean_service_time = 8
    # s_nr = 10
    # A = 1*8
    # n = 100
    # states = [1]
    # for i in range(0, n-1):
    #     accept = False
    #     sample = np.random.choice(a=[-1, 0, 1])
    #     if states[-1] + sample > 10:
    #         y = 0
    #     elif states[-1] + sample < 0:
    #         y = 10
    #     else:
    #         y = states[-1] + sample
    #     g_x = A**states[-1]/math.factorial(states[-1])
    #     g_y = A**y/math.factorial(y)
    #     rnd = random.random()
    #
    #     accept = (g_y/g_x > rnd)
    #
    #     if accept:
    #         states.append(y)
    #     else:
    #         states.append(states[-1])
    #
    # plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], probabilities(states,n))
    # plt.show()
    #
    # stations = []
    # station_sum = 0
    # for i in range(s_nr+1):
    #     station_sum += A ** i / math.factorial(i)
    #
    # for i in range(s_nr+1):
    #     stations.append((A ** i / math.factorial(i)) / station_sum)
    #
    #
    # plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], stations)
    # plt.show()
    #
    # print(chisquare(probabilities(states,n), f_exp=stations))

    print("-------------------TWO CALL LINES DIRECTLY---------------------")
    s_nr = 10
    A_i = 1*4
    A_j = 1*4
    n = 10000
    states_i = [0]
    states_j = [0]
    states_combined = [0]
    for i in range(0, n-1):
        accept = False
        sample_i = np.random.choice(a=[-1, 0, 1])
        sample_j = np.random.choice(a=[-1, 0, 1])
        new_i = states_i[-1]+sample_i
        new_j = states_j[-1]+sample_j
        if new_i+new_j == s_nr+2:
            new_i = 0
            new_j = 0
        elif new_i+new_j > s_nr:
            if new_i > new_j:
                new_i = new_i-new_j
                new_j = 0
            else:
                new_j = new_j - new_i
                new_i = 0
        elif new_i < 0 and new_j >= 0:
            new_i = s_nr-new_j
        elif new_j < 0 and new_i >= 0:
            new_j = s_nr-new_i
        elif new_i<0 and new_j<0:
            new_i = 5
            new_j = 5
        g_x = A_i**states_i[-1]/math.factorial(states_i[-1])*A_j**states_j[-1]/math.factorial(states_j[-1])
        g_y = A_i**new_i/math.factorial(new_i)*A_j**new_j/math.factorial(new_j)
        rnd = random.random()

        accept = (g_y/g_x > rnd)
        if accept:
            states_i.append(new_i)
            states_j.append(new_j)
        else:
            states_i.append(states_i[-1])
            states_j.append(states_j[-1])

    # plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], probabilities(states_combined,n))
    # plt.show()
    prob_mat = probabilities_2d(array1, array2, n)
    stations = [[0]*11]*11
    #stations = [0]*11
    station_sum = 0

    for i in range(s_nr+1):
        for j in range(s_nr+1-i):
            stations[i][j]=stations[i][j]+A_j**j*A_i**i/math.factorial(i)/math.factorial(j)


    stations = stations / np.linalg.norm(stations)
    # plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], stations)
    # plt.show()
    chi2 = 0
    for i, row in enumerate(prob_mat):
        for j, cell in enumerate(row):
            chi2 = chi2 + (cell-stations[i][j])**2/stations[i][j]




if __name__ == "__main__":
    main()
