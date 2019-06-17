
import sys
import numpy as np
import math
from scipy.stats import t
import random
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import scipy.io as sio

#####################################################3
def distMat(stations):
    l = len(stations)
    dist_mat = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            p1 = stations[i, :]
            p2 = stations[j, :]
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            dist_mat[i, j] = dist
    return  dist_mat

def permute(r, n):
    r_new = r.copy()
    ind1 = np.random.choice(a=range(n-1))+1
    ind2 = np.random.choice(a=range(n-1))+1
    r_new[ind2] = r[ind1]
    r_new[ind1] = r[ind2]
    return(r_new)

def routeCost(i_list, dist_mat):
    s = 0
    for i in range(len(i_list)-1):
        s += dist_mat[int(i_list[i]), int(i_list[i+1])]
    return s

def newRoute(i_list, route, n):
    new_route = np.zeros((n+1,2))
    new_route[0] = route[0]
    for i in range(n):
        i=i+1
        id = int(i_list[i])
        new_route[i]= route[id]
    return new_route

#######################################################
# Implement simulated annealing for TSP:
n = 20
ite = 20000
stations = np.random.rand(n,2)
dist_mat = distMat(stations)

route = np.zeros((n+1,2))
route_index = np.zeros(n+1)
route[n] = stations[0,:]
route_index[n] = 0

for i in range(n):
    route[i] = stations[i,:]
    route_index[i] = i

c=0
for k in range(ite):
    accept = False
    new_ind = permute(route_index, n)
    T_k = 1.0/math.sqrt(1+k)
    X_k = routeCost(route_index, dist_mat)
    U_k = routeCost(new_ind, dist_mat)

    accept = (U_k <= X_k)
    if (U_k > X_k):
        nr = np.random.rand(1)
        acc_prob = math.exp(-(U_k-X_k)/T_k)
        accept = (nr <= acc_prob)

    if accept:
        c+=1
        route_index = new_ind

######################################################
# Plotting route:
org_index = np.array(range(n+1))
org_index[n] = org_index[0]
# original route:
x_end = route[:,0]
y_end = route[:,1]
plt.plot(x_end,y_end,'b-',alpha=0.1)
plt.title("Original Route")
plt.show()

print("--------Original Route---------")
print("Original route: " + str(org_index))
print("Cost of route: " + str(routeCost(org_index, dist_mat)))
# new route:
new_route = newRoute(route_index, route, n)
x_end = new_route[:,0]
y_end = new_route[:,1]
plt.plot(x_end,y_end,'b-',alpha=0.1)
plt.title("New route")
plt.show()
print("--------New Route---------")
print("New route: " + str(route_index))
print("Cost of new route: " + str(routeCost(route_index, dist_mat)))
print("Number of updates accepted: " + str(c))

##########################################################
# Debugging using the unit circle:
n=20
ite = 20000
s = np.array(range(n))/float(n)*2*math.pi
x_s = np.cos(s)
y_s = np.sin(s)
stations = np.transpose(np.array([x_s, y_s]))
dist_mat = distMat(stations)

route_index = np.array(range(n-1))+np.ones(n-1)
random.shuffle(route_index)
route_index = np.insert(route_index, 0, 0)
route_index = np.append(route_index, 0)
org_index = route_index.copy()
org_route = np.append(stations, [[1,0]], axis =0)

route = np.zeros((n+1,2))
route[n] = stations[int(route_index[0]),:]
for i in range(n):
    id = int(route_index[i])
    route[i] = stations[id,:]
route = newRoute(route_index, route, n)

c=0
for k in range(ite):
    accept = False
    new_ind = permute(route_index, n)
    T_k = 1.0/math.sqrt(1+k)
    X_k = routeCost(route_index, dist_mat)
    U_k = routeCost(new_ind, dist_mat)

    accept = (U_k <= X_k)
    if (U_k > X_k):
        nr = np.random.rand(1)
        acc_prob = math.exp(-(U_k-X_k)/T_k)
        accept = (nr <= acc_prob)

    if accept:
        c+=1
        route_index = new_ind

###################################################
# Plotting circle:
#Original route:
x_end = route[:,0]
y_end = route[:,1]
print("--------Original Route - Circle---------")
print("Original route: " + str(org_index))
print("Cost of route: " + str(routeCost(org_index, dist_mat)))
plt.plot(x_end,y_end,'b-',alpha=0.1)
plt.title("Original Route - Circle")
plt.show()

# new route:
new_route = newRoute(route_index, org_route, n)
x_end = new_route[:,0]
y_end = new_route[:,1]
print("--------New Route - Circle---------")
print("New route: " + str(route_index))
print("Cost of new route: " + str(routeCost(route_index, dist_mat)))
print("Number of updates accepted: " + str(c))
plt.plot(x_end,y_end,'b-',alpha=0.1)
plt.title("New route - Circle")
plt.show()

########################################################################

# Importing matlab cost matrix:
n=20
ite = 20000
mat = sio.loadmat("matrix.mat")
dist_mat = mat['c']
route_index = np.array(range(n-1))+np.ones(n-1)
route_index = np.insert(route_index, 0, 0)
route_index = np.append(route_index, 0)
org_index = route_index.copy()

c=0
for k in range(ite):
    accept = False
    new_ind = permute(route_index, n)
    T_k = 1.0/math.sqrt(1+k)
    X_k = routeCost(route_index, dist_mat)
    U_k = routeCost(new_ind, dist_mat)

    accept = (U_k <= X_k)
    if (U_k > X_k):
        nr = np.random.rand(1)
        acc_prob = math.exp(-(U_k-X_k)/T_k)
        accept = (nr <= acc_prob)

    if accept:
        c+=1
        route_index = new_ind

#print(mat.keys())
#print(mat['c'])

print("--------Original Route - Dist mat from DTU inside---------")
print("Original route: " + str(org_index))
print("Cost of route: " + str(routeCost(org_index, dist_mat)))
print("--------New Route - Dist mat from DTU inside---------")
print("New route: " + str(route_index))
print("Cost of new route: " + str(routeCost(route_index, dist_mat)))
print("Number of updates accepted: " + str(c))