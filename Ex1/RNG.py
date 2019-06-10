
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import norm

#########################################################
def LCG(x, a, c, M):
    #Generates a random number using LCG
    x_i = (a*x+c)%M
    return x_i


def runSim(seed, a, c, M, p):
    x = np.zeros(10000)
    x[0] = seed

    for i in range(len(x)-1):
        i=i+1
        x[i]=LCG(x[i-1],a,c,M)

    if True:
        plt.hist(x, range(int(M)+1))

    return(x)

def countOcc(lst, M):
    x = np.zeros(int(M))
    for i in range(len(lst)):
        z=int(lst[i])
        x[z]+=1
    return(x)

def run1(lst, med):
    v = 0
    n1 = 0
    n2 = 0
    for i in range(len(lst)):
        if ((lst[i] <= med) and (v==1)):
            n1 += 1
            v=0
        elif ((lst[i] > med) and (v==0)):
            n2 += 1
            v=1
    return (n1, n2)

def run2(lst):
    R = np.zeros(6)
    tempLen = 0
    temp = lst[0]
    for i in range(len(lst))[1:]:
        if (lst[i] >= temp and tempLen < 5):
            tempLen += 1
        else:
            R[tempLen] +=1
            tempLen = 0
        temp = lst[i]
    if tempLen != 0:
        R[tempLen] += 1
    return R
####################################################
# We generate 10000 numbers:
seed=0
a=1.0
c=2.0
M=13.0
p=True

x1 = runSim(seed, a, c, M, p)
U1 = x1/M
print(x1[0:14]) #The period is 13 - Full cycle length. See slide
#We see, that not all values are simulated.


# We now make a=1:
seed=0
a=2
c=2
M=13
p=True

x2 = runSim(seed, a, c, M, p)
U2= x2/M
print(x2[0:14]) #The period is 12

#Is is possible to create full cycle with M=20? No -since mod(a,4)=1 \imples mod(a,5) \neq 1

###################################################
# We evaluate the quality of the different generators:
#The null hypothesis: Numbers are uniformly distributed.

#We compute the test statistic of the first run:
n_obs1 = countOcc(x1, M)
n_ex1 = 10000.0/M
T1=np.dot((n_obs1-n_ex1*np.ones(int(M))), (n_obs1-n_ex1*np.ones(int(M))))/n_ex1
#p-value:
1-chi2.cdf(T1, M-1)
#accept

#We compute the test statistic of the second run:
n_obs2 = countOcc(x2, M)
n_ex2 = 10000.0/M
T2=np.dot((n_obs2-n_ex2*np.ones(M)), (n_obs2-n_ex2*np.ones(M)))/n_ex2
#p-value:
1-chi2.cdf(T2, M-1)
#reject

#################################################################

c1 = np.cumsum(n_obs1)*(1.0/10000)
plt.plot(range(M),(np.cumsum(n_obs1)*(1.0/10000)))

c2 = np.cumsum(n_obs2)*(1.0/10000)
plt.plot(range(M),np.cumsum(n_obs2)*(1.0/10000))

#We compute the test statistic for the Kolmogorov test:
diff1 = c1-np.cumsum(np.ones(M, dtype=float))*(1.0/13.0)
D1 = max((abs(max(diff1)), abs(min(diff1))))

diff2 = c2-np.cumsum(np.ones(M, dtype=float))*(1.0/13.0)
D2 = max((abs(max(diff2)), abs(min(diff2))))


#Run 1:
D1_adjusted = (np.sqrt(10000) + 0.12 + 0.11/100)*D1
#accept

#Run 2:
D2_adjusted = (np.sqrt(10000) + 0.12 + 0.11/100)*D2
#reject

###############################################################3##
# First the scatter plot:
U1_o = U1[:len(U1)-1]
U1_s = U1[1:]
plt.scatter(U1_o, U1_s)

U2_o = U2[:len(U2)-1]
U2_s = U2[1:]
plt.scatter(U2_o, U2_s)

##################################################################
#Conradsen test:

#Testing first batch:
median = np.median(x1)
n1_1, n2_1 = run1(x1, median)

mean1 = 2*(n1_1*n2_1)/(n1_1 + n2_1) + 1
t1 = n1_1*n2_1
var1 = 2*t1*(2*t1 - n1_1 - n2_1)/((n1_1+n2_1)**2*(n1_1+n2_1-1))

#n1_1 is the test statistic.
norm.cdf(n1_1, loc=mean1, scale=np.sqrt(var1))

#Testing second batch:
median = np.median(x2)
n1_2, n2_2 = run1(x2, median)

mean2 = 2*(n1_2*n2_2)/(n1_2 + n2_2) + 1
t2 = n1_2*n2_2
var2 = 2*t2*(2*t2 - n1_2 - n2_2)/((n1_2+n2_2)**2*(n1_2+n2_2-1))

#n1_1 is the test statistic.
norm.cdf(n1_2, loc=mean2, scale=np.sqrt(var2))

###################################################################
#extra stuff
R1 = run2(x1)
R2 = run2(x2)

B = np.array([1.0/6, 5.0/24, 11.0/120, 19.0/720, 29.0/5040, 1.0/840])

Z1 = (1.0/(10000.0-6.0))*(R1-10000.0*B)
