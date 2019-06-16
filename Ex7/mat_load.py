
import scipy.io as sio

mat = sio.loadmat("matrix.mat")
print(mat.keys())
print(mat['c'])