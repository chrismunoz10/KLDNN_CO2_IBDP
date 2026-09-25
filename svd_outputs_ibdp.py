import numpy as np
import matplotlib.pyplot as plt
import math
import time as time
import scipy.linalg as spla
import scipy.sparse.linalg as sspla
import numpy.linalg as la
import os
import xarray as xr
from svd_routines_multikl import  *
######################################################### SVD: 3d 2024 DATASET ########################################################
NR = 80 #Number of realizations used

test_case = np.arange(10, NR + 1, step = 10, dtype = int)
test_size = len(test_case)
train_size = NR - test_size
cases = np.arange(1, NR+1, step = 1, dtype = int)
train_case = np.delete(cases - 1, test_case - 1)

#Terms on each KL Expansion
Nxi = 6

file_sats = '' #shape: (Nens, nx, ny, nz, nt)

#Plots:
fsize = 20 #fontsize
lsize = 12 #labelsize for ticks

#Reduced domain for everything
x_min = 30
x_max = 69
y_min = 50
y_max = 93
z_min = 0
z_max = 93
xyz_boundaries = [x_min, x_max, y_min, y_max, z_min, z_max]

####################################################### SATURATION #################################################################

if NR == 20:
    sats = np.load(file_sats)[-NR:]
else:
    sats = np.load(file_sats)[:NR]
print(sats.shape)

#Change impermeable cells (value = -99), to zero
sats[sats == -99] = 0.0

Nens, nx, ny, nz, nt = sats.shape
sats = np.reshape(sats, (Nens, -1))

#Split into train / test
s_train = np.delete(sats, test_case - 1, axis = 0) #(Ntrain, Nxyz)
s_test = sats[test_case - 1] #(Ntrain, Nxyz)

Nxyz = s_train.shape[1]
Nens = train_size

del sats

weights = np.load('') #shape: (nx, ny, nz)
weights = np.tile(weights[..., np.newaxis], nt)
weights = np.reshape(weights, (-1, 1))
weights[weights<1000.0] = 1000.0


############################# Saturation Field Decomposition ########################################################

############################# Do KL Expansion of Saturation ##########################################
print('Saturation')
start_time = time.time()
s_mean = np.mean(s_train, axis = 0) #(Nzxy)

xi_s, psi_s, xi_s_test, eigvals_s, eigvecs_s, _, _ = get_xi_weights(s_train, s_test, weights, Nxi = Nxi, word = 's', number = 0)
A = la.inv(psi_s.T @ psi_s) @ psi_s.T
b = -1 * la.inv(psi_s.T @ psi_s) @ psi_s.T @ s_mean

np.save(f'outputs_weight/xi_s_Nxi{Nxi}_NR{NR}.npy', xi_s)
np.save(f'outputs_weight/psi_s_Nxi{Nxi}_NR{NR}.npy', psi_s)
np.save(f'outputs_weight/xi_s_test_Nxi{Nxi}_NR{NR}.npy', xi_s_test)
np.save(f'outputs_weight/eigvals_s_NR{NR}.npy', eigvals_s)
np.save(f'outputs_weight/matrix_s_NR{NR}_Nxi{Nxi}.npy', A)
np.save(f'outputs_weight/bias_s_NR{NR}_Nxi{Nxi}.npy', b)
np.save(f'outputs_weight/s_mean_NR{NR}.npy', s_mean)

print("KL Expansion of whole Domain xy: --- %s seconds ---" % (time.time() - start_time))


####################### Reconstruction of Train and Test Fields #######################
s_train_rec = s_mean + (psi_s @ xi_s.T).T
s_test_rec = s_mean + (psi_s @ xi_s_test.T).T
