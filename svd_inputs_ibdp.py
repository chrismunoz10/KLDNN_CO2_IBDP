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

weights = np.load('') #shape: (nx, ny, nz)
weights = np.reshape(weights, (-1, 1))
weights[weights<1000.0] = 1000.0
weights_all = np.concatenate((weights, weights, weights), axis = 0)
print(weights_all.shape)
weights = weights.flatten()
total_weights_all = np.sum(weights_all)
total_weights = np.sum(weights)

#Terms on each KL Expansion
Nxi = 3
if NR == 20:
    Nxi_all = 9
elif NR == 80: 
    Nxi_all = 18

file_perm = '' #shape: (Nens, nx, ny, nz, 3)
file_por = '' #shape: (Nens, nx, ny, nz)

#Plots:
fsize = 20 #fontsize
lsize = 12 #labelsize for ticks

####################################################### PERMEABILITIES #################################################################
if NR == 20:
    Kxyz = np.load(file_perm)[-NR:]
    por = np.load(file_por)[-NR:]
    
else:
    Kxyz = np.load(file_perm)[:NR]
    por = np.load(file_por)[:NR]

Kxyz[Kxyz == 0.0] = np.nan

Kmin = []
for i in range(NR):
    for j in range(3):
        Kmin.append(0.1 * np.nanmin(Kxyz[i, :, :, :, j]))
        Kxyz_i = Kxyz[i, :, :, :, j].copy()
        Kxyz_i[np.where(np.isnan(Kxyz_i))] = 0.1 * np.nanmin(Kxyz_i)
        Kxyz[i, :, :, :, j] = Kxyz_i.copy()


Kxy = 0.5 * (Kxyz[:, :, :, :, 0] + Kxyz[:, :, :, :, 1]) #Use average of x and y direction as the horizontal permeability

y_ens_xy = np.log(Kxy)
y_ens_z = np.log(Kxyz[:, :, :, :, 2])

Nens, nx, ny, nz = y_ens_xy.shape
y_ens_xy = np.reshape(y_ens_xy, (Nens, -1))
y_ens_z = np.reshape(y_ens_z, (Nens, -1))
por = np.reshape(por, (Nens, -1))

inputs_all = np.concatenate((y_ens_xy, y_ens_z, por), axis = 1)
#Split into train / test
y_train_xy = np.delete(y_ens_xy, test_case - 1, axis = 0) #(Ntrain, Nxyz)
y_test_xy = y_ens_xy[test_case - 1] #(Ntrain, Nxyz)
y_mean_xy = np.mean(y_train_xy, axis = 0) #(Nzxy)

y_train_z = np.delete(y_ens_z, test_case - 1, axis = 0) #(Ntrain, Nxyz)
y_test_z = y_ens_z[test_case - 1] #(Ntrain, Nxyz)
y_mean_z = np.mean(y_train_z, axis = 0) #(Nzxy)

por_train = np.delete(por, test_case - 1, axis = 0) #(Ntrain, Nxyz)
por_test = por[test_case - 1] #(Ntrain, Nxyz)
por_mean = np.mean(por_train, axis = 0) #(Nzxy)

all_train = np.delete(inputs_all, test_case - 1, axis = 0)
all_test = inputs_all[test_case - 1]

Nxyz = y_train_xy.shape[1]
Nens = train_size
print(y_ens_xy.shape)
print(y_train_xy.shape)
print(y_test_xy.shape)
print(inputs_all.shape)


############################# Do KL Expansion of ALL Properties at Once ##########################################
print('All Properties')
start_time = time.time()
all_mean = np.mean(all_train, axis = 0) #(Nzxy)

xi_all, psi_all, xi_all_test, eigvals_all, eigvecs_all, _, _ = get_xi(all_train, all_test, Nxi = Nxi_all, word = 'all', number = 0)
A = la.inv(psi_all.T @ psi_all) @ psi_all.T
b = -1 * la.inv(psi_all.T @ psi_all) @ psi_all.T @ all_mean

np.save(f'inputs_all/xi_all_Nxi{Nxi_all}_NR{NR}.npy', xi_all)
np.save(f'inputs_all/psi_all_Nxi{Nxi_all}_NR{NR}.npy', psi_all)
np.save(f'inputs_all/xi_all_test_Nxi{Nxi_all}_NR{NR}.npy', xi_all_test)
np.save(f'inputs_all/eigvals_all_NR{NR}.npy', eigvals_all)
np.save(f'inputs_all/eigvecs_all_NR{NR}.npy', eigvecs_all)
np.save(f'inputs_all/matrix_all_NR{NR}_Nxi{Nxi_all}.npy', A)
np.save(f'inputs_all/bias_all_NR{NR}_Nxi{Nxi_all}.npy', b)
np.save(f'inputs_all/y_mean_all_NR{NR}_Nxi{Nxi_all}.npy', all_mean)

print("KL Expansion of whole Domain All Properties: --- %s seconds ---" % (time.time() - start_time))


####################### Reconstruction of Train and Test Fields #######################

#np.save('inputs/mean_y_xy.npy', y_mean_xy)
y_train_rec_all = all_mean + (psi_all @ xi_all.T).T
y_test_rec_all = all_mean + (psi_all @ xi_all_test.T).T

y_train_rec_xy = y_train_rec_all[:, :Nxyz]
y_train_rec_z = y_train_rec_all[:, Nxyz:2*Nxyz]
y_train_rec_por = y_train_rec_all[:, 2*Nxyz:]

y_test_rec_xy = y_test_rec_all[:, :Nxyz]
y_test_rec_z = y_test_rec_all[:, Nxyz:2*Nxyz]
y_test_rec_por = y_test_rec_all[:, 2*Nxyz:]
