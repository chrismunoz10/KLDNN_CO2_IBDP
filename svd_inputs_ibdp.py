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

weights = np.load('../IBDP_2024Dataset/ibdp-tartan-grid-volumes_float32.npy')
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

file_perm = '/home/camunoz4/IBDP_Files/perm_xyz_scaled_float32.npy'
file_por = '/home/camunoz4/IBDP_Files/porosity_float32.npy'

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

#Change impermeable cells (value = 0), to the smallest value of its respective realization
#Realization 8 has 0 everywhere for Kx, put Ky into it
Kxyz[8, :, :, :, 0] = Kxyz[8, :, :, :, 1].copy()

Kxyz[Kxyz == 0.0] = np.nan

Kmin = []
for i in range(NR):
    for j in range(3):
        Kmin.append(0.1 * np.nanmin(Kxyz[i, :, :, :, j]))
        Kxyz_i = Kxyz[i, :, :, :, j].copy()
        Kxyz_i[np.where(np.isnan(Kxyz_i))] = 0.1 * np.nanmin(Kxyz_i)
        Kxyz[i, :, :, :, j] = Kxyz_i.copy()


Kxy = 0.5 * (Kxyz[:, :, :, :, 0] + Kxyz[:, :, :, :, 1])

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

"""
############################# Conductivity Field Decomposition ########################################################

############################# Do KL Expansion of Horizontal Permeability ##########################################
print('Horizontal Permeability')
start_time = time.time()


xi_xy, psi_xy, xi_xy_test, eigvals_xy, eigvecs_xy, _, _ = get_xi(y_train_xy, y_test_xy, Nxi = Nxi, word = 'xy', number = 0)
A = la.inv(psi_xy.T @ psi_xy) @ psi_xy.T
b = -1 * la.inv(psi_xy.T @ psi_xy) @ psi_xy.T @ y_mean_xy

np.save(f'inputs/xi_xy_Nxi{Nxi}_NR{NR}.npy', xi_xy)
np.save(f'inputs/psi_xy_Nxi{Nxi}_NR{NR}.npy', psi_xy)
np.save(f'inputs/xi_xy_test_Nxi{Nxi}_NR{NR}.npy', xi_xy_test)
np.save(f'inputs/eigvals_xy_NR{NR}.npy', eigvals_xy)
np.save(f'inputs/eigvecs_xy_NR{NR}.npy', eigvecs_xy)
np.save(f'inputs/matrix_xy_NR{NR}_Nxi{Nxi}.npy', A)
np.save(f'inputs/bias_xy_NR{NR}_Nxi{Nxi}.npy', b)
np.save(f'inputs/y_mean_xy_NR{NR}_Nxi{Nxi}.npy', y_mean_xy)

#np.save(f'residuals_train_layer_Nxi{Nxi}.npy', residuals_train_layer)
#np.save(f'residuals_test_layer_Nxi{Nxi}.npy', residuals_test_layer)
print("KL Expansion of whole Domain xy: --- %s seconds ---" % (time.time() - start_time))


####################### Reconstruction of Train and Test Fields #######################

np.save('inputs/mean_y_xy.npy', y_mean_xy)
y_train_rec_xy = y_mean_xy + (psi_xy @ xi_xy.T).T
y_test_rec_xy = y_mean_xy + (psi_xy @ xi_xy_test.T).T

np.save(f'inputs/y_train_rec_xy_Nxi{Nxi}_NR{NR}.npy', y_train_rec_xy)
np.save(f'inputs/y_test_rec_xy_Nxi{Nxi}_NR{NR}.npy', y_test_rec_xy)

### Reconstruction errors
errors_linear = np.zeros((test_size, 2))
for i, test in enumerate(test_case):
    u_true = y_test_xy[i]
    u_regres = y_test_rec_xy[i]
    error = (u_regres - u_true)**2
    error_mean = (y_mean_xy - u_true)**2
    errors_linear[i, 0] = np.sqrt(np.mean(error))
    errors_linear[i, 1] = np.sqrt(np.mean(error_mean))
print(f'Testing KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')   
np.save(f'inputs/errors_y_reconst_xy_svd_Nxi{Nxi}_NR{NR}_test.npy', errors_linear)

### Reconstruction errors
errors_linear = np.zeros((train_size, 2))
for i, test in enumerate(train_case):
    u_true = y_train_xy[i]
    u_regres = y_train_rec_xy[i]
    error = (u_regres - u_true)**2
    error_mean = (y_mean_xy - u_true)**2
    errors_linear[i, 0] = np.sqrt(np.mean(error))
    errors_linear[i, 1] = np.sqrt(np.mean(error_mean))
print(f'Training KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')     
np.save(f'inputs/errors_y_reconst_xy_svd_Nxi{Nxi}_NR{NR}_train.npy', errors_linear)


############################# Do KL Expansion of VERTICAL Permeability ##########################################

start_time = time.time()

xi_z, psi_z, xi_z_test, eigvals_z, eigvecs_z, _, _ = get_xi(y_train_z, y_test_z, Nxi = Nxi, word = 'z', number = 0)
A = la.inv(psi_z.T @ psi_z) @ psi_z.T
b = - 1 * la.inv(psi_z.T @ psi_z) @ psi_z.T @ y_mean_z

np.save(f'inputs/matrix_z_NR{NR}_Nxi{Nxi}.npy', A)
np.save(f'inputs/bias_z_NR{NR}_Nxi{Nxi}.npy', b)
np.save(f'inputs/xi_z_Nxi{Nxi}_NR{NR}.npy', xi_z)
np.save(f'inputs/psi_z_Nxi{Nxi}_NR{NR}.npy', psi_z)
np.save(f'inputs/xi_z_test_Nxi{Nxi}_NR{NR}.npy', xi_z_test)
np.save(f'inputs/eigvals_z_NR{NR}.npy', eigvals_z)
np.save(f'inputs/eigvecs_z_NR{NR}.npy', eigvecs_z)
print("KL Expansion of whole Domain z: --- %s seconds ---" % (time.time() - start_time))

####################### Reconstruction of Train and Test Fields #######################
#y_mean_z = np.mean(y_train_z, axis = 0) #(Nzxy)
np.save('inputs/mean_y_z.npy', y_mean_z)
y_train_rec_z = y_mean_z + (psi_z @ xi_z.T).T
y_test_rec_z = y_mean_z + (psi_z @ xi_z_test.T).T

np.save(f'inputs/y_train_rec_z_Nxi{Nxi}_NR{NR}.npy', y_train_rec_z)
np.save(f'inputs/y_test_rec_z_Nxi{Nxi}_NR{NR}.npy', y_test_rec_z)

### Reconstruction errors
errors_linear = np.zeros((test_size, 2))
for i, test in enumerate(test_case):
    u_true = y_test_z[i]
    u_regres = y_test_rec_z[i]
    error = (u_regres - u_true)**2
    error_mean = (y_mean_z - u_true)**2
    errors_linear[i, 0] = np.sqrt(np.mean(error))
    errors_linear[i, 1] = np.sqrt(np.mean(error_mean))
print(f'Testing KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')     
np.save(f'inputs/errors_y_reconst_z_svd_Nxi{Nxi}_NR{NR}_test.npy', errors_linear)

### Reconstruction errors
errors_linear = np.zeros((train_size, 2))
for i, test in enumerate(train_case):
    u_true = y_train_z[i]
    u_regres = y_train_rec_z[i]
    error = (u_regres - u_true)**2
    error_mean = (y_mean_z - u_true)**2
    errors_linear[i, 0] = np.sqrt(np.mean(error))
    errors_linear[i, 1] = np.sqrt(np.mean(error_mean))
print(f'Training KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')     
np.save(f'inputs/errors_y_reconst_z_svd_Nxi{Nxi}_NR{NR}_train.npy', errors_linear)

############################# Do KL Expansion of POROSITY ##########################################

start_time = time.time()

xi_por, psi_por, xi_por_test, eigvals_por, eigvecs_por, _, _ = get_xi(por_train, por_test, Nxi = Nxi, word = 'por', number = 0)

A = la.inv(psi_por.T @ psi_por) @ psi_por.T
b = -1 * la.inv(psi_por.T @ psi_por) @ psi_por.T @ por_mean
np.save(f'inputs/matrix_por_NR{NR}_Nxi{Nxi}.npy', A)
np.save(f'inputs/bias_por_NR{NR}_Nxi{Nxi}.npy', b)
np.save(f'inputs/xi_por_Nxi{Nxi}_NR{NR}.npy', xi_por)
np.save(f'inputs/psi_por_Nxi{Nxi}_NR{NR}.npy', psi_por)
np.save(f'inputs/xi_por_test_Nxi{Nxi}_NR{NR}.npy', xi_por_test)
np.save(f'inputs/eigvals_por_NR{NR}.npy', eigvals_por)
np.save(f'inputs/eigvecs_por_NR{NR}.npy', eigvecs_por)
#np.save(f'residuals_train_layer_Nxi{Nxi}.npy', residuals_train_layer)
#np.save(f'residuals_test_layer_Nxi{Nxi}.npy', residuals_test_layer)
print("KL Expansion of whole Domain Porosity: --- %s seconds ---" % (time.time() - start_time))

####################### Reconstruction of Train and Test Fields #######################

np.save('inputs/mean_porosity.npy', por_mean)
por_train_rec = por_mean + (psi_por @ xi_por.T).T
por_test_rec = por_mean + (psi_por @ xi_por_test.T).T

np.save(f'inputs/por_train_rec_Nxi{Nxi}_NR{NR}.npy', por_train_rec)
np.save(f'inputs/por_test_rec_Nxi{Nxi}_NR{NR}.npy', por_test_rec)

### Reconstruction errors
errors_linear = np.zeros((test_size, 2))
for i, test in enumerate(test_case):
    u_true = por_test[i]
    u_regres = por_test_rec[i]
    error = (u_regres - u_true)**2
    error_mean = (por_mean - u_true)**2
    errors_linear[i, 0] = np.sqrt(np.mean(error))
    errors_linear[i, 1] = np.sqrt(np.mean(error_mean))
print(f'Testing KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')     
np.save(f'inputs/errors_por_reconst_svd_Nxi{Nxi}_NR{NR}_test.npy', errors_linear)

### Reconstruction errors
errors_linear = np.zeros((train_size, 2))
for i, test in enumerate(train_case):
    u_true = por_train[i]
    u_regres = por_train_rec[i]
    error = (u_regres - u_true)**2
    error_mean = (por_mean - u_true)**2
    errors_linear[i, 0] = np.sqrt(np.mean(error))
    errors_linear[i, 1] = np.sqrt(np.mean(error_mean))
print(f'Training KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')     
np.save(f'inputs/errors_por_reconst_svd_Nxi{Nxi}_NR{NR}_train.npy', errors_linear)
"""

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

#np.save(f'residuals_train_layer_Nxi{Nxi}.npy', residuals_train_layer)
#np.save(f'residuals_test_layer_Nxi{Nxi}.npy', residuals_test_layer)
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

#np.save(f'inputs/y_train_rec_all_Nxi{Nxi}_NR{NR}.npy', y_train_rec_all)
#np.save(f'inputs/y_test_rec_all_Nxi{Nxi}_NR{NR}.npy', y_test_rec_all)

### Reconstruction errors
######################################################## Horizontal Permeability #######################################
errors_linear = np.zeros((test_size, 2))
for i, test in enumerate(test_case):
    u_true = y_test_xy[i]
    u_regres = y_test_rec_xy[i]
    error = (u_regres - u_true)**2
    error_mean = (y_mean_xy - u_true)**2
    weighted_error = np.sum(error * weights) / total_weights
    weighted_error_mean = np.sum(error_mean * weights) / total_weights
    errors_linear[i, 0] = np.sqrt(weighted_error)
    errors_linear[i, 1] = np.sqrt(weighted_error_mean)
print(f'Testing KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')   
np.save(f'inputs_all/errors_y_reconst_xy_svd_Nxi{Nxi}_NR{NR}_test.npy', errors_linear)

### Reconstruction errors
errors_linear = np.zeros((train_size, 2))
for i, test in enumerate(train_case):
    u_true = y_train_xy[i]
    u_regres = y_train_rec_xy[i]
    error = (u_regres - u_true)**2
    error_mean = (y_mean_xy - u_true)**2
    weighted_error = np.sum(error * weights) / total_weights
    weighted_error_mean = np.sum(error_mean * weights) / total_weights
    errors_linear[i, 0] = np.sqrt(weighted_error)
    errors_linear[i, 1] = np.sqrt(weighted_error_mean)
print(f'Training KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')     
np.save(f'inputs_all/errors_y_reconst_xy_svd_Nxi{Nxi}_NR{NR}_train.npy', errors_linear)

######################################################## Vertical Permeability #######################################
errors_linear = np.zeros((test_size, 2))
for i, test in enumerate(test_case):
    u_true = y_test_z[i]
    u_regres = y_test_rec_z[i]
    error = (u_regres - u_true)**2
    error_mean = (y_mean_z - u_true)**2
    weighted_error = np.sum(error * weights) / total_weights
    weighted_error_mean = np.sum(error_mean * weights) / total_weights
    errors_linear[i, 0] = np.sqrt(weighted_error)
    errors_linear[i, 1] = np.sqrt(weighted_error_mean)
print(f'Testing KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')   
np.save(f'inputs_all/errors_y_reconst_z_svd_Nxi{Nxi}_NR{NR}_test.npy', errors_linear)

### Reconstruction errors
errors_linear = np.zeros((train_size, 2))
for i, test in enumerate(train_case):
    u_true = y_train_z[i]
    u_regres = y_train_rec_z[i]
    error = (u_regres - u_true)**2
    error_mean = (y_mean_z - u_true)**2
    weighted_error = np.sum(error * weights) / total_weights
    weighted_error_mean = np.sum(error_mean * weights) / total_weights
    errors_linear[i, 0] = np.sqrt(weighted_error)
    errors_linear[i, 1] = np.sqrt(weighted_error_mean)
print(f'Training KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')     
np.save(f'inputs_all/errors_y_reconst_z_svd_Nxi{Nxi}_NR{NR}_train.npy', errors_linear)

######################################################## Porosity #######################################
errors_linear = np.zeros((test_size, 2))
for i, test in enumerate(test_case):
    u_true = por_test[i]
    u_regres = y_test_rec_por[i]
    error = (u_regres - u_true)**2
    error_mean = (por_mean - u_true)**2
    weighted_error = np.sum(error * weights) / total_weights
    weighted_error_mean = np.sum(error_mean * weights) / total_weights
    errors_linear[i, 0] = np.sqrt(weighted_error)
    errors_linear[i, 1] = np.sqrt(weighted_error_mean)
print(f'Testing KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')   
np.save(f'inputs_all/errors_y_reconst_n_svd_Nxi{Nxi}_NR{NR}_test.npy', errors_linear)

### Reconstruction errors
errors_linear = np.zeros((train_size, 2))
for i, test in enumerate(train_case):
    u_true = por_train[i]
    u_regres = y_train_rec_por[i]
    error = (u_regres - u_true)**2
    error_mean = (por_mean - u_true)**2
    weighted_error = np.sum(error * weights) / total_weights
    weighted_error_mean = np.sum(error_mean * weights) / total_weights
    errors_linear[i, 0] = np.sqrt(weighted_error)
    errors_linear[i, 1] = np.sqrt(weighted_error_mean)
print(f'Training KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')     
np.save(f'inputs_all/errors_y_reconst_n_svd_Nxi{Nxi}_NR{NR}_train.npy', errors_linear)

"""
##################################### EIGENVALUES PLOTS ##########################################
kxy_normal = eigvals_xy / eigvals_xy[0]
kz_normal = eigvals_z / eigvals_z[0]
n_normal = eigvals_por / eigvals_por[0]
N = train_size - 1
N = 15
#Plot Normalized Eigenvalues
plt.figure(figsize = (16, 12), dpi = 300)
plt.semilogy(np.arange(N), kxy_normal[:N], label = r'$K_{xy}$', marker = 'o')
plt.semilogy(np.arange(N), kz_normal[:N], label = r'$K_{z}$', marker = 'x')
plt.semilogy(np.arange(N), n_normal[:N], label = r'$n$', marker = '^')
plt.legend()
#plt.title(f'Normalized Eigenvalues using Ensemble Size of {NR}')
plt.xlabel('$i$', fontsize = fsize)
plt.ylabel('$\lambda$', fontsize = fsize)
plt.xticks(fontsize = lsize * 2)
plt.yticks(fontsize = lsize * 2)
plt.legend(fontsize = lsize * 2)
plt.savefig(f'paper_plots/normalized_eigvals_inputs_NR{NR}.png', dpi = 300)
plt.savefig(f'inputs/normalized_eigvals_NR{NR}.png', dpi = 300) 
plt.close()
"""


all_normal = eigvals_all / eigvals_all[0]
plt.figure(figsize = (16, 12), dpi = 300)
plt.semilogy(np.arange(N), all_normal[:N], marker = 'o')
plt.xlabel(r'$i$', fontsize = fsize)
plt.ylabel(r'$\lambda$', fontsize = fsize)
plt.xticks(fontsize = lsize * 2)
plt.yticks(fontsize = lsize * 2)
#plt.title(f'Normalized Eigenvalues (Mixed Decomposition) using Ensemble Size of {NR}')
plt.savefig(f'paper_plots/normalized_eigvals_inputs_all_NR{NR}.png', dpi = 300)
plt.savefig(f'inputs_all/normalized_eigvals_NR{NR}.png', dpi = 300) 
plt.close()
################################### Plot first xi component for each input variable ################

#All
#Kxy 
plt.figure()
plt.scatter(train_case, xi_all[:, 0], label = 'Train', marker = 'x')
plt.scatter(test_case, xi_all_test[:, 0], label = 'Test', marker = 'x')
plt.legend()
plt.grid()
plt.title(r'$\xi_1$ of $K_{xy}$ for each Case')
plt.savefig(f'inputs_all/xi_all_component1_NR{NR}.png', dpi = 300)
plt.close()
print(xi_all[-10:, 0])

# """
#Kxy 
# plt.figure()
# plt.scatter(train_case, xi_xy[:, 0], label = 'Train')
# plt.scatter(test_case, xi_xy_test[:, 0], label = 'Test')
# plt.legend()
# plt.title('\xi_1 of K_{xy} for each Case')
# plt.savefig(f'inputs/xi_xy_component1_NR{NR}.png', dpi = 300)
# plt.close()

#Kz
# plt.figure()
# plt.scatter(train_case, xi_z[:, 0], label = 'Train')
# plt.scatter(test_case, xi_z_test[:, 0], label = 'Test')
# plt.legend()
# plt.title('$\xi_1$ of $K_z$ for each Case')
# plt.savefig(f'inputs/xi_z_component1_NR{NR}.png', dpi = 300)
# plt.close()

#Porosity
# plt.figure()
# plt.scatter(train_case, xi_por[:, 0], label = 'Train')
# plt.scatter(test_case, xi_por_test[:, 0], label = 'Test')
# plt.legend()
# plt.title('$\xi_1$ of $n$ for each Case')
# plt.savefig(f'inputs/xi_por_component1_NR{NR}.png', dpi = 300)
# """
