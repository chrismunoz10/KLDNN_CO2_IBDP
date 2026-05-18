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

file_sats = '/home/camunoz4/IBDP_Files/sats.npy'
file_pres = '/home/camunoz4/IBDP_Files/pressures_modified.npy'

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
print(sats.shape)
print(s_train.shape)
print(s_test.shape)

del sats

weights = np.load('../IBDP_2024Dataset/ibdp-tartan-grid-volumes_float32.npy')[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
weights = np.tile(weights[..., np.newaxis], nt)
weights = np.reshape(weights, (-1, 1))
weights[weights<1000.0] = 1000.0

weights_t = weights.flatten()
total_weights = np.sum(weights_t)

print(weights.shape)
print(weights_t.shape)

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
#np.save(f'outputs_weight/eigvecs_s_NR{NR}.npy', eigvecs_s)
np.save(f'outputs_weight/matrix_s_NR{NR}_Nxi{Nxi}.npy', A)
np.save(f'outputs_weight/bias_s_NR{NR}_Nxi{Nxi}.npy', b)
np.save(f'outputs_weight/s_mean_NR{NR}.npy', s_mean)

#np.save(f'residuals_train_layer_Nxi{Nxi}.npy', residuals_train_layer)
#np.save(f'residuals_test_layer_Nxi{Nxi}.npy', residuals_test_layer)
print("KL Expansion of whole Domain xy: --- %s seconds ---" % (time.time() - start_time))


####################### Reconstruction of Train and Test Fields #######################
s_train_rec = s_mean + (psi_s @ xi_s.T).T
s_test_rec = s_mean + (psi_s @ xi_s_test.T).T

#np.save(f'outputs/s_train_rec_Nxi{Nxi}_NR{NR}.npy', s_train_rec)
#np.save(f'outputs/s_test_rec_Nxi{Nxi}_NR{NR}.npy', s_test_rec)

### Reconstruction errors
errors_linear = np.zeros((test_size, 2))
for i, test in enumerate(test_case):
    u_true = s_test[i]
    u_regres = s_test_rec[i]
    error = (u_regres - u_true)**2
    error_mean = (s_mean - u_true)**2
    weighted_error = np.sum(error * weights_t) / total_weights
    weighted_error_mean = np.sum(error_mean * weights_t) / total_weights
    errors_linear[i, 0] = np.sqrt(weighted_error)
    errors_linear[i, 1] = np.sqrt(weighted_error_mean)
print(f'Weighted Testing KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')   
np.save(f'outputs_weight/errors_s_reconst_svd_Nxi{Nxi}_NR{NR}_test.npy', errors_linear)

errors_linear = np.zeros((test_size, 2))
for i, test in enumerate(test_case):
    u_true = s_test[i]
    u_regres = s_test_rec[i]
    error = (u_regres - u_true)**2
    error_mean = (s_mean - u_true)**2
    errors_linear[i, 0] = np.sqrt(np.mean(error))
    errors_linear[i, 1] = np.sqrt(np.mean(error_mean))
print(f'Unweighted Testing KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')   
#np.save(f'outputs_weight/errors_s_reconst_svd_Nxi{Nxi}_NR{NR}_test.npy', errors_linear)

### Reconstruction errors
errors_linear = np.zeros((train_size, 2))
for i, test in enumerate(train_case):
    u_true = s_train[i]
    u_regres = s_train_rec[i]
    error = (u_regres - u_true)**2
    error_mean = (s_mean - u_true)**2
    weighted_error = np.sum(error * weights_t) / total_weights
    weighted_error_mean = np.sum(error_mean * weights_t) / total_weights
    errors_linear[i, 0] = np.sqrt(weighted_error)
    errors_linear[i, 1] = np.sqrt(weighted_error_mean)
print(f'Weighted Training KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')     
np.save(f'outputs_weight/errors_s_reconst_svd_Nxi{Nxi}_NR{NR}_train.npy', errors_linear)

errors_linear = np.zeros((train_size, 2))
for i, test in enumerate(train_case):
    u_true = s_train[i]
    u_regres = s_train_rec[i]
    error = (u_regres - u_true)**2
    error_mean = (s_mean - u_true)**2
    errors_linear[i, 0] = np.sqrt(np.mean(error))
    errors_linear[i, 1] = np.sqrt(np.mean(error_mean))
print(f'Unweighted Training KL Approximation / Average Error: {np.mean(errors_linear[:, 0])} / {np.mean(errors_linear[:, 1])}')     

del s_train_rec
del s_test_rec
del A, b, s_mean


##################################### EIGENVALUES PLOTS ##########################################
s_normal = eigvals_s / eigvals_s[0]
#p_normal = eigvals_p / eigvals_p[0]

N = train_size - 1
if NR == 80:
    N = 50
else:
    N = 15
#Plot Normalized Eigenvalues
plt.figure(figsize = (16, 12), dpi = 300)
plt.semilogy(np.arange(N), s_normal[:N], label = r'$s$', marker = 'o')
plt.legend()
#plt.title(f'Normalized Eigenvalues using Ensemble Size of {NR}')
plt.xlabel(r'$i$', fontsize = fsize)
plt.ylabel(r'$\lambda$', fontsize = fsize)
plt.xticks(fontsize = lsize * 2)
plt.yticks(fontsize = lsize * 2)
plt.legend(fontsize = lsize * 2)
#plt.savefig(f'paper_plots/normalized_eigvals_outputs_NR{NR}.png', dpi = 300)
plt.savefig(f'outputs_weight/normalized_eigvals_NR{NR}.png', dpi = 300) 
plt.close()

################################### Plot first xi component for each input variable ################

#s 
plt.figure()
plt.scatter(train_case, xi_s[:, 0], label = 'Train')
plt.scatter(test_case, xi_s_test[:, 0], label = 'Test')
plt.legend()
plt.title(r'$\xi_1$ of $K_{xy}$ for each Case')
plt.savefig(f'outputs_weight/xi_sat_component1_NR{NR}.png', dpi = 300)
plt.close()


