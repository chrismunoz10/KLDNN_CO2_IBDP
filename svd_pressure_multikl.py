import numpy as np
import matplotlib.pyplot as plt
import math
import time
import scipy.linalg as spla
import scipy.sparse.linalg as sspla
import numpy.linalg as la
import os
from svd_routines_multikl import  *

NR = 80 #Number of realizations used
fsize = 30 #fontsize
lsize = 18 #labelsize for ticks

test_case = np.arange(10, NR + 1, step = 10, dtype = int)
ens_case = np.arange(1, NR+1, step = 1, dtype = int)
train_case = np.delete(ens_case, test_case - 1)
train_size = NR - len(test_case)
test_size = len(test_case)

#Terms on each KL Expansion
Neta = 4 #Nxyz -> Neta
Ngamma = 3 #nt -> Ngamma

realization = 10
if NR == 20:
    file_ensemble = '/home/camunoz4/IBDP_Files/pressures_modified_20.npy'
else:
    file_ensemble = '/home/camunoz4/IBDP_Files/pressures_modified.npy'

start_time = time.time()
#Load files:
p_ens_mesh = np.load(file_ensemble)[:NR]

Nens, nx, ny, nz, nt = p_ens_mesh.shape
Nxy = int(nx * ny * nz)
print(f'ny: {ny}\nnx: {nx}')

#Reshape by layers (Nens, Nxy, nz, nt)
p_ens_mesh = np.reshape(p_ens_mesh, (Nens, Nxy, nt))
p_ens_mesh = np.transpose(p_ens_mesh, (0, 2, 1)) #(Nens, nt, Nxy)

#p_ens_new = p_ens_mesh.copy()

#Split into train / test
p_train = np.delete(p_ens_mesh, test_case - 1, axis = 0) #(Ntrain, nt, Nxy)
p_test = p_ens_mesh[test_case - 1] #(Ntest, nt, Nxy)
p_mean = np.mean(p_train, axis = 0) #(nt, Nxy)
Nens = train_size

weights = np.load('../IBDP_2024Dataset/ibdp-tartan-grid-volumes_float32.npy')
weights = np.reshape(weights, (-1, 1))
weights[weights<1000.0] = 1000.0


weights_t = np.tile(weights.flatten()[..., np.newaxis], nt)
weights_t = np.transpose(weights_t, (1, 0))
total_weights_t = np.sum(weights_t)
print(p_ens_mesh.shape)
print(p_train.shape)
print(p_test.shape)
print(weights_t.shape)
print("Loading and reshaping all files: --- %s seconds ---" % (time.time() - start_time))

del p_ens_mesh

############################# Do KL Expansion on every Timestep / Layer ##########################################
start_time = time.time()
eta_time = np.zeros((nt, Nens, Neta))
psi_time = np.zeros((nt, Nxy, Neta))
eigvecs_time = np.zeros((nt, Nxy, Neta))
eigvals_time = np.zeros((nt, train_size))
eta_time_test = np.zeros((nt, test_size, Neta))

"""
for l in range(nt):
    start_nt = time.time()
    #print(f'Starting {l+1} / {nt} timestep')
    eta_time_l, psi_time_l, eta_time_test_l, eigvals_time_l, eigvecs_time_l, _, _ = get_xi_weights(p_train[:, l, :], p_test[:, l, :], weights, Nxi = Neta, word = 'time', number = str(l))
    eta_time[l] = eta_time_l
    psi_time[l] = psi_time_l
    eta_time_test[l] = eta_time_test_l
    eigvals_time[l] = eigvals_time_l
    eigvecs_time[l] = eigvecs_time_l
    #print("KL Expansion of one timestep: --- %s seconds ---" % (time.time() - start_nt))
np.save(f'Pressure_weight/eta_time_Neta{Neta}_NR{NR}.npy', eta_time)
np.save(f'Pressure_weight/psi_time_Neta{Neta}_NR{NR}.npy', psi_time)
np.save(f'Pressure_weight/eta_time_test_Neta{Neta}_NR{NR}.npy', eta_time_test)
np.save(f'Pressure_weight/eigvals_time_NR{NR}.npy', eigvals_time)
np.save(f'Pressure_weight/eigvecs_time_NR{NR}.npy', eigvecs_time)
"""
print("KL Expansion of every timestep: --- %s seconds ---" % (time.time() - start_time))



eta_time = np.load(f'Pressure_weight/eta_time_Neta{Neta}_NR{NR}.npy')
psi_time = np.load(f'Pressure_weight/psi_time_Neta{Neta}_NR{NR}.npy')
eta_time_test = np.load(f'Pressure_weight/eta_time_test_Neta{Neta}_NR{NR}.npy')

############################# Do KL Expansion on each Timestep / eta component ####################################
start_time_2 = time.time()
eta_component = np.zeros((Neta, Nens, Ngamma))
psi_component = np.zeros((Neta, nt, Ngamma))
eigvecs_eta = np.zeros((Neta, nt, Ngamma))
eigvals_eta = np.zeros((Neta, nt))
eta_time_2 = np.transpose(eta_time, (1, 0, 2)) #(Nens, nt, Neta)
eta_time_test_2 = np.transpose(eta_time_test, (1, 0, 2)) #(test, nt, Neta)
eta_component_test = np.zeros((Neta, test_size, Ngamma))

for i in range(Neta):
    if (i == 0):
        start_time = time.time()
    eta_component_i, psi_component_i, eta_component_test_i, eigvals_eta_i, eigvecs_eta_i, _, _ = get_xi(eta_time_2[:, :, i], eta_time_test_2[:, :, i], Nxi = Ngamma, word = 'eta', number = str(i))
    eta_component[i] = eta_component_i
    psi_component[i] = psi_component_i
    eta_component_test[i] = eta_component_test_i
    eigvals_eta[i] = eigvals_eta_i
    eigvecs_eta[i] = eigvecs_eta_i
    if (i == 0):
        print("KL Expansion of one component: --- %s seconds ---" % (time.time() - start_time))
np.save(f'Pressure_weight/gamma_eta_Neta{Neta}_Ngamma{Ngamma}_NR{NR}.npy', eta_component)
np.save(f'Pressure_weight/psi_eta_Neta{Neta}_Ngamma{Ngamma}_NR{NR}.npy', psi_component)
np.save(f'Pressure_weight/gamma_eta_test_Neta{Neta}_Ngamma{Ngamma}_NR{NR}.npy', eta_component_test)
np.save(f'Pressure_weight/eigvals_eta_Neta{Neta}_NR{NR}.npy', eigvals_eta)
np.save(f'Pressure_weight/eigvecs_eta_Neta{Neta}_NR{NR}.npy', eigvecs_eta)
print("KL Expansion of every eta component: --- %s seconds ---" % (time.time() - start_time_2))

"""
######################################################## TIME KL (USING ETA) ####################################################
# Prediction for Testing Set
p_test_pred = np.zeros((test_size, nt, Nxy))
#eta_rec_test = np.zeros((nt, test_size, Neta))
#eta_time_mean = np.mean(eta_time, axis = 1) #(nt, Neta)

for i in range(nt):
    p_test_pred[:, i, :] = p_mean[i, :] + (psi_time[i, :, :Neta] @ eta_time_test[i, :, :].T).T
    
# Calculate Errors - TESTING
errors_linear_test = np.zeros((test_size, 2))
for i, test in enumerate(test_case):
    u_true = p_test[i]
    u_regres = p_test_pred[i]
    error = (u_regres - u_true)**2
    error_mean = (p_mean - u_true)**2
    errors_linear_test[i, 0] = np.sqrt(np.mean(error))
    errors_linear_test[i, 1] = np.sqrt(np.mean(error_mean))
avg_errors_test = np.mean(errors_linear_test, axis=0)
np.save(f'Pressure/errors_p_reconst_etas_Neta{Neta}_NR{NR}_test.npy', errors_linear_test)
# Plotting TESTING Errors
plt.figure(dpi=300)
plt.scatter(test_case-1, errors_linear_test[:, 1], label='Average', color='blue', marker='o')
plt.scatter(test_case-1, errors_linear_test[:, 0], label='Regression', color='red', marker='x')
plt.xlabel('Test Case')
plt.ylabel('Pressure RMSE')
plt.legend()
plt.savefig(f'Pressure/errors_p_reconst_ETAS_Neta{Neta}_NR{NR}_test.png')
plt.close()
del p_test_pred

print(f'Using etas: {errors_linear_test}')

# Prediction for Training Set
p_train_pred = np.zeros((train_size, nt, Nxy))
#eta_rec_test = np.zeros((nt, test_size, Neta))
#eta_time_mean = np.mean(eta_time, axis = 1) #(nt, Neta)

for i in range(nt):
    p_train_pred[:, i, :] = p_mean[i, :] + (psi_time[i, :, :Neta] @ eta_time[i, :, :].T).T
    
# Calculate Errors - TESTING
errors_linear_train = np.zeros((train_size, 2))
for i, test in enumerate(train_case):
    u_true = p_train[i]
    u_regres = p_train_pred[i]
    error = (u_regres - u_true)**2
    error_mean = (p_mean - u_true)**2
    errors_linear_train[i, 0] = np.sqrt(np.mean(error))
    errors_linear_train[i, 1] = np.sqrt(np.mean(error_mean))
avg_errors_test = np.mean(errors_linear_train, axis=0)
np.save(f'Pressure/errors_p_reconst_etas_Neta{Neta}_NR{NR}_train.npy', errors_linear_train)
# Plotting TRAIN Errors
plt.figure(dpi=300)
plt.scatter(train_case-1, errors_linear_train[:, 1], label='Average', color='blue', marker='o')
plt.scatter(train_case-1, errors_linear_train[:, 0], label='Regression', color='red', marker='x')
plt.xlabel('Train Case')
plt.ylabel('Pressure RMSE')
plt.legend()
plt.savefig(f'Pressure/errors_p_reconst_ETAS_Neta{Neta}_NR{NR}_train.png')
plt.close()
del p_train_pred

#print(f'Using etas: {errors_linear_test}')

"""

######################################################## MULTI KL (USING GAMMA) ####################################################
# Prediction for Testing Set
p_test_pred = np.zeros((test_size, nt, Nxy))
eta_rec_test = np.zeros((nt, test_size, Neta))
eta_time_mean = np.mean(eta_time, axis = 1) #(nt, Neta)

for j in range(Neta):
    eta_rec_test[:, :, j] = (eta_time_mean[:, j] + (psi_component[j, :, :] @ eta_component_test[j, :, :].T).T).T   

for i in range(nt):
    p_test_pred[:, i, :] = p_mean[i, :] + (psi_time[i, :, :Neta] @ eta_rec_test[i, :, :].T).T
    
# Calculate Errors - TESTING
errors_linear_test = np.zeros((test_size, 2))
for i, test in enumerate(test_case):
    u_true = p_test[i]
    u_regres = p_test_pred[i]
    error = (u_regres - u_true)**2
    error_mean = (p_mean - u_true)**2
    weighted_error = np.sum(error * weights_t) / total_weights_t
    weighted_error_mean = np.sum(error_mean * weights_t) / total_weights_t
    errors_linear_test[i, 0] = np.sqrt(weighted_error)
    errors_linear_test[i, 1] = np.sqrt(weighted_error_mean)
avg_errors_test = np.mean(errors_linear_test, axis=0)
np.save(f'Pressure_weight/errors_p_reconst_Neta{Neta}_Ngamma{Ngamma}_NR{NR}_test.npy', errors_linear_test)
# Plotting TESTING Errors
plt.figure(dpi=300)
plt.scatter(test_case, errors_linear_test[:, 1], label='Average', color='blue', marker='o')
plt.scatter(test_case, errors_linear_test[:, 0], label='Regression', color='red', marker='x')
plt.xlabel('Test Case')
plt.ylabel('Pressure RMSE')
plt.legend()
plt.savefig(f'Pressure_weight/errors_p_reconst_Neta{Neta}_Ngamma{Ngamma}_NR{NR}_test.png')
plt.close()
print(f' Weighted: {np.mean(errors_linear_test, axis = 0)}')

# Calculate Errors - TESTING
errors_linear_test = np.zeros((test_size, 2))
for i, test in enumerate(test_case):
    u_true = p_test[i]
    u_regres = p_test_pred[i]
    error = (u_regres - u_true)**2
    error_mean = (p_mean - u_true)**2
    errors_linear_test[i, 0] = np.sqrt(np.mean(error))
    errors_linear_test[i, 1] = np.sqrt(np.mean(error_mean))
avg_errors_test = np.mean(errors_linear_test, axis=0)
#np.save(f'Pressure_weight/errors_p_reconst_Neta{Neta}_Ngamma{Ngamma}_NR{NR}_test.npy', errors_linear_test)
# Plotting TESTING Errors
plt.figure(dpi=300)
plt.scatter(test_case, errors_linear_test[:, 1], label='Average', color='blue', marker='o')
plt.scatter(test_case, errors_linear_test[:, 0], label='Regression', color='red', marker='x')
plt.xlabel('Test Case')
plt.ylabel('Pressure RMSE')
plt.legend()
#plt.savefig(f'Pressure_weight/errors_p_reconst_Neta{Neta}_Ngamma{Ngamma}_NR{NR}_test.png')
plt.close()
print(f' Unweighted: {np.mean(errors_linear_test, axis = 0)}')


del p_test_pred

