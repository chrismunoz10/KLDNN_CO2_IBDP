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

file_ensemble = '' #shape: (Nens, nx, ny, nz, nt)
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

eta_time = np.zeros((nt, Nens, Neta))
psi_time = np.zeros((nt, Nxy, Neta))
eigvecs_time = np.zeros((nt, Nxy, Neta))
eigvals_time = np.zeros((nt, train_size))
eta_time_test = np.zeros((nt, test_size, Neta))

for l in range(nt):
    start_nt = time.time()
    eta_time_l, psi_time_l, eta_time_test_l, eigvals_time_l, eigvecs_time_l, _, _ = get_xi_weights(p_train[:, l, :], p_test[:, l, :], weights, Nxi = Neta, word = 'time', number = str(l))
    eta_time[l] = eta_time_l
    psi_time[l] = psi_time_l
    eta_time_test[l] = eta_time_test_l
    eigvals_time[l] = eigvals_time_l


############################# Do KL Expansion on each Timestep / eta component ####################################

eta_component = np.zeros((Neta, Nens, Ngamma))
psi_component = np.zeros((Neta, nt, Ngamma))
eigvecs_eta = np.zeros((Neta, nt, Ngamma))
n2_eigvals_eta = min(nt, Nens)
eigvals_eta = np.zeros((Neta, n2_eigvals_eta))
eta_time_2 = np.transpose(eta_time, (1, 0, 2)) #(Nens, nt, Neta)
eta_time_test_2 = np.transpose(eta_time_test, (1, 0, 2)) #(test, nt, Neta)
eta_component_test = np.zeros((Neta, test_size, Ngamma))

for i in range(Neta):
    eta_component_i, psi_component_i, eta_component_test_i, eigvals_eta_i, eigvecs_eta_i, _, _ = get_xi(eta_time_2[:, :, i], eta_time_test_2[:, :, i], Nxi = Ngamma, word = 'eta', number = str(i))
    eta_component[i] = eta_component_i
    psi_component[i] = psi_component_i
    eta_component_test[i] = eta_component_test_i
    eigvals_eta[i] = eigvals_eta_i
    eigvecs_eta[i] = eigvecs_eta_i

np.save(f'Pressure_weight/gamma_eta_Neta{Neta}_Ngamma{Ngamma}_NR{NR}.npy', eta_component)
np.save(f'Pressure_weight/psi_eta_Neta{Neta}_Ngamma{Ngamma}_NR{NR}.npy', psi_component)
np.save(f'Pressure_weight/gamma_eta_test_Neta{Neta}_Ngamma{Ngamma}_NR{NR}.npy', eta_component_test)
np.save(f'Pressure_weight/eigvals_eta_Neta{Neta}_NR{NR}.npy', eigvals_eta)
np.save(f'Pressure_weight/eigvecs_eta_Neta{Neta}_NR{NR}.npy', eigvecs_eta)


######################################################## MULTI KL (USING GAMMA) ####################################################
# Prediction for Testing Set
p_test_pred = np.zeros((test_size, nt, Nxy))
eta_rec_test = np.zeros((nt, test_size, Neta))
eta_time_mean = np.mean(eta_time, axis = 1) #(nt, Neta)

for j in range(Neta):
    eta_rec_test[:, :, j] = (eta_time_mean[:, j] + (psi_component[j, :, :] @ eta_component_test[j, :, :].T).T).T   

for i in range(nt):
    p_test_pred[:, i, :] = p_mean[i, :] + (psi_time[i, :, :Neta] @ eta_rec_test[i, :, :].T).T
    

