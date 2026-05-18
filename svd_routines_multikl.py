

import matplotlib.pyplot as plt
import math
import numpy as np
import time as time
import scipy.linalg as spla
import scipy.sparse.linalg as sspla
import numpy.linalg as la
import os
import torch

###################################### Eigendecomposition of Covariance ##############################################
def get_eigen(data, reg = 0.0, variable = 'k', word = 'layer', number = '0'):
    file_k_eigval = f'{variable}_eigvals_{word}_{number}.npy'
    file_k_eigvecs = f'{variable}_eigvecs_{word}_{number}.npy'
    mean = np.mean(data, axis = 0)
    train_size = data.shape[0]
    Nxy = data.shape[1]
    #Decide if Eigendecomposition has to be done
    eigen_done = 0 #file_k_eigval in self.files) and (file_k_eigvecs in self.files)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('Start eigendecomp')
    if eigen_done:
        print(f'Loading Eigendecomposition of {word} {number}')
        eigval_k = np.load(file_k_eigval)
        eigvecs_k = np.load(file_k_eigvecs)
    else:
        #print(f'Starting Eigendecomposition of {word} {number}')
        #Do eigendecomposition of k
        M = data - mean
        del data
        #print(f'M: {M.shape}')
        M = torch.tensor(M, dtype=torch.float32).to(device)

        #Eigen decomposition of k
        start_time = time.time()

        q_size = int(min(train_size, Nxy))

        if variable == 'k':
            U, s, V = torch.svd_lowrank(M, q = q_size, niter = 4)
            s = s.detach().cpu().numpy()
            V = V.detach().cpu().numpy()
            s_old = s.copy()
            eigval = s_old**2 / (train_size - 1)
            psi_old = np.sqrt(eigval[:10]) * V[:,:10]
            
        else:
            U, s, V = torch.linalg.svd(M)
            s = s.detach().cpu().numpy()
            V = V.detach().cpu().numpy()
            
        #Add regularization
        if reg > 0.0:
            print('Adding Regularization')
            s_size = len(s)
            s_new = np.zeros((train_size, train_size))
            s_new[:s_size, :s_size] = np.diag(s) + reg * np.eye(s_size)
            U = U.detach().cpu().numpy()
            
            M = U @ s_new @ V.T
            #np.dot(U, np.dot(s_new, V.T))
            print(M.shape)
            M = torch.tensor(M, dtype=torch.float32).to(device)
            U, s, V = torch.svd_lowrank(M, q = q_size, niter = 3)
            s = s.detach().cpu().numpy()
            V = V.detach().cpu().numpy()
        
        eigval = s**2 / (train_size - 1)
        eigvecs = V
        #psi_new = np.sqrt(eigval[:10]) * V[:,:10]
        #print(np.max(psi_new - psi_old))
        idx_sort = eigval.argsort()[::-1]
        eigval_k = eigval[idx_sort]
        if Nxy > 10000:
            eigval_k = eigval_k[:train_size]
        
        eigvecs_k = eigvecs[:, idx_sort]
        if Nxy > 10000:
            eigvecs_k = eigvecs_k[:, :train_size] #only save up to NR eigenvectors to avoid large files
        #np.save(file_k_eigval, eigval_k)
        #np.save(file_k_eigvecs, eigvecs_k) 
        #print(f'Files saved for Eigendecomposition of {variable}')
        
    return [eigval_k, eigvecs_k, mean]

#######################################################################################################################

###################################################### Optimization to find vector of xi #############################
def get_xi(data, data_test, Nxi = 2, reg = 0.0, variable = 'k', word = 'layer', number = '0'):
    #Load eigenvalues and eigenvecctors
    eigval_k, eigvecs_k, mean = get_eigen(data, reg, variable, word, number)
    #if word == 'component':
    #    print(f'Max: {np.max(eigval_k)} Min: {np.min(eigval_k)}')
    file_xis = f'{variable}_xi_Nxi{Nxi}_{word}_{number}.npy'
    file_psi = f'{variable}_psi_Nxi{Nxi}_{word}_{number}.npy'
    train_size = data.shape[0]
    Nxy = data.shape[1]
    test_size = data_test.shape[0]
    
    #Decide if Optimization has to be done
    optim_done = 0 #(file_xi_train in self.files) and (file_psi_k_train in self.files) and (file_xi_test in self.files)
    
    if optim_done:
        #print(f'Loading Optimization of {word} {number}')
        xi_train = np.load(file_xi_train)
        psi_k = np.load(file_psi_k_train)
        xi_test = np.load(file_xi_test)
        
    else:
        #Optimization to find xi vector for training
        psi_k = np.sqrt(eigval_k[:Nxi]) * eigvecs_k[:,:Nxi]
        rhs = np.zeros((Nxy, train_size))
        #if word == 'component':
        #    print(f'Max: {np.max(psi_k)} Min: {np.min(psi_k)}')
        for i in range(train_size):
            rhs[:,i] = data[i,:] - mean
        del data
        #print(psi_k.shape)
        #print(rhs.shape)
        xis, residuals_train = spla.lstsq(psi_k, rhs)[:2]
        xi_train = xis.T
        #print(np.max(residuals_k))
        #np.save(file_xis, xi_train)
        #np.save(file_psi, psi_k)
        
        #Optimization to find xi vector for testing
        rhs_test = np.zeros((Nxy, test_size))
        for i in range(test_size):
          rhs_test[:,i] = data_test[i,:] - mean
        del data_test  
        xi_test, residuals_test = spla.lstsq(psi_k, rhs_test)[:2]
        xi_test = xi_test.T
        #print(residual_test)
        #np.save(file_xi_test, xi_test)
        
        #print('Optimization for xi done')
        
    return [xi_train, psi_k, xi_test, eigval_k, eigvecs_k[:, :Nxi], residuals_train, residuals_test]

##########################################################################################################################

def get_eigen_weights(data, weights, reg = 0.0, variable = 'k', word = 'layer', number = '0'):
    file_k_eigval = f'{variable}_eigvals_{word}_{number}.npy'
    file_k_eigvecs = f'{variable}_eigvecs_{word}_{number}.npy'
    
    mean = np.mean(data, axis = 0)
    train_size = data.shape[0]
    Nxy = data.shape[1]
    #Decide if Eigendecomposition has to be done
    eigen_done = 0 #file_k_eigval in self.files) and (file_k_eigvecs in self.files)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    sqrt_w = np.sqrt(weights)
    weighted_data = (data.T * sqrt_w).T
    del data
    weighted_mean = np.mean(weighted_data, axis = 0)
    #print('Start eigendecomp')
    if eigen_done:
        print(f'Loading Eigendecomposition of {word} {number}')
        eigval_k = np.load(file_k_eigval)
        eigvecs_k = np.load(file_k_eigvecs)
    else:
        #print(f'Starting Eigendecomposition of {word} {number}')
        #Do eigendecomposition of k
        M = weighted_data - weighted_mean
        del weighted_data
        #print(f'M: {M.shape}')
        M = torch.tensor(M, dtype=torch.float32).to(device)
        #print(M.shape)
        #print('Tensor done')
        #Eigen decomposition of k
        start_time = time.time()
        #if Nxy > 10000:
        #    q_size = int(min(train_size, Nxy))
        #else:
        #    q_size = Nxy
        q_size = int(min(train_size, Nxy))
        #q_size = 3
        #print(f'q_size = {q_size}')
        if variable == 'k':
            U, s, V = torch.svd_lowrank(M, q = q_size, niter = 5)
            s = s.detach().cpu().numpy()
            V = V.detach().cpu().numpy()
            s_old = s.copy()
            eigval = s_old**2 / (train_size - 1)
            psi_old = np.sqrt(eigval[:10]) * V[:,:10]
            
        else:
            U, s, V = torch.linalg.svd(M)
            s = s.detach().cpu().numpy()
            V = V.detach().cpu().numpy()
            
        #Add regularization
        if reg > 0.0:
            print('Adding Regularization')
            s_size = len(s)
            s_new = np.zeros((train_size, train_size))
            s_new[:s_size, :s_size] = np.diag(s) + reg * np.eye(s_size)
            U = U.detach().cpu().numpy()
            
            M = U @ s_new @ V.T
            #np.dot(U, np.dot(s_new, V.T))
            print(M.shape)
            M = torch.tensor(M, dtype=torch.float32).to(device)
            U, s, V = torch.svd_lowrank(M, q = q_size, niter = 3)
            s = s.detach().cpu().numpy()
            V = V.detach().cpu().numpy()
        
        eigval = s**2 / (train_size - 1)
        eigvecs = V * (sqrt_w)**(-1)
        #psi_new = np.sqrt(eigval[:10]) * V[:,:10]
        #print(np.max(psi_new - psi_old))
        del M, V, U, s
        idx_sort = eigval.argsort()[::-1]
        eigval_k = eigval[idx_sort]
        if Nxy > 10000:
            eigval_k = eigval_k[:train_size]
        
        eigvecs_k = eigvecs[:, idx_sort]
        if Nxy > 10000:
            eigvecs_k = eigvecs_k[:, :train_size] #only save up to NR eigenvectors to avoid large files
        #np.save(file_k_eigval, eigval_k)
        #np.save(file_k_eigvecs, eigvecs_k) 
        #print(f'Files saved for Eigendecomposition of {variable}')
        
    return [eigval_k, eigvecs_k, mean]
    

def get_xi_weights(data, data_test, weights, Nxi = 2, reg = 0.0, variable = 'k', word = 'layer', number = '0'):
    #Load eigenvalues and eigenvectors
    start_time = time.time()
    eigval_k, eigvecs_k, mean = get_eigen_weights(data, weights, reg, variable, word, number)
    print("POD Decomposition: --- %s seconds ---" % (time.time() - start_time))
    #if word == 'component':
    #    print(f'Max: {np.max(eigval_k)} Min: {np.min(eigval_k)}')
    file_xis = f'{variable}_xi_Nxi{Nxi}_{word}_{number}.npy'
    file_psi = f'{variable}_psi_Nxi{Nxi}_{word}_{number}.npy'
    train_size = data.shape[0]
    Nxy = data.shape[1]
    test_size = data_test.shape[0]
    
    sqrt_w = np.sqrt(weights)
    weighted_data = (data.T * sqrt_w).T
    #weighted_data_test = (data_test.T * sqrt_w).T
    weighted_mean = np.mean(weighted_data, axis = 0)
    
    del data
    #del data_test
    del mean
    #Decide if Optimization has to be done
    optim_done = 0 #(file_xi_train in self.files) and (file_psi_k_train in self.files) and (file_xi_test in self.files)
    
    if optim_done:
        #print(f'Loading Optimization of {word} {number}')
        xi_train = np.load(file_xi_train)
        psi_k = np.load(file_psi_k_train)
        xi_test = np.load(file_xi_test)
        
    else:
        #Optimization to find xi vector for training
        psi_k = np.sqrt(eigval_k[:Nxi]) * eigvecs_k[:,:Nxi]
        weighted_psi = psi_k * sqrt_w
        rhs = np.zeros((Nxy, train_size))
        start_time = time.time()
        
        for i in range(train_size):
            rhs[:,i] = weighted_data[i,:] - weighted_mean
        del weighted_data
        print('Starting Training LS')
        xis, residuals_train = spla.lstsq(weighted_psi, rhs)[:2]
        print("Training LS Fitting: --- %s seconds ---" % (time.time() - start_time))
        del rhs
        xi_train = xis.T
        del xis
        #Optimization to find xi vector for testing
        weighted_data_test = (data_test.T * sqrt_w).T
        del data_test
        rhs_test = np.zeros((Nxy, test_size))
        print('rhs test done')
        start_time = time.time()
        for i in range(test_size):
          rhs_test[:,i] = weighted_data_test[i,:] - weighted_mean
        del weighted_mean
        del weighted_data_test
        
        xi_test, residuals_test = spla.lstsq(weighted_psi, rhs_test)[:2]
        print("Testing LS Fitting: --- %s seconds ---" % (time.time() - start_time))
        xi_test = xi_test.T
        #print(residual_test)
        #np.save(file_xi_test, xi_test)
        
        #print('Optimization for xi done')
        
    return [xi_train, weighted_psi, xi_test, eigval_k, eigvecs_k[:, :Nxi], residuals_train, residuals_test]
    
    
def get_xi_weights_batches(data, weights, psi_k, mean, Nxi = 2, reg = 0.0, variable = 'k', word = 'layer', number = '0'):
    #Load eigenvalues and eigenvecctors
    start_time = time.time()
    #if word == 'component':
    #    print(f'Max: {np.max(eigval_k)} Min: {np.min(eigval_k)}')
    file_xis = f'{variable}_xi_Nxi{Nxi}_{word}_{number}.npy'
    file_psi = f'{variable}_psi_Nxi{Nxi}_{word}_{number}.npy'
    train_size = data.shape[0]
    Nxy = data.shape[1]
    
    sqrt_w = np.sqrt(weights)
    weighted_data = (data.T * sqrt_w).T
    #weighted_data_test = (data_test.T * sqrt_w).T
    weighted_mean = np.mean(weighted_data, axis = 0)
    
    del data

    del mean
    #Decide if Optimization has to be done
    optim_done = 0 #(file_xi_train in self.files) and (file_psi_k_train in self.files) and (file_xi_test in self.files)
    
    if optim_done:
        #print(f'Loading Optimization of {word} {number}')
        xi_train = np.load(file_xi_train)
        psi_k = np.load(file_psi_k_train)
        xi_test = np.load(file_xi_test)
        
    else:
        #Optimization to find xi vector for training
        weighted_psi = psi_k * sqrt_w
        rhs = np.zeros((Nxy, train_size))
        start_time = time.time()
        
        for i in range(train_size):
            rhs[:,i] = weighted_data[i,:] - weighted_mean
        del weighted_data
        #print('Starting Training LS')
        xis, residuals_train = spla.lstsq(weighted_psi, rhs)[:2]
        #print("Training LS Fitting: --- %s seconds ---" % (time.time() - start_time))
        del rhs
        xi_train = xis.T
        del xis

        #print(residual_test)
        #np.save(file_xi_test, xi_test)
        
        #print('Optimization for xi done')
        
    return xi_train