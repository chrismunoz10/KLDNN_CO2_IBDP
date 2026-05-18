
import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import time as time
import scipy.linalg as spla
import numpy.linalg as la
import os
import sys
import pandas as pd

#%% Input Data
#change them
#total ensembled sample size
NR = 80
Nxi = 18
Neta = 16

#Neural Network
N_NN = 1 #number of NN training for UQ
n_epoch = 12_000

# hidden_size = 36
batch_size = 4 #
learning_rate = 1e-3

file_ensemble =  '/home/camunoz4/IBDP_Files/sats.npy'
test_case = np.arange(10, NR + 1, step = 10, dtype = int)
ens_case = np.arange(1, NR + 1, step = 1, dtype = int)
train_case = np.delete(ens_case, test_case - 1)
test_size = len(test_case)
train_size = NR - test_size
if NR == 20:
    p_1 = np.load(file_ensemble)[-NR:]
else:
    p_1 = np.load(file_ensemble)[:NR]
print(p_1.shape)
p_1[p_1 == -99] = 0.0

Nens, nx, ny, nz, nt = p_1.shape
p_1 = np.reshape(p_1, (Nens, -1))


#Split into train / test
p_1_train = np.delete(p_1, test_case - 1, axis = 0) #(Ntrain, Nxyz)
p_1_test = p_1[test_case - 1] #(Ntrain, Nxyz)
p_1_mean = np.mean(p_1_train, axis = 0) #(Nzxy)

#p_vw_train = np.delete(p_vw, test_case - 1, axis = 0) #(Ntrain, Nxyz)
#p_vw_test = p_vw[test_case - 1] #(Ntrain, Nxyz)
#p_vw_mean = np.mean(p_vw_train, axis = 0) #(Nzxy)
#Inputs
#Inputs for all variables at once
xi_train = np.load(f'inputs_all/xi_all_Nxi{Nxi}_NR{NR}.npy')
xi_test = np.load(f'inputs_all/xi_all_test_Nxi{Nxi}_NR{NR}.npy')
"""
xi_train_xy = np.load(f'inputs/xi_xy_Nxi{Nxi}_NR{NR}.npy')
xi_test_xy = np.load(f'inputs/xi_xy_test_Nxi{Nxi}_NR{NR}.npy')
xi_train_z = np.load(f'inputs/xi_z_Nxi{Nxi}_NR{NR}.npy')
xi_test_z = np.load(f'inputs/xi_z_test_Nxi{Nxi}_NR{NR}.npy')
xi_train_por = np.load(f'inputs/xi_por_Nxi{Nxi}_NR{NR}.npy')
xi_test_por = np.load(f'inputs/xi_por_test_Nxi{Nxi}_NR{NR}.npy')
"""
#Outputs
eta_train = np.load(f'outputs/xi_s_Nxi{Neta}_NR{NR}.npy')
eta_test = np.load(f'outputs/xi_s_test_Nxi{Neta}_NR{NR}.npy')
psi_1 = np.load(f'outputs/psi_s_Nxi{Neta}_NR{NR}.npy')

eigvals_1 = np.load(f'outputs/eigvals_s_NR{NR}.npy')
# ############################################################### DNN xi -> eta ###################################################
# Construct Neural Network
class NN(torch.nn.Module):
    def __init__(self,input_n,hidden1_n,hidden2_n,hidden3_n,output_n):
        super(NN, self).__init__()  
        self.input = torch.nn.Linear(input_n,hidden1_n)    # Input layer
        self.hidden1 = torch.nn.Linear(hidden1_n,hidden2_n)  #Hidden layer
        self.hidden2 = torch.nn.Linear(hidden2_n,hidden3_n)  #Hidden layer
        self.predict = torch.nn.Linear(hidden3_n,output_n)   #Output layer
    
    def forward(self,x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        y = self.predict(x)
        return y  

############################################################### Simple DNN with L2 Regularization ###################################################      
class DNN:
    def __init__(self, xi_train, xi_test, eta_component_train, eta_component_test, hidden_size, lambda_reg, weights_eta, file_eta_component_train, file_eta_component_test, NR = NR, N_NN = N_NN, batch_size = batch_size, n_epoch = n_epoch, learning_rate = learning_rate):
        self.NR = NR
        self.N_NN = N_NN
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        
        self.xi_train = xi_train
        self.xi_test = xi_test
        self.eta_component_train = eta_component_train
        self.eta_component_test = eta_component_test
        
        self.train_size = self.xi_train.shape[0]
        self.test_size = self.xi_test.shape[0]
        
        self.Nxi_dnn = xi_train.shape[1]
        self.Neta_Ngamma_dnn = eta_component_train.shape[1]
        
        # self.Nu = Nu # Total number of nodes in mesh (space-time)
        self.lambda_reg = lambda_reg
        self.weights_eta = weights_eta
        # L2 regularization is handled by weight_decay in the optimizer
    
    def weighted_mse_loss(output, target, weight):
        return ((weight * (output - target)) ** 2).mean()
        
    def do_DNN(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        x_train = self.xi_train.copy()
        y_train = self.eta_component_train.copy()
        x_test = self.xi_test.copy()
        y_real = self.eta_component_test.copy()

        input_dim = x_train.shape[1]
        output_dim = y_train.shape[1]
        
        # Convert data to tensors
        x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_real = torch.tensor(y_real, dtype=torch.float32).to(device)
        weights_eta = torch.tensor(self.weights_eta, dtype = torch.float32).to(device)
        # Initialize ensemble containers
        ensemble_eta_component_train = np.zeros((self.N_NN, self.train_size, self.Neta_Ngamma_dnn))
        ensemble_eta_component_test = np.zeros((self.N_NN, self.test_size, self.Neta_Ngamma_dnn))
        mid_size_1 = (input_dim + self.hidden_size)//2
        mid_size_2 = (output_dim + self.hidden_size)//2
        # Define the model architecture
        net = NN(input_dim, mid_size_1, self.hidden_size, mid_size_2, output_dim)
        net.to(device)
        total_params = sum(p.numel() for p in net.parameters())
        print(f"Total number of parameters: {total_params}")
        
        # Loop for N trainings of the NN
        for n in range(self.N_NN):
            torch.manual_seed(n)
            np.random.seed(n)
            
            print(f'Start Training {n+1} / {self.N_NN}')
            net = NN(input_dim, mid_size_1, self.hidden_size, mid_size_2, output_dim)
            net.to(device)
            
            # Using Adam optimizer with L2 regularization (weight decay)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.lambda_reg)
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.n_epoch//2)
            criterion = torch.nn.MSELoss()
            
            loss_train = []
            loss_test = []
            for e in range(self.n_epoch):
                
                for k in range(0, self.train_size, self.batch_size):
                    net.train()
                    optimizer.zero_grad()
                    x = x_train[k:k + self.batch_size, :]
                    y = y_train[k:k + self.batch_size, :]
                    y_pred_train = net(x)
                    
                    # Compute loss
                    loss = criterion(y_pred_train, y)
                    #loss = DNN.weighted_mse_loss(y_pred_train, y, weights_eta)
                    loss.backward()
                    optimizer.step()
                    
                
                scheduler.step()
                loss_train.append(loss.item())
                
                with torch.no_grad():
                    net.eval()
                    y_pred_test = net(x_test)
                    loss_test_val=criterion(y_pred_test, y_real).detach().cpu().numpy()
                    loss_test.append(loss_test_val)
                
                if e % 100 == 0:
                    print(f'Epoch: {e:03d}, Train Loss: {loss:.4f}, Test Loss:{loss_test_val:.4f}')
            print('Training Done')
            # Plot loss curves
            plt.figure(figsize=(8,6))
            plt.title(f'Hidden Layer size = {self.hidden_size}')
            plt.semilogy(np.arange(self.n_epoch), np.array(loss_train), label='Train loss')
            plt.semilogy(np.arange(self.n_epoch), np.array(loss_test), label='Test loss')
            plt.xlabel('# of epochs')
            plt.ylabel('Loss')
            plt.legend(loc='best')
            plt.grid(True)
            plt.savefig(f'DNN_s_all_inputs/Loss_curve_Nxi{self.Nxi_dnn}_NetaNgamma{self.Neta_Ngamma_dnn}_hs{self.hidden_size}_NR{self.NR}_lambdareg{self.lambda_reg}_{n}.png')
            
            # Test prediction
            y_pred_test_np = net(x_test).detach().cpu().numpy()
            print(f'eta_component prediction shape: {y_pred_test_np.shape}')
            print(loss_test[-1])

            # Train prediction
            y_pred_train_np = net(x_train).detach().cpu().numpy()
            print(f'eta_component prediction shape: {y_pred_train_np.shape}')
            print(loss_train[-1])
            
            ensemble_eta_component_train[n] = y_pred_train_np
            ensemble_eta_component_test[n] = y_pred_test_np
            
            print(f'Done Training {n+1} / {self.N_NN}\n')

        # Save the trained model
        model_filename = f'DNN_s_all_inputs/model_NR{self.NR}_Netadnn{self.Neta_Ngamma_dnn}_{n}.pth'
        torch.save(net.state_dict(), model_filename)
        print(f'Model saved to {model_filename}')
        
        # Save Ensembles
        eta_component_train_mean = np.mean(ensemble_eta_component_train, axis=0)
        eta_component_test_mean = np.mean(ensemble_eta_component_test, axis=0)
        np.save(file_eta_component_train, ensemble_eta_component_train)
        np.save(file_eta_component_test, ensemble_eta_component_test)
           
##############################################################################################################################################################
results = []
#Conductivity Latent Variables:
# Nxi_new_list = [4]
# Neta_dnn_list= [1,2]
# Ngamma_dnn_list= [3]
# hidden_size_list=[24]
# lambda_reg_list=[0]
Nxi_new_list = [18]
Neta_dnn_list= [6]
hidden_size_list=[48, 60]
lambda_reg_list=[0.01, 0.015, 0.02]
for lambda_reg in lambda_reg_list:
    for hidden_size in hidden_size_list:
        for Nxi_new in Nxi_new_list:
            for Neta_dnn in Neta_dnn_list:
                #weights_eta = (Neta_dnn*np.sqrt(eigvals_1[:int(Neta_dnn)])/np.sum(np.sqrt(eigvals_1[:Neta_dnn])))
                weights_eta = np.ones(Neta_dnn)
                print(weights_eta)
                Nxi_dnn=Nxi_new
                label = f"{lambda_reg}_{hidden_size}_{Nxi_dnn}_{Neta_dnn}"
                print(f'inputsize{Nxi_dnn}_Netadnn{Neta_dnn}_hiddensize{hidden_size}_lambdareg{lambda_reg}')
                
                #input preprocessing 
                """
                xi_train = np.concatenate((xi_train_xy[:, :Nxi_new], xi_train_z[:, :Nxi_new], xi_train_por[:, :Nxi_new]), axis = 1)
                xi_test = np.concatenate((xi_test_xy[:, :Nxi_new], xi_test_z[:, :Nxi_new], xi_test_por[:, :Nxi_new]), axis = 1)
                """
                print('xi_train.shape',xi_train.shape)
                print('xi_test.shape',xi_test.shape)
                xi_train = xi_train.reshape((train_size, -1))
                xi_test = xi_test.reshape((test_size, -1))
                
                
                #output preprocessing
                #eta_train = np.concatenate((eta_train_1[:, :Neta_dnn], eta_train_vw[:, :Neta_dnn]), axis = 1)
                #eta_test = np.concatenate((eta_test_1[:, :Neta_dnn], eta_test_vw[:, :Neta_dnn]), axis = 1)
                eta_train = eta_train[:, :Neta_dnn].reshape((train_size, -1))
                eta_test = eta_test[:, :Neta_dnn].reshape((test_size, -1))
                print('eta_train.shape',eta_train.shape)
                print('eta_test.shape',eta_test.shape) 
                
                #Use DNN class
                file_eta_component_train = f'DNN_s_all_inputs/pred_eta_component_Nxi{Nxi_dnn}_Neta{Neta_dnn}_hs{hidden_size}_NR{NR}_lambdareg{lambda_reg}_train.npy'	
                file_eta_component_test = f'DNN_s_all_inputs/pred_eta_component_Nxi{Nxi_dnn}_Neta{Neta_dnn}_hs{hidden_size}_NR{NR}_lambdareg{lambda_reg}_test.npy'
                model = DNN(xi_train, xi_test, eta_train, eta_test, hidden_size, lambda_reg, weights_eta, file_eta_component_train, file_eta_component_test) 
                model.do_DNN()
                eta_train_pred = np.mean(np.load(file_eta_component_train), axis = 0)  #(train_size,Neta_dnn)
                eta_test_pred = np.mean(np.load(file_eta_component_test), axis = 0)
                
                # eta_components for Test Set
                errors_eta_component_test = np.zeros((test_size))
                for i, test in enumerate(test_case):
                    u_true = eta_test[i]
                    u_regres = eta_test_pred[i]
                    error = (u_regres - u_true)**2
                    errors_eta_component_test[i] = np.sqrt(np.mean(error))
                avg_eta_component_errors_test = np.mean(errors_eta_component_test, axis=0)

                for i, test in enumerate(test_case):
                    plt.figure()
                    plt.scatter(np.arange(len(eta_test[i])), eta_test[i], marker='o', label='True')
                    plt.scatter(np.arange(len(eta_test_pred[i])), eta_test_pred[i], marker='x', label='Prediction')
                    plt.legend()
                    plt.title('Eta_component Comparison')
                    plt.savefig(f'DNN_s_all_inputs/eta_component_s_comparison_Nxi{Nxi_dnn}_Neta{Neta_dnn}_hs{hidden_size}_NR{NR}_lambdareg{lambda_reg}_test{test}.png', dpi=300)
                    plt.close() 

                # eta_components for Train Set
                errors_eta_component_train = np.zeros((train_size))
                for i, train in enumerate(train_case):
                    u_true = eta_train[i]
                    u_regres = eta_train_pred[i]
                    error = (u_regres - u_true)**2
                    errors_eta_component_train[i] = np.sqrt(np.mean(error))
                avg_eta_component_errors_train = np.mean(errors_eta_component_train, axis=0)

                plt.figure()
                plt.scatter(np.arange(len(eta_train[0])), eta_train[0], marker='o', label='True')
                plt.scatter(np.arange(len(eta_train_pred[0])), eta_train_pred[0], marker='x', label='Prediction')
                plt.legend()
                plt.title('Eta_component Comparison')
                plt.savefig(f'DNN_s_all_inputs/eta_component_s_comparison_Nxi{Nxi_dnn}_Neta{Neta_dnn}_hs{hidden_size}_NR{NR}_lambdareg{lambda_reg}_train.png', dpi=300)
                plt.close()
                
                # Prediction for Testing Set
                
                p_1_test_pred = p_1_mean + (psi_1[:, :Neta_dnn] @ eta_test_pred.T).T
                
                #np.save(f'DNN_{NR}_all_inputs/prediction_s_Nxi{Nxi_dnn}_Neta{Neta_dnn}_hs{hidden_size}_NR{NR}_lambdareg{lambda_reg}_test.npy', p_1_test_pred)
                
                
                #Prediction for Training Set
                p_1_train_pred = p_1_mean + (psi_1[:, :Neta_dnn] @ eta_train_pred.T).T
                
                #np.save(f'DNN_{NR}_all_inputs/prediction_s_Nxi{Nxi_dnn}_Neta{Neta_dnn}_hs{hidden_size}_NR{NR}_lambdareg{lambda_reg}_train.npy', p_1_train_pred)
                
                # Calculate Errors - TESTING CCS1
                errors_linear_test = np.zeros((test_size, 2))
                for i, test in enumerate(test_case):
                    u_true = p_1_test[i]
                    u_regres = p_1_test_pred[i]
                    error = (u_regres - u_true)**2
                    error_mean = (p_1_mean - u_true)**2
                    errors_linear_test[i, 0] = np.sqrt(np.mean(error))
                    errors_linear_test[i, 1] = np.sqrt(np.mean(error_mean))
                avg_errors_test = np.mean(errors_linear_test, axis=0)
                np.save(f'DNN_s_all_inputs/errors_s_Nxi{Nxi_dnn}_Neta{Neta_dnn}_hs{hidden_size}_NR{NR}_lambdareg{lambda_reg}_test.npy', errors_linear_test)
                # Plotting TESTING Errors
                plt.figure(dpi=300)
                plt.scatter(test_case-1, errors_linear_test[:, 1], label='Average', color='blue', marker='o')
                plt.scatter(test_case-1, errors_linear_test[:, 0], label='DNN', color='red', marker='x')
                plt.xlabel('Test Case')
                plt.ylabel('Pressure RMSE')
                plt.legend()
                plt.savefig(f'DNN_s_all_inputs/errors_s_Nxi{Nxi_dnn}_Neta{Neta_dnn}_hs{hidden_size}_NR{NR}_lambdareg{lambda_reg}_test.png')
                plt.close()
                
                 # Calculate Errors - TRAINING CCS1
                errors_linear_test = np.zeros((train_size, 2))
                for i, test in enumerate(train_case):
                    u_true = p_1_train[i]
                    u_regres = p_1_train_pred[i]
                    error = (u_regres - u_true)**2
                    error_mean = (p_1_mean - u_true)**2
                    errors_linear_test[i, 0] = np.sqrt(np.mean(error))
                    errors_linear_test[i, 1] = np.sqrt(np.mean(error_mean))
                avg_errors_test = np.mean(errors_linear_test, axis=0)
                np.save(f'DNN_s_all_inputs/errors_s_Nxi{Nxi_dnn}_Neta{Neta_dnn}_hs{hidden_size}_NR{NR}_lambdareg{lambda_reg}_train.npy', errors_linear_test)
                # Plotting TESTING Errors
                plt.figure(dpi=300)
                plt.scatter(train_case-1, errors_linear_test[:, 1], label='Average', color='blue', marker='o')
                plt.scatter(train_case-1, errors_linear_test[:, 0], label='DNN', color='red', marker='x')
                plt.xlabel('Train Case')
                plt.ylabel('Pressure RMSE')
                plt.legend()
                plt.savefig(f'DNN_s_all_inputs/errors_s_Nxi{Nxi_dnn}_Neta{Neta_dnn}_hs{hidden_size}_NR{NR}_lambdareg{lambda_reg}_train.png')
                plt.close()

                #apend to results list
                #results.append([label, avg_errors_train[0],avg_errors_train[1],avg_errors_test[0],avg_errors_test[1],avg_eta_component_errors_train,avg_eta_component_errors_test])
#print(weights_eta)

"""               
# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Label", "Train_DNN_RMSE", "Train_Avg_RMSE", "Test_DNN_RMSE", "Test_Avg_RMSE","Train_eta_component_RMSE","Test_eta_component_RMSE"])

# Write to Excel file
results_df.to_excel("p_errors.xlsx", index=False)
print("Error metrics saved to 'p_errors.xlsx'")
"""