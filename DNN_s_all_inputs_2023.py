
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
if NR == 80:
    Nxi = 18
else: 
    Nxi = 9
Neta = 6

#Neural Network
N_NN = 1 #number of NN training for UQ
n_epoch = 12_000

# hidden_size = 36
batch_size = 4 #
learning_rate = 1e-3

test_case = np.arange(10, NR + 1, step = 10, dtype = int)
ens_case = np.arange(1, NR + 1, step = 1, dtype = int)
train_case = np.delete(ens_case, test_case - 1)
test_size = len(test_case)
train_size = NR - test_size

nx, ny, nz, nt = [40, 44, 94, 50]

s_mean = np.load(f'sats/s_mean_NR{NR}.npy')

#Inputs
#Inputs for all variables at once
xi_train = np.load(f'inputs/xi_all_Nxi{Nxi}_NR{NR}.npy')
xi_test = np.load(f'inputs/xi_all_test_Nxi{Nxi}_NR{NR}.npy')

#Outputs
eta_train = np.load(f'sats/xi_s_Nxi{Neta}_NR{NR}.npy')
eta_test = np.load(f'sats/xi_s_test_Nxi{Neta}_NR{NR}.npy')
psi_1 = np.load(f'sats/psi_s_Nxi{Neta}_NR{NR}.npy')

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
hidden_size = 60
lambda_reg = 0.02
Neta_dnn = 6
if NR == 80:
    Nxi_dnn = 18
else:
    Nxi_dnn = 9

weights_eta = np.ones(Neta_dnn)

print(f'inputsize{Nxi_dnn}_Netadnn{Neta_dnn}_hiddensize{hidden_size}_lambdareg{lambda_reg}')

#input preprocessing 
xi_train = xi_train.reshape((train_size, -1))
xi_test = xi_test.reshape((test_size, -1))

#output preprocessing
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


# Prediction for Testing Set

s_test_pred = s_mean + (psi_1[:, :Neta_dnn] @ eta_test_pred.T).T #(test_size, nt, Nxyz)

s_test_pred =  np.reshape(s_test_pred, (test_size, nx, ny, nz, nt))