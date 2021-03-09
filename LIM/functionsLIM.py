from sklearn.linear_model import LinearRegression
from statsmodels.graphics import tsaplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
import numpy as np
import os

###########################################################

def center(data):
    return data - np.mean(data, axis=0)

###########################################################

def dlmread(fname, n_run):
    
    data_list = []
    fpath = os.getcwd() + '/../' + fname
    fid = open(fpath,'r')
    #### Read the full data ####
    for i in range(n_run):
        a_str = fid.readline()
        a_str_split = a_str.split()
        a_num = [float(k) for k in a_str_split]
        data_list.append(a_num)
        
    fid.close()
    
    #### Changing the data from list to array ####
    data_arr = np.array(data_list)
    del data_list
    
    return data_arr

###########################################################

def trainingData(data, n_train, normalize):
    
    #### Now partition the training and test datasets ####
    train_data = data[:n_train, :]

    if normalize=='standard':
        #### Standardize the training dataset ####
        scaler = StandardScaler()
        scaler.fit(train_data)
        
    elif normalize=='minmax':
        scaler = MinMaxScaler()
        scaler.fit(train_data)
        
    elif normalize=='standard_top_pc':
        scaler = StandardScaler()
        scaler.fit(np.tile(train_data[:,0].reshape(-1,1), 
                           (1, np.shape(train_data)[1])))
    else:
        raise ValueError('normalization type not recognised')

    train_data = scaler.transform(train_data)
    
    return train_data, scaler

################################################################

def RMSE(k,y_true,y_pred,eofs,n_maxlead):
    psi1_true = y_true[k+1:k+n_maxlead+1,:].dot(eofs)
    psi1_pred = y_pred[k,:,:].dot(eofs)
    return np.sqrt(np.mean((psi1_true - psi1_pred)**2, axis=1))

################################################################