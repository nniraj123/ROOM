from statsmodels.graphics import tsaplots
import math
import os
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import rmsprop, SGD, Adagrad, Adadelta

###########################################################

def dlmread(fname, n_run, n_pc):
    
    data_list = []
    fpath = os.getcwd() + '/../' + fname
    fid = open(fpath,'r')
    #### Read the full data ####
    for i in range(n_run):
        a_str = fid.readline()
        a_str_split = a_str.split()
        a_num = [float(k) for k in a_str_split]
        data_list.append(a_num[:n_pc])
        
    fid.close()
    
    #### Changing the data from list to array ####
    data_arr = np.array(data_list)
    del data_list
    
    return data_arr

######################################################################

def trainingdata(data, n_pc, n_train, normalize):
    
    # for modelling y(n) = ANN(y(n-1))
    x_train = np.zeros((n_train, n_pc))
    y_train = np.zeros((n_train, n_pc))

    #### Now partition the training and test datasets ####
    train_data = data[:n_train, :]

    if normalize=='standard':
        #### Standardize the training dataset ####
        scaler = StandardScaler()
    
    elif normalize=='minmax':
        scaler = MinMaxScaler()
        
    else:
        raise ValueError('normalization type not recognised')

    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    
    x_train = train_data[:-1,:]
    y_train = train_data[1:,:]
    
    return x_train, y_train, scaler
    
#######################################################################

def addhiddenlayer(model, npl, activation):
    model.add(Dense(npl, activation=activation))
    
def ANN(x, y, n_in_param, 
        activation='tanh', npl=100, 
        loss='mae', optimizer='adam',
        metric='mae',epochs=100,
        numhidlayers=4,validation_split=0.2):
    
    # This now the keras part. The network is 
    # "dense" with each neurons connected to each neuron of the previous
    # and following layer

    model = Sequential()
    model.add(Dense(n_in_param, input_dim=n_in_param, activation=activation))
    for k in range(numhidlayers):
        addhiddenlayer(model, npl, activation)
    model.add(Dense(n_in_param, activation='linear'))

    # Configuring the learning process
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    # This is the training part of the network:
    history = model.fit(x, y, epochs=epochs, validation_split=validation_split)
    
    # And here the network is stored:
    model.save_weights("./weights")

    return model, history
######################################################################

def forecast(n_pred,n_in_param,x_start,deltaT,nn_model,scaler,dW,isnoise):
    
    x_start = scaler.transform(x_start.reshape(1,-1))
    
    y_pred = np.zeros([n_pred, n_in_param])
        
    state = np.zeros((1, n_in_param))
    state[0,:] = x_start
    
    # This is the forecast loop
    for j in range(n_pred):
        if isnoise:
            # Use the neural network and noise to predict the rhs
            state = nn_model.predict(state, batch_size=1) + dW[j,:]
        else:
            state = nn_model.predict(state, batch_size=1)
            
        y_pred[j,:] = state[:]
    
    y_pred = scaler.inverse_transform(y_pred)
    
    return y_pred

################################################################

def frequency(k,data,dt):
    nt = data.shape[0]
    acorr = tsaplots.acf(data[:,k], nlags=nt, fft=False)
    temp = np.where(acorr<(1/np.exp(1)))
    if len(temp[0])==0:    # if temp is empty
        return 0
    else:
        # First occurance of autocorr less than 1/e
        return 365/(temp[0][0]*dt)  # unit: cycles/year

################################################################

def ITCC(data1, data2):
    data1_ = (data1-np.mean(data1,axis=0))/np.std(data1, axis=0)
    data2_ = (data2-np.mean(data2,axis=0))/np.std(data2, axis=0)
    cc = np.sum(data1_*data2_, axis=0)/(data1.shape[0]-1)
    return cc

################################################################

def ISCC(data1, data2):
    data1_ = (data1-np.mean(data1,axis=1).reshape(-1,1))/np.std(data1, axis=1).reshape(-1,1)
    data2_ = (data2-np.mean(data2,axis=1).reshape(-1,1))/np.std(data2, axis=1).reshape(-1,1)
    cc = np.sum(data1_*data2_, axis=1)/(data1.shape[1]-1)
    return cc

################################################################

def ICC_wrapper(k,y_true,y_pred,eofs,n_maxlead,mode):
    psi1_true = y_true[k+1:k+n_maxlead+1,:].dot(eofs)
    psi1_pred = y_pred[k,:,:].dot(eofs)
    if mode == 'spatial':
        return ISCC(psi1_true, psi1_pred)
    elif mode == 'temporal':
        return ITCC(psi1_true, psi1_pred)
    else:
        raise ValueError('mode value not recognised')

################################################################

def RMSE(k,y_true,y_pred,eofs,n_maxlead):
    psi1_true = y_true[k+1:k+n_maxlead+1,:].dot(eofs)
    psi1_pred = y_pred[k,:,:].dot(eofs)
    return np.sqrt(np.mean((psi1_true - psi1_pred)**2, axis=1))

################################################################