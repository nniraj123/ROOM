from statsmodels.graphics import tsaplots
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, LSTM, Dropout, Activation
from tensorflow.python.keras.optimizers import RMSprop, SGD, Adagrad, Adadelta
import os
import numpy as np

###########################################################

def center(data):
    return data - np.mean(data, axis=0)

###########################################################

def dlmread(fname, n_run, n_pc):
    
    data_list = []
    fid = open(fname,'r')
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

#############################################################

def trainingdata(data, n_pc, n_train, look_back, norm):
    
    # Define the LSTM inputs/outputs
    # x_train shape: [samples, lag time_steps, features]
    # y_train shape: [samples, labels]
    x_train = []
    y_train = []
    
    #### Now partition the training dataset ####
    train_data = data[:n_train, :]
    train_data, scaler = normalize(train_data, norm)

    for i in range(n_train-look_back):
        temp = train_data[i:(i+look_back), :]
        x_train.append(temp)
        y_train.append(train_data[i + look_back, :])
        
    return np.array(x_train), np.array(y_train), scaler

#    for i in range(1,n_train):
#        x_train.append(train_data[i-1:i,:])
#        y_train.append(train_data[i,:])

#######################################################################

def normalize(data, norm):

    if norm=='standard':
        #### Standardize the training dataset ####
        scaler = StandardScaler()
        scaler.fit(data)
    elif norm=='minmax':
        scaler = MinMaxScaler()
        scaler.fit(data)
    elif norm=='standard_top_pc':
        scaler = StandardScaler()
        scaler.fit(np.tile(data[:,0].reshape(-1,1), 
                           (1, np.shape(data)[1])))
    else:
        raise ValueError('normalization type not recognised')

    normalized = scaler.transform(data)
    
    return normalized, scaler

#############################################################

def addhiddenlayer(model, numhidunits, activation, isdrop, 
                    dropmag):
    model.add(LSTM(numhidunits, activation=activation, 
                   return_sequences=True))
    if isdrop==True:
        model.add(Dropout(dropmag))
        
    return model

#############################################################

def fitLSTM(x, y, n_pc, 
            numhidunits=50, loss='mae', optimizer='adam', 
            epochs=10, numhidlayer=3, activation='tanh',
            isdrop=True, dropmag=0.2, validation_split=0.2):

    model = Sequential()

    # input layer
    model.add(LSTM(numhidunits, activation=activation,
                   return_sequences=True,
                   input_shape=(x.shape[1], x.shape[2])))
    if isdrop==True:
        model.add(Dropout(dropmag)) # dropout layer
    # hidden layers
    for i in range(numhidlayer-1):
        model = addhiddenlayer(model, numhidunits, activation, isdrop, dropmag)
    # last hidden layer
    model.add(LSTM(numhidunits, activation=activation))
    # output layer
    model.add(Dense(y.shape[-1]))
    
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    history = model.fit(x, y,
                    epochs=epochs,
                    verbose=2,
                    batch_size=32,
                    shuffle=False,
                    validation_split=validation_split)

    return model, history

###############################################################

def forecast(n_pred, n_in_param, x_start, deltaT, lstm_model, 
              scaler_x, n_ensem=1, look_back=1, dW=None):
    
    y_pred = np.zeros((n_ensem, n_pred, n_in_param))
    temp = np.zeros((n_pred, n_in_param))
    state = np.zeros((1, n_in_param))
    x_in = np.zeros((look_back, n_in_param))
    
    for i in range(n_ensem):
        # Set the initial state
        x_in = x_start.copy()
    
        for j in range(n_pred):
            # Use LSTM to predict the state
            state = lstm_model.predict(np.expand_dims(x_in, axis=0), 
                                       batch_size=1) + dW[i*n_pred+j,:]
        
            # store the new state in the output
            temp[j,:] = state[:]
            
            # roll over x_in to include the new state
            x_in[:-1,:] = x_in[1:,:]
            x_in[-1,:] = state[:]
        
        y_pred[i,:,:] = scaler_x.inverse_transform(temp)
    
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

def ICC_wrapper(k,y_true,y_pred,eofs,n_maxlead,look_back,mode):
    psi1_true = y_true[k+look_back:k+n_maxlead+look_back,:].dot(eofs)
    psi1_pred = y_pred[k,:,:].dot(eofs)
    if mode == 'spatial':
        return ISCC(psi1_true, psi1_pred)
    elif mode == 'temporal':
        return ITCC(psi1_true, psi1_pred)
    else:
        raise ValueError('mode not recognised')
        
################################################################

def RMSE(k,y_true,y_pred,eofs,n_maxlead,look_back):
    psi1_true = y_true[k+look_back:k+look_back+n_maxlead,:].dot(eofs)
    psi1_pred = y_pred[k,:,:].dot(eofs)
    return np.sqrt(np.mean((psi1_true - psi1_pred)**2, axis=1))

################################################################

def mean_and_variance(data):
    climatology = np.mean(data, axis=0)
    variance = np.var(data, axis=0)
    return climatology, variance

#################################################################
