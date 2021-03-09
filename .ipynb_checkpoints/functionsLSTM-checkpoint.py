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

def trainingdata(data, n_pc, n_train, mem, norm):
    
    # Define the LSTM inputs/outputs and reshape x_train to 
    # [samples, lag time_steps, features]
    x_train = []
    y_train = []
    
    #### Now partition the training dataset ####
    train_data = data[:n_train, :]
    train_data, scaler = normalize(train_data, norm)

    for i in range(mem,n_train-1):
        x_train.append(train_data[i-mem:i,:])
        y_train.append(train_data[i,:])

    x_train, y_train = np.array(x_train), np.array(y_train)
    
    return x_train, y_train, scaler
    
#######################################################################

def normalize(data, norm):

    if norm=='standard':
        #### Standardize the training dataset ####
        scaler = StandardScaler()
    
    elif norm=='minmax':
        scaler = MinMaxScaler()
        
    else:
        raise ValueError('normalization type not recognised')

    scaler.fit(data)
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
            isdrop=True, dropmag=0.2,
            validation_split=0.2):

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

def forecast(n_pred,n_in_param,x_start,deltaT,model,scaler,dW,isnoise):
    
    x_start = scaler.transform(x_start.reshape(1,-1))
    
    y_pred = np.zeros([n_pred, n_in_param])
        
    state = np.zeros((1, n_in_param))
    state[0,:] = x_start
    
    # This is the forecast loop
    for j in range(n_pred):
        # compute the LSTM output
        temp = model.predict(state.reshape(state.shape[0],
                                           1,state.shape[1]),batch_size=1)
        if isnoise:
            state[0,:] =  np.squeeze(temp) + dW[j,:]
        else:
            state[0,:] =  np.squeeze(temp)
            
        y_pred[j,:] = state
    
    y_pred = scaler.inverse_transform(y_pred)
    
    return y_pred

################################################################