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
from sklearn.linear_model import LinearRegression

###########################################################

def center(data):
    return data - np.mean(data, axis=0)

####################################################################

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

######################################################################

def training4tendency(data, n_pc, n_train, dt, normalize):
    
    # modelling dy(n)/dt = f(y(n))
    x_train = np.zeros((n_train, n_pc))
    y_train = np.zeros((n_train, n_pc))

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
    
    x_train = train_data
    y_train = np.diff(x_train, axis=0)/dt
    
    return x_train[:-1,:], y_train, scaler

######################################################################

def linear_regression(x,y):
    ###############################################
    ######### Simple linear regression ############
    ###############################################
    n_train = np.shape(x)[0]
    n_in_param = np.shape(x)[1]
    
    model = LinearRegression()
    
    # Variance error estimates
    r2_score = np.zeros(n_in_param)
    linregmodel = np.zeros([n_in_param, n_in_param])
    reg_res = np.zeros([n_train, n_in_param])

    # Fit the model
    model.fit(x,y)
    linregmodel = model.coef_

    # Residual sum of squares
    rss = np.sum((y - model.predict(x))**2, axis=0)
    
    # Total sum of squares
    tss = np.sum((y - np.mean(y, axis=0))**2, axis=0)
    
    # R2 score = 1 - rss/tss 
    r2_score = 1 - rss/tss

    # Calculate the regression residuals
    reg_res = y - model.predict(x)
    
    return model, reg_res, r2_score 

#######################################################################

def addhiddenlayer(model, npl, activation):
    model.add(Dense(npl, activation=activation))
    
def ANN(x, y, n_in_param, 
        activation='tanh', npl=200, 
        loss='mae', optimizer='adam',
        metric='mae',epochs=100,
        numhidlayers=3,validation_split=0.2):
    
    # This now the keras part. The network is "dense" with each neurons 
    # connected to each neuron of the previous and following layer

    model = Sequential()
    model.add(Dense(x.shape[1], input_dim=x.shape[1], activation=activation))
    for k in range(numhidlayers):
        addhiddenlayer(model, npl, activation)
    model.add(Dense(y.shape[1], activation='linear'))

    # Configuring the learning process
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    # This is the training part of the network:
    history = model.fit(x, y, epochs=epochs, validation_split=validation_split)
    
    # And here the network is stored:
    model.save_weights("./weights")

    return model, history

######################################################################

def forecast_hybrid(n_pred, n_in_param, LR_icond, ann_icond, deltaT, ann_model, 
                    linregmodel, scaler_x, scaler_linregres, n_ensem=1, dW=None):
    
    y_pred = np.zeros((n_ensem, n_pred, n_in_param))
    temp = np.zeros((n_pred, n_in_param))
    state = np.zeros((1, n_in_param))
    out0, out1, out2, out3 = state, state, state, state
    
    for i in range(n_ensem):
        # Set the initial state
        state = LR_icond
        stateAndRes = ann_icond
        
        # Set all out# variables to zero
        out0=0; out1=0; out2=0; out3=0
    
        for j in range(n_pred):
            out3 = out2
            out2 = out1
            # Use LSTM to predict the residual
            res_next = ann_model.predict(stateAndRes, batch_size=1) + dW[i*n_pred+j,:]
            res_next_un = scaler_linregres.inverse_transform(res_next)
    
            out1 = linregmodel.predict(state) + res_next_un
            # The following is a simple Adams-Bashforth timestepping scheme 
            # with special treatment of step one and two:
            if j == 0:
                out0 = out1
            if j == 1:
                out0 = 1.5*out1-0.5*out2
            if j > 1:
                out0 = (23.0/12.0)*out1-(4.0/3.0)*out2+(5.0/12.0)*out3
        
            # Update the state
            state = state + out0*deltaT
            # store the new state in the output
            temp[j,:] = state[:]
            # update the LSTM input
            stateAndRes = np.hstack([state, res_next])
        
        y_pred[i,:,:] = scaler_x.inverse_transform(temp)
    
    return y_pred

######################################################################

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
    psi1_true = y_true[k+2:k+n_maxlead+2,:].dot(eofs)
    psi1_pred = y_pred[k,:,:].dot(eofs)
    if mode == 'spatial':
        return ISCC(psi1_true, psi1_pred)
    elif mode == 'temporal':
        return ITCC(psi1_true, psi1_pred)
    else:
        raise ValueError('mode value not recognised')
        
################################################################

def RMSE(k,y_true,y_pred,eofs,n_maxlead):
    psi1_true = y_true[k+2:k+n_maxlead+2,:].dot(eofs)
    psi1_pred = y_pred[k,:,:].dot(eofs)
    return np.sqrt(np.mean((psi1_true - psi1_pred)**2, axis=1))

################################################################

def mean_and_variance(data):
    climatology = np.mean(data, axis=0)
    variance = np.var(data, axis=0)
    return climatology, variance

#################################################################

def mean_and_variance(data):
    climatology = np.mean(data, axis=0)
    variance = np.var(data, axis=0)
    return climatology, variance

#################################################################