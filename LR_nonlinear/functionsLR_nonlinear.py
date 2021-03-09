from sklearn.linear_model import LinearRegression
from statsmodels.graphics import tsaplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
import numpy as np
import os
from itertools import combinations 
from sklearn.preprocessing import PolynomialFeatures

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

def training4tendency(data, n_pc, n_train, dt, normalize):
    
    # modelling dy(n)/dt = f(y(n))
    x_train = np.zeros((n_train, n_pc))
    y_train = np.zeros((n_train, n_pc))

    #### Now select the training dataset ####
    train_data = data[:n_train, :n_pc]

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

###########################################################

def linear_regression(x,y,deg,bias=False):
    ###############################################
    ######### Simple linear regression ############
    ###############################################
    
    # determine predictors using the feature variables
    poly = PolynomialFeatures(degree=deg,include_bias=bias)
    predictor = poly.fit_transform(x)

    # fit the model
    model = LinearRegression()
    model.fit(predictor, y)
    
    n_t = np.shape(x)[0]
    n_pc = np.shape(x)[1]
    
    # Variance error estimates
    r2_score = np.zeros(n_pc)
    reg_res = np.zeros([n_t, n_pc])

    # Calculate the regression residuals
    reg_res = y - model.predict(predictor)
    
    # Residual sum of squares
    rss = np.sum(reg_res**2, axis=0)
    
    # Total sum of squares
    tss = np.sum((y - np.mean(y, axis=0))**2, axis=0)
    
    # R2 score = 1 - rss/tss 
    r2_score = 1 - rss/tss
    
    return model, reg_res, r2_score

###############################################################

def forecast(n_pred,n_in_param,x_start,deltaT,linregmodel,deg,scaler,bias=False):
    
    y_pred = np.zeros([n_pred, n_in_param])
    
    # Normalize x_start
    x_start = scaler.transform(x_start.reshape(1,-1))
    
    # declare the auxiliary variables
    out0 = np.zeros((1,n_in_param))
    out1 = np.zeros((1,n_in_param))
    out2 = np.zeros((1,n_in_param))
    out3 = np.zeros((1,n_in_param))
    
    state = x_start   # predictand
    poly = PolynomialFeatures(degree=deg,include_bias=bias)
    predictor = poly.fit_transform(state)  # predictor
        
    # Set all out# variables to zero
    out0 = 0; out1=0; out2=0; out3=0
    
    # This is the forecast loop
    for j in range(n_pred):
        out3 = out2
        out2 = out1
        # Predict the linear tendency
        out1 = linregmodel.predict(predictor)
        # The following is a simple Adams-Bashforth timestepping scheme 
        # with special treatment of step one and two:
        if j == 0:
            out0 = out1
        if j == 1:
            out0 = 1.5*out1-0.5*out2
        if j > 1:
            out0 = (23.0/12.0)*out1-(4.0/3.0)*out2+(5.0/12.0)*out3
        
        # Update the state vector
        state = state + out0*deltaT
        y_pred[j,:] = state
        predictor = poly.fit_transform(state)
    
    y_pred = scaler.inverse_transform(y_pred)

    return y_pred

################################################################

def RMSE(k,y_true,y_pred,eofs,n_maxlead):
    psi1_true = y_true[k+1:k+n_maxlead+1,:].dot(eofs)
    psi1_pred = y_pred[k,:,:].dot(eofs)
    return np.sqrt(np.mean((psi1_true - psi1_pred)**2, axis=1))

################################################################