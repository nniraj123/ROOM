############################################################
########## Artificial Neural Network (ANN) with noise ######
############################################################
import numpy as np
import matplotlib.pyplot as plt
import math
from netCDF4 import Dataset
import json
from datetime import datetime
from functionsANN import *
from datetime import datetime
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
# np.random.seed(42)

print('Number of cores:',multiprocessing.cpu_count())
num_cores = 24

# Define other parameters of the problem
n_pc = 150         # number of PCs
n_record = 50000   # total records
n_maxtrain = 40000 # maximum possible length of training data
n_test = n_record - n_maxtrain # length of the test data
test0_idx = n_maxtrain              # test data start index
dt = 10            # temporal resolution

#### Load the pcs ####
f = 'psi1_DG_0_500K_100days_filtered_150PCs.dat'
fpath = os.getcwd() + '/' + f
pcs = dlmread(fpath, n_record, n_pc)
pcs = center(pcs)
# get the test dataset
test_data = pcs[test0_idx:test0_idx+n_test, :]
 
#### Load the EOFs ####
nx = 513; ny=513
f = 'psi1_DG_0_500K_100days_filtered_150EOFs.nc'
eoffile = os.getcwd() + '/' + f
fid = Dataset(eoffile,'r')
eofs = fid.variables['EOFs'][:n_pc,:,:]
eofs = eofs.reshape((n_pc, nx*ny))

# get the training dataset
n_train = n_maxtrain # Length used for training
[x_train, y_train, scaler] = trainingdata(pcs, n_pc, n_train, 'standard')

# Train the Neural Network
start = datetime.now()

hyperparams = {'activation':'tanh', 'npl':100, 
               'loss':'mae', 'optimizer':'adam',
               'metric':'mse', 'epochs':50,
               'numhidlayers':2,'validation_split':0.2}

model, history = ANN(x_train, y_train, n_pc, **hyperparams)

print('Training time:',datetime.now()-start)

ypred_ANN = np.zeros((n_train-1, n_pc))

for k in range(n_train-1):
    ypred_ANN[k,:] = model.predict(np.reshape(x_train[k,:], [1,-1]), batch_size=1)

# compute ANN residuals
residual = y_train - ypred_ANN

# Spatially correlated white noise
def spatialCorrWhtNoise(nt, npc, residual):
    dW = np.random.randn(nt, npc) # additive white noise
    covn = np.cov(residual.T) # Correlation coefficients
    rr = np.linalg.cholesky(covn)
    stdres = np.std(residual, axis=0)
    return dW.dot(rr.T)*stdres
"""
################################################################
##################### Short term forecasts #####################
################################################################
n_maxlead = 10  # Prediction of 20 records for each realization
n_ic = int((n_test - n_maxlead))    # Number of initial conditions 
n_ensem = 100
y_pred = np.zeros([n_ic, n_ensem, n_maxlead, n_pc])
isnoise = True
test_dataN = scaler.transform(test_data)

start = datetime.now()
for k in range(n_ic):
    # initial conditions
    x_start = test_dataN[k,:].reshape(1,-1)
    
    if isnoise:
        # generate the noise for all ensembles
        dW = spatialCorrWhtNoise(n_ensem*n_maxlead, n_pc, residual)
    else:
        # set zero values to the noise component
        dW = np.zeros((n_ensem*n_maxlead, n_pc))
    
    # produce forecasts
    y_pred[k,:,:,:] = forecast(n_maxlead,n_pc,x_start,
                               model,scaler,n_ensem,dW)

enMean_y_pred = np.mean(y_pred, axis=1)
print('Prediction time:', datetime.now()-start)
print('Number of realizations obtained:',n_ic)
print('Prediction length of each realization:',n_maxlead)

####################################################################
############ Metric 1 : RMSE on the physical space #################
###### Use each data point of the test dataset as an IC ############
## Obtain 100 stochastic realizations (if applicable) for each IC ##
####################################################################
# Project the modelled PCs onto the EOFs and calculate mean RMSE
rmse = np.zeros((n_ic, n_maxlead))
start = datetime.now()
# start a parallel pool and implement thread parallelism
if __name__ == "__main__":
    rmse = Parallel(n_jobs=num_cores, prefer="threads")(delayed(RMSE)(i,test_data,enMean_y_pred,eofs,n_maxlead) for i in tqdm(range(n_ic)))
    
meanRMSE = np.mean(rmse, axis=0)
np.save('RMSE_Psi1_100days_predictions_ANN+Noise_npc=150_norm=own_nic=9990_nensem=100_ntrain=400K_ntest=100K',meanRMSE)

# Plot the mean RMSE
#plt.plot(np.arange(dt,n_maxlead*dt+1,dt),meanRMSE)
#plt.xlabel('Time (in days)')
#plt.ylabel('RMSE')
#plt.ylim([10, 80])
#plt.xlim([dt, n_maxlead*dt])
#plt.grid(color='k', linestyle='--', linewidth=0.2)
#plt.savefig('RMSE_Psi1_100days_predictions_ANN+Noise_npc=150_norm=own_nic=9990_nensem=100_ntrain=400K_ntest=100K.png',dpi=100)
#plt.close()

####################################################################
# Metric 2a : Instantaneous temporal correlation coefficient (ITCC) #
####################################################################
itcc = np.zeros((n_ic, nx*ny))

start = datetime.now()
if __name__ == "__main__":
    itcc = Parallel(n_jobs=num_cores, prefer="threads")(delayed(ICC_wrapper)(i,test_data,enMean_y_pred,eofs,n_maxlead,'temporal') for i in tqdm(range(n_ic)))

print('Time taken:',datetime.now()-start)
meanitcc = np.mean(itcc, axis=0)

plt.imshow(meanitcc.reshape(nx,ny), origin='lower',cmap='jet')
plt.colorbar()
plt.clim([0, 1])
plt.savefig('ITCC_Psi1_100days_predictions_ANN+Noise_npc=150_norm=own_nic=9990_nensem=100_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
np.save('ITCC_Psi1_100days_predictions_ANN+Noise_npc=150_norm=own_nic=9990_nensem=100_ntrain=400K_ntest=100K',meanitcc)

####################################################################
# Metric 2b : Instantaneous spatial correlation coefficient (ISCC) #
####################################################################
iscc = np.zeros((n_ic, nx*ny))

start = datetime.now()
if __name__ == "__main__":
    iscc = Parallel(n_jobs=num_cores, prefer="threads")(delayed(ICC_wrapper)(i,test_data,enMean_y_pred,eofs,n_maxlead,'spatial') for i in tqdm(range(n_ic)))

print('Time taken:',datetime.now()-start)
meaniscc = np.mean(iscc, axis=0)

#plt.plot(np.arange(dt,n_maxlead*dt+1,dt), meaniscc)
#plt.xlabel('Time (in days)')
#plt.ylabel('ISCC')
#plt.xlim([dt, n_maxlead*dt])
#plt.ylim([0.7, 1])
#plt.grid(color='k', linestyle='--', linewidth=0.2)
#plt.savefig('ISCC_Psi1_100days_predictions_ANN+Noise_npc=150_norm=own_nic=9990_ntrain=400K_ntest=100K.png',dpi=100)
#plt.close()
np.save('ISCC_Psi1_100days_predictions_ANN+Noise_npc=150_norm=own_nic=9990_nensem=100_ntrain=400K_ntest=100K',meaniscc)
"""
################################################################
#################### Long term forecasts #######################
################################################################
n_maxlead = 20000  # Prediction of 1000 records for each realization
n_ic = 1    # Number of initial conditions 
isnoise = True
n_ensem = 1
y_pred = np.zeros([n_ic, n_ensem, n_maxlead, n_pc])
test_dataN = scaler.transform(test_data)

start = datetime.now()
for k in range(n_ic):
    # initial conditions
    x_start = test_dataN[k,:].reshape(1,-1)
    
    if isnoise:
        # generate the noise for all ensembles
        dW = spatialCorrWhtNoise(n_ensem*n_maxlead, n_pc, residual)
    else:
        # set zero values to the noise component
        dW = np.zeros((n_ensem*n_maxlead, n_pc))

    # produce forecasts
    y_pred[k,:,:,:] = forecast(n_maxlead,n_pc,x_start,
                               model,scaler,n_ensem,dW)

enMean_y_pred = np.mean(y_pred, axis=1)
print('Prediction time:', datetime.now()-start)
print('Number of realizations obtained:',n_ic)
print('Prediction length of each realization:',n_maxlead)

################################################################
##### Metric 3-5 : Climatology, Variance, and Frequency map ####
############### using long time scale forecasts  ###############
############ frequency = 1/decorrelation time ##################
################################################################
psi1_climatology_pred = np.zeros((n_ic,nx*ny))
psi1_variance_pred = np.zeros((n_ic,nx*ny))
freq_pred = np.zeros((n_ic,nx*ny))
mean_tmp = np.zeros((nx*ny))
var_tmp = np.zeros((nx*ny))

# Project the modelled PCs onto the EOFs and calculate climatology
for kk in range(n_ic):
    psi1_pred = enMean_y_pred[kk,:,:].dot(eofs)
    mean_tmp, var_tmp = mean_and_variance(psi1_pred)
    psi1_climatology_pred[kk,:] = mean_tmp
    psi1_variance_pred[kk,:] = var_tmp
    if __name__ == "__main__":
        freq_ = Parallel(n_jobs=num_cores, prefer="threads")(delayed(frequency)(i,psi1_pred,dt) for i in tqdm(range(nx*ny)))
    freq_pred[kk,:] = freq_

mean_psi1_climatology_pred = np.mean(psi1_climatology_pred, axis=0)
mean_psi1_variance_pred = np.mean(psi1_variance_pred, axis=0)
mean_freq_pred = np.mean(freq_pred, axis=0)
del psi1_pred

# Climatology and variance of the predictions
plt.imshow(mean_psi1_climatology_pred.reshape(nx,ny), origin='lower', cmap='RdBu')
plt.clim(-20, 20)
plt.colorbar()
plt.savefig('Climatology_Psi1_200Kdays_predictions_ANN+Noise_npc=150_norm=own_nic=1_nensem=1_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
np.save('Climatology_Psi1_200Kdays_predictions_ANN+Noise_npc=150_norm=own_nic=1_nensem=1_ntrain=400K_ntest=100K',mean_psi1_climatology_pred)

plt.imshow(mean_psi1_variance_pred.reshape(nx,ny), origin='lower', cmap='jet')
plt.clim(0, 100000)
plt.colorbar()
plt.savefig('Variance_Psi1_200Kdays_predictions_ANN+Noise_npc=150_norm=own_nic=1_nensem=1_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
np.save('Variance_Psi1_200Kdays_predictions_ANN+Noise_npc=150_norm=own_nic=1_nensem=1_ntrain=400K_ntest=100K',mean_psi1_variance_pred)

# Frequency map of the predictions
plt.imshow(np.array(mean_freq_pred).reshape(nx,ny), origin='lower', cmap='jet')
plt.colorbar()
plt.clim([0,4])
plt.savefig('Frequency_Psi1_200Kdays_predictions_ANN+Noise_npc=150_norm=own_nic=1_nensem=1_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
np.save('Frequency_Psi1_200Kdays_predictions_ANN+Noise_npc=150_norm=own_nic=1_nensem=1_ntrain=400K_ntest=100K',mean_freq_pred)
