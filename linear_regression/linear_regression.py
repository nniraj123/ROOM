##################################################
################ Method: SLR  ####################
##################################################
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from netCDF4 import Dataset
from functionsLR import *
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
test0_idx = 40000              # test data start index
dt = 10            # temporal resolution

#### Load the pcs ####
f = 'psi1_DG_0_500K_100days_filtered_150PCs.dat'
pcs = dlmread(f, n_record, n_pc)
# remove the mean
pcs = center(pcs)

# get the test dataset
test_data = pcs[test0_idx:test0_idx+n_test, :]
 
#### Load the EOFs ####
nx = 513; ny=513
eoffile = os.getcwd() + '/psi1_DG_0_500K_100days_filtered_150EOFs.nc'
fid = Dataset(eoffile,'r')
eofs = fid.variables['EOFs'][:n_pc,:,:]
eofs = eofs.reshape((n_pc, nx*ny))

# Choose a training length and perform SLR 
n_train = n_maxtrain
x_train, y_train, scaler = training4tendency(pcs, n_pc, n_train, dt, 'standard')

# Train the Linear Regression model
start = datetime.now()
linregmodel, reg_res, r2_sco = linear_regression(x_train,y_train)
print('Training time:',datetime.now()-start)
"""
####################################################################
#################### Short term forecasts ##########################
####################################################################
n_maxlead = 10  # maximum lead time steps for forecasts
n_ic = int((n_test - n_maxlead))    # Number of initial conditions 
y_pred = np.zeros([n_ic, n_maxlead, n_pc])

start = datetime.now()
for k in range(n_ic):
    #### Start Forecasts ####
    x_start = test_data[k,:]
    
    y_pred[k,:,:] = forecast(n_maxlead,n_pc,x_start,dt,linregmodel,scaler)

print('Prediction time:', datetime.now()-start)
print('Number of initial conditions:',n_ic)
print('Length of each forecast:',n_maxlead)

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
    rmse = Parallel(n_jobs=num_cores, prefer="threads")(delayed(RMSE)(i,test_data,y_pred,eofs,n_maxlead) for i in tqdm(range(n_ic)))

meanRMSE = np.mean(rmse, axis=0)
np.save('RMSE_Psi1_100days_predictions_SLR_npc=150_norm=own_nic=9990_ntrain=400K_ntest=100K',meanRMSE)

# Plot the mean RMSE
#plt.plot(np.arange(dt,n_maxlead*dt+1,dt),meanRMSE)
#plt.xlabel('Time (in days)')
#plt.ylabel('RMSE')
#plt.ylim([10, 80])
#plt.xlim([dt, n_maxlead*dt])
#plt.grid(color='k', linestyle='--', linewidth=0.2)
#plt.savefig('RMSE_Psi1_50days_predictions_SLR_npc=150_nic=9995_ntrain=400K_ntest=100K.png',dpi=100)
#plt.show()

# Calculate the 'Persistence' RMSE
y_persist = np.zeros([n_ic, n_maxlead, n_pc])

for k in range(n_ic):
    #### Start Forecasts ####
    x_start = test_data[k,:]
    y_persist[k,:,:] = np.repeat(np.reshape(x_start,[1,-1]),n_maxlead,axis=0)
    
rmse_persist = np.zeros((n_ic, n_maxlead))

# start a parallel pool and implement thread parallelism
if __name__ == "__main__":
    rmse_persist = Parallel(n_jobs=num_cores, prefer="threads")(delayed(RMSE)(i,test_data,y_persist,eofs,n_maxlead) for i in tqdm(range(n_ic)))

meanRMSE_persist = np.mean(rmse_persist, axis=0)
np.save('RMSE_Psi1_100days_predictions_Persistence_npc=150_nic=9990_ntrain=400K_ntest=100K',meanRMSE_persist)

#plt.plot(np.arange(dt,n_maxlead*dt+1,dt),meanRMSE_persist)
#plt.xlabel('Time (in days)')
#plt.ylabel('RMSE')
#plt.ylim([10, 80])
#plt.xlim([dt, n_maxlead*dt])
#plt.grid(color='k', linestyle='--', linewidth=0.2)
#plt.savefig('RMSE_Psi1_100days_predictions_Persistence_npc=150_nic=9990_ntrain=400K_ntest=100K.png',dpi=100)
#plt.close()
#np.save('RMSE_Psi1_100days_predictions_Persistence_npc=150_nic=9990_ntrain=400K_ntest=100K',meanRMSE_persist)

###################################################################
# Metric 2a : Instantaneous temporal correlation coefficient (ITCC) #
###################################################################
itcc = np.zeros((n_ic, nx*ny))

start = datetime.now()
if __name__ == "__main__":
    itcc = Parallel(n_jobs=num_cores, prefer="threads")(delayed(ICC_wrapper)(i,test_data,y_pred,eofs,n_maxlead,'temporal') for i in tqdm(range(n_ic)))

print('Time taken:',datetime.now()-start)

meanitcc = np.mean(itcc, axis=0)

plt.imshow(meanitcc.reshape(nx,ny), origin='lower',cmap='jet')
plt.colorbar()
plt.clim([0, 1])
plt.savefig('ITCC_Psi1_100days_predictions_SLR_npc=150_norm=own_nic=9990_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
np.save('ITCC_Psi1_100days_predictions_SLR_npc=150_norm=own_nic=9990_ntrain=400K_ntest=100K',meanitcc)

###################################################################
# Metric 2b : Instantaneous spatial correlation coefficient (ISCC) #
###################################################################
iscc = np.zeros((n_ic, nx*ny))

start = datetime.now()
if __name__ == "__main__":
    iscc = Parallel(n_jobs=num_cores, prefer="threads")(delayed(ICC_wrapper)(i,test_data,y_pred,eofs,n_maxlead,'spatial') for i in tqdm(range(n_ic)))

print('Time taken:',datetime.now()-start)
meaniscc = np.mean(iscc, axis=0)

plt.plot(np.arange(dt,n_maxlead*dt+1,dt), meaniscc)
plt.xlabel('Time (in days)')
plt.ylabel('ISCC')
plt.xlim([dt, n_maxlead*dt])
plt.grid(color='k', linestyle='--', linewidth=0.2)
plt.savefig('ISCC_Psi1_100days_predictions_SLR_npc=150_norm=own_nic=9990_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
np.save('ISCC_Psi1_100days_predictions_SLR_npc=150_norm=own_nic=9990_ntrain=400K_ntest=100K',meaniscc)
"""
################################################################
##################### Long term forecasts ######################
################################################################
n_ic = 1    # Number of initial conditions   
n_maxlead = 20000  # Prediction of 50K records for each realization
y_pred = np.zeros([n_ic, n_maxlead, n_pc])

start = datetime.now()
for k in range(n_ic):
    #### Start Forecasts ####
    x_start = test_data[k,:]
    
    y_pred[k,:,:] = forecast(n_maxlead,n_pc,x_start,dt,linregmodel,scaler)

print('Prediction time:', datetime.now()-start)
"""
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
    psi1_pred = y_pred[kk,:,:].dot(eofs)
    mean_tmp, var_tmp = mean_and_variance(psi1_pred)
    psi1_climatology_pred[kk,:] = mean_tmp
    psi1_variance_pred[kk,:] = var_tmp
    if __name__ == "__main__":
        freq_ = Parallel(n_jobs=num_cores, prefer="threads")(delayed(frequency)(i,psi1_pred,dt) for i in tqdm(range(nx*ny)))
    freq_pred[kk,:] = freq_

mean_psi1_climatology_pred = np.mean(psi1_climatology_pred, axis=0)
mean_psi1_variance_pred = np.mean(psi1_variance_pred, axis=0)
mean_freq_pred = np.mean(freq_pred, axis=0)

# Climatology and variance of the predictions
plt.imshow(mean_psi1_climatology_pred.reshape(nx,ny), origin='lower', cmap='RdBu')
plt.clim(-20, 20)
plt.colorbar()
plt.savefig('Climatology_Psi1_200Kdays_predictions_SLR_npc=150_norm=own_nic=1_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
np.save('Climatology_Psi1_200Kdays_predictions_SLR_npc=150_norm=own_nic=1_ntrain=400K_ntest=100K',mean_psi1_climatology_pred)

plt.imshow(mean_psi1_variance_pred.reshape(nx,ny), origin='lower', cmap='jet')
plt.clim(0, 100000)
plt.colorbar()
plt.savefig('Variance_Psi1_200Kdays_predictions_SLR_npc=150_norm=own_nic=1_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
np.save('Variance_Psi1_200Kdays_predictions_SLR_npc=150_norm=own_nic=1_ntrain=400K_ntest=100K',mean_psi1_variance_pred)

# Frequency map of the predictions
plt.imshow(np.array(mean_freq_pred).reshape(nx,ny), origin='lower', cmap='jet')
plt.colorbar()
plt.clim([0,4])
plt.savefig('Frequency_Psi1_200Kdays_predictions_SLR_npc=150_norm=own_nic=1_ntrain=400K_ntest=100K.png',dpi=300)
plt.close()
np.save('Frequency_Psi1_200Kdays_predictions_SLR_npc=150_norm=own_nic=1_ntrain=400K_ntest=100K',mean_freq_pred)

del psi1_pred
"""
############################################################################
##### Climatology, Variance, and Frequency map for the reference truth #####
############################################################################
psi1_climatology_true = np.zeros((n_ic,nx*ny))
psi1_variance_true = np.zeros((n_ic,nx*ny))
freq_true = np.zeros((n_ic,nx*ny))

for kk in range(n_ic):
    psi1_true = pcs[kk:kk+n_maxlead,:].dot(eofs)
    mean_tmp, var_tmp = mean_and_variance(psi1_true)
    psi1_climatology_true[kk,:] = mean_tmp
    psi1_variance_true[kk,:] = var_tmp
    # here the parallelism is done over the number of spatial points
    #if __name__ == "__main__":
    #    freq_ = Parallel(n_jobs=num_cores, prefer="threads")(delayed(frequency)(i,psi1_true,dt) for i in tqdm(range(nx*ny)))
    #freq_true[kk,:] = freq_ 

mean_psi1_climatology_true = np.mean(psi1_climatology_true, axis=0)
mean_psi1_variance_true = np.mean(psi1_variance_true, axis=0)
#mean_freq_true = np.mean(freq_true, axis=0)
"""
# Climatology and variance of the truth
plt.imshow(mean_psi1_climatology_true.reshape(nx,ny), origin='lower', cmap='RdBu')
plt.clim(-20,20)
plt.colorbar()
plt.savefig('Climatology_Psi1_200Kdays_true_npc=150_nic=1.png',dpi=100)
plt.close()
np.save('Climatology_Psi1_200Kdays_true_npc=150_nic=1',mean_psi1_climatology_true)
"""
plt.imshow(mean_psi1_variance_true.reshape(nx,ny), origin='lower', cmap='jet')
plt.colorbar()
plt.clim(0,100000)
plt.savefig('Variance_Psi1_200Kdays_true_npc=150_nic=1.png',dpi=100)
plt.close()
np.save('Variance_Psi1_200Kdays_true_npc=150_nic=1',mean_psi1_variance_true)
"""
# Frequency map
plt.imshow(np.array(mean_freq_true).reshape(nx,ny), origin='lower', cmap='jet')
plt.colorbar()
plt.clim([0,4])
plt.savefig('Frequency_Psi1_200Kdays_true_npc=150_nic=1.png',dpi=300)
plt.close()
np.save('Frequency_Psi1_200Kdays_true_npc=150_nic=1',mean_freq_true)
del psi1_true

############################################################################
################### Metric 5 : Prediction Horizon ##########################
############################################################################
ev = np.linalg.eig(linregmodel.coef_)
eigval = ev[0]
timescale = 1./np.real(eigval)
print('Prediction Horizon (in days):',timescale*dt)
print('Overall prediction horizon:',np.min(np.abs(timescale))*dt)
"""
