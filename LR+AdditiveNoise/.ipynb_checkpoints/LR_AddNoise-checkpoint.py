##################################################
######### Method: SLR + Additive Noise  ##########
##################################################
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from netCDF4 import Dataset
from functionsLR_AddNoise import *
from scipy import fftpack
from datetime import datetime
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
# np.random.seed(42)

num_cores = multiprocessing.cpu_count()
print('Number of cores:',num_cores)

# Define other parameters of the problem
n_pc = 150         # number of PCs
n_record = 50000   # total records
n_maxtrain = 40000 # maximum possible length of training data
n_test = n_record - n_maxtrain # length of the test data
test0_idx = 40000              # test data start index
dt = 10            # temporal resolution

#### Load the pcs ####
f = 'psi1_DG_0_500K_100days_filtered_150PCs.dat'
fpath = os.getcwd() + '/' + f
pcs = dlmread(fpath,n_record,n_pc)
# get the test dataset
test_data = pcs[test0_idx:test0_idx+n_test, :]
 
#### Load the EOFs ####
nx = 513; ny=513
f = 'psi1_DG_0_500K_100days_filtered_150EOFs.nc'
eoffile = os.getcwd() + '/' + f
fid = Dataset(eoffile,'r')
eofs = fid.variables['EOFs'][:n_pc,:,:]
eofs = eofs.reshape((n_pc, nx*ny))

# train the LR
n_train = n_maxtrain # Length used for training
[x_train, y_train, scaler] = training4tendency(pcs, n_pc, n_train, dt, 'standard_top_pc')
[linregmodel, reg_res, r2_sco] = linear_regression(x_train,y_train)

# plot LR output vs the truth
y_tend_LR = linregmodel.predict(x_train)

#################################################################
######### Calculation of decorrelation time for each PC #########
#################################################################
num_lags = 1200
tau = np.zeros((n_pc,))

start = datetime.now()

for k in range(n_pc):
    autocor_x = tsaplots.acf(x_train[:,k], nlags=num_lags, fft=False)
    tau[k] = np.where(autocor_x<1/np.exp(1))[0][0]

tau = tau*dt
print('Time taken = ',datetime.now()-start)
print(tau)

### Red and White noise definitions ###

def l2_norm(array):
    return np.sqrt(np.sum(np.power(array,2)))

# Red Noise
def corrRedNoise(n_samples, n_in_param, residual, tau, deltaT):
    
    eta = np.zeros((n_samples, n_in_param))
    
    ## compute std of the white noise
    var_res = np.var(residual, axis=0)
    gamma = 1/tau 
    c = (1 - gamma*deltaT)
    sigma = np.sqrt(var_res*(1 - c**2)/(deltaT**2))
    
    # generate white noise and spinup the red noise system to draw 
    # samples from the correct distribution
    n_spin = 500000
    eta_ = np.zeros((n_spin, n_in_param))
    for i in range(1,n_spin):
        zeta_ = sigma*np.random.randn(1, n_in_param)
        eta_[i,:] = c*eta_[i-1,:] + deltaT*zeta_
        
    # compare the std of eta_ with that of the residuals
    std_diff = np.sqrt(var_res) - np.std(eta_, axis=0)
    print('L2 norm of the error in std(red noise):', l2_norm(std_diff))
    
    eta[0,:] = eta_[-1,:]
    for i in range(1, n_samples): 
        zeta = sigma*np.random.randn(1, n_in_param)
        eta[i,:] = c*eta[i-1,:] + deltaT*zeta
    
    covn = np.corrcoef(residual.T) # Correlation coefficients
    rr = np.linalg.cholesky(covn)
    
    return eta.dot(rr.T)

# Correlated white noise
def corrWhtNoise(nx, ny, residual):
    dW = np.random.randn(nx, ny) # additive white noise
    covn = np.corrcoef(residual.T) # Correlation coefficients
    rr = np.linalg.cholesky(covn)
    stdres = np.std(residual, axis=0)
    return dW.dot(rr.T)*stdres

####################################################################
################ Short term forecasts: 50 days #####################
####################################################################
n_maxlead = 5  # maximum lead time for the forecasts
n_ic = int((n_test - n_maxlead))    # Number of initial conditions 
n_ensem = 100   # Number of stochastic ensembles for each I.C.
y_pred = np.zeros([n_ic, n_ensem, n_maxlead, n_pc])
noisetype = 'red'

start = datetime.now()

# generate noise
if noisetype=='white':
    dW = corrWhtNoise(n_ic*n_maxlead*n_ensem, n_pc, reg_res)
elif noisetype=='red':
    dW = corrRedNoise(n_ic*n_maxlead*n_ensem, n_pc, reg_res, tau, dt)
else:
    raise ValueError('noise type not supported')
        
for i in range(n_ic):
    # initial condition
    x_start = test_data[i,:]
    y_pred[i,:,:,:] = forecast(n_ensem, n_maxlead, n_pc, x_start,
                               dt, linregmodel, scaler, 
                               dW[i*n_maxlead*n_ensem:(i+1)*n_maxlead*n_ensem])
    
# Ensemble mean for each IC
enMean_y_pred = np.mean(y_pred, axis=1)

print('Prediction time:', datetime.now()-start)
print('Number of initial conditions:', n_ic)
print('Number of ensembles per initial condition:', n_ensem)
print('Length of each forecast:', n_maxlead)

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

# Plot the mean RMSE
plt.plot(np.arange(dt,n_maxlead*dt+1,dt),meanRMSE)
plt.xlabel('Time (in days)')
plt.ylabel('RMSE')
plt.ylim([10, 80])
plt.xlim([dt, n_maxlead*dt])
plt.grid(color='k', linestyle='--', linewidth=0.2)
plt.savefig('RMSE_Psi1_50days_predictions_AddRedNoise_npc=150_nic=9995_nensem=100_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
np.save('RMSE_Psi1_50days_predictions_AddRedNoise_npc=150_nic=9995_nensem=100_ntrain=400K_ntest=100K',meanRMSE)

#####################################################################
# Metric 2a : Instantaneous temporal correlation coefficient (ITCC) #
#####################################################################
itcc = np.zeros((n_ic, nx*ny))

start = datetime.now()
if __name__ == "__main__":
    itcc = Parallel(n_jobs=num_cores, prefer="threads")(delayed(ICC_wrapper)(i,test_data,enMean_y_pred,eofs,n_maxlead,'temporal') for i in tqdm(range(n_ic)))

print('Time taken:',datetime.now()-start)

meanitcc = np.mean(itcc, axis=0)

plt.imshow(meanitcc.reshape(nx,ny), origin='lower', cmap='jet')
plt.colorbar()
plt.clim([0, 1])
plt.savefig('ITCC_Psi1_50days_predictions_AddRedNoise_npc=150_nic=9995_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
np.save('ITCC_Psi1_50days_predictions_AddRedNoise_npc=150_nic=9995_ntrain=400K_ntest=100K',meanitcc)

####################################################################
# Metric 2b : Instantaneous spatial correlation coefficient (ISCC) #
####################################################################
iscc = np.zeros((n_ic, n_maxlead))

start = datetime.now()
if __name__ == "__main__":
    iscc = Parallel(n_jobs=num_cores, prefer="threads")(delayed(ICC_wrapper)(i,test_data,enMean_y_pred,eofs,n_maxlead,'spatial') for i in tqdm(range(n_ic)))

print('Time taken:',datetime.now()-start)
meaniscc = np.mean(iscc, axis=0)

plt.plot(np.arange(dt,n_maxlead*dt+1,dt), meaniscc)
plt.xlabel('Time (in days)')
plt.ylabel('ISCC')
plt.xlim([dt, n_maxlead*dt])
plt.ylim([0.7, 1])
plt.grid(color='k', linestyle='--', linewidth=0.2)
plt.savefig('ISCC_Psi1_50days_predictions_AddRedNoise_npc=150_nic=9995_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
np.save('ISCC_Psi1_50days_predictions_AddRedNoise_npc=150_nic=9995_ntrain=400K_ntest=100K',meaniscc)

################################################################
############### Long term forecasts: 10,000 days ###############
################################################################
n_maxlead = 1000  # maximum lead time for the forecasts
n_ic = 1    # Number of initial conditions 
n_ensem = 100   # Number of stochastic ensembles for each I.C.
y_pred = np.zeros([n_ic, n_ensem, n_maxlead, n_pc])
noisetype = 'red'
    
start = datetime.now()

# generate noise
if noisetype=='white':
    dW = corrWhtNoise(n_ic*n_maxlead*n_ensem, n_pc, reg_res)
elif noisetype=='red':
    dW = corrRedNoise(n_ic*n_maxlead*n_ensem, n_pc, reg_res, tau, dt)
else:
    raise ValueError('noise type not supported')
    
for k in range(n_ic):
    #### Start Forecasts ####
    x_start = test_data[k,:]
    
    y_pred[i,:,:,:] = forecast(n_ensem, n_maxlead, n_pc, x_start,
                               dt, linregmodel, scaler, 
                               dW[k*n_maxlead*n_ensem:(k+1)*n_maxlead*n_ensem])

# Ensemble mean for each IC
enMean_y_pred = np.mean(y_pred, axis=1)

print('Prediction time:', datetime.now()-start)
print('Number of initial conditions:',n_ic)
print('Number of ensembles per initial condition:',n_ensem)
print('Length of each forecast:',n_maxlead)

################################################################
############# Metric 3 : Climatology and Variance ##############
######### using multiple realizations of 10,000 days  #########
################################################################
psi1_climatology_true = np.zeros((n_ic,nx*ny))
psi1_variance_true = np.zeros((n_ic,nx*ny))
psi1_climatology_pred = np.zeros((n_ic,nx*ny))
psi1_variance_pred = np.zeros((n_ic,nx*ny))

# Project the modelled PCs onto the EOFs and calculate climatology
for kk in range(n_ic):
    psi1_true = test_data[kk+1:kk+n_maxlead+1,:].dot(eofs)
    psi1_climatology_true[kk,:] = np.mean(psi1_true, axis=0)
    psi1_variance_true[kk,:] = np.var(psi1_true, axis=0)
    
    psi1_pred = enMean_y_pred[kk,:,:].dot(eofs)
    psi1_climatology_pred[kk,:] = np.mean(psi1_pred, axis=0)
    psi1_variance_pred[kk,:] = np.var(psi1_pred, axis=0)

mean_psi1_climatology_pred = np.mean(psi1_climatology_pred, axis=0)
mean_psi1_variance_pred = np.mean(psi1_variance_pred, axis=0)
mean_psi1_climatology_true = np.mean(psi1_climatology_true, axis=0)
mean_psi1_variance_true = np.mean(psi1_variance_true, axis=0)

# Climatology and variance of the predictions
plt.imshow(mean_psi1_climatology_pred.reshape(nx,ny), origin='lower', cmap='RdBu')
plt.clim(-50, 50)
plt.colorbar()
plt.savefig('Climatology_Psi1_10Kdays_predictions_AddRedNoise_npc=150_nic=1_nensem=100_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()

plt.imshow(mean_psi1_variance_pred.reshape(nx,ny), origin='lower', cmap='jet')
plt.clim(0, 100000)
plt.colorbar()
plt.savefig('Variance_Psi1_10Kdays_predictions_AddRedNoise_npc=150_nic=1_nensem=100_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()

##################################################
############# Metric 4: Frequency map ############
####### frequency = 1/decorrelation time #########
##################################################
freq_pred = np.zeros((n_ic,nx*ny))

for kk in range(n_ic):
    psi1_pred = enMean_y_pred[kk,:,:].dot(eofs)
    start = datetime.now()
    if __name__ == "__main__":
        freq_ = Parallel(n_jobs=num_cores, prefer="threads")(delayed(frequency)(i,psi1_pred,dt) for i in tqdm(range(nx*ny)))
    freq_pred[kk,:] = freq_
    print('Time taken:',datetime.now()-start)
    
# mean frequency
mean_freq_pred = np.mean(freq_pred, axis=0)

plt.imshow(np.array(mean_freq_pred).reshape(nx,ny), origin='lower', cmap='jet')
plt.colorbar()
plt.clim([0,4])
plt.savefig('Frequency_Psi1_10Kdays_predictions_AddRedNoise_npc=150_nic=1_nensem=100_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
