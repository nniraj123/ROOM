##################################################
#### Method: Multi-level Linear Regression  ######
##################################################
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from netCDF4 import Dataset
from functionsML_LR import *
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
fname = 'psi1_DG_0_500K_100days_filtered_150PCs.dat'
pcs = dlmread(fname,n_record,n_pc,' ')
pcs = center(pcs)
# get the test dataset
test_data = pcs[test0_idx:test0_idx+n_test, :]
 
#### Load the EOFs ####
nx = 513; ny=513
eoffile = os.getcwd() + '/psi1_DG_0_500K_100days_filtered_150EOFs.nc'
fid = Dataset(eoffile,'r')
eofs = fid.variables['EOFs'][:n_pc,:,:]
eofs = eofs.reshape((n_pc, nx*ny))
"""
################################################################
##################### Short term forecasts ######################
################################################################
#### Load the predictions #####
f = 'psi1_pred_150PCs_tlength=400K_plength=100K_lead=100_icond=9991_inorm=1_nensembles=100.dat'
n_maxlead = 10
n_ic = n_test - n_maxlead + 1
y_pred = dlmread(f,n_ic*n_maxlead,n_pc,',') 
y_pred_100D = y_pred.reshape(n_ic,n_maxlead,n_pc,order='F')

####################################################################
############ Metric 1 : RMSE on the physical space #################
################### For short-term forecasts #######################
####################################################################
# Project the modelled PCs onto the EOFs and calculate mean RMSE
rmse = np.zeros((n_ic, n_maxlead))

start = datetime.now()
# start a parallel pool and implement thread parallelism
if __name__ == "__main__":
    rmse = Parallel(n_jobs=num_cores, prefer="threads")(delayed(RMSE)(i,test_data,y_pred_100D,eofs,n_maxlead) for i in tqdm(range(n_ic)))

meanRMSE = np.mean(rmse, axis=0)
np.save('RMSE_Psi1_100days_predictions_MLLR_nlevel=2_npc=150_norm=own_nic=9991_nensem=100_ntrain=400K_ntest=100K',meanRMSE)

####################################################################
# Metric 2a : Instantaneous temporal correlation coefficient (ITCC) #
####################################################################
itcc = np.zeros((n_ic, nx*ny))

start = datetime.now()
if __name__ == "__main__":
    itcc = Parallel(n_jobs=num_cores, prefer="threads")(delayed(ICC_wrapper)(i,test_data,y_pred_100D,eofs,n_maxlead,'temporal') for i in tqdm(range(n_ic)))

print('Time taken:',datetime.now()-start)
meanitcc = np.mean(itcc, axis=0)

plt.imshow(meanitcc.reshape(nx,ny), origin='lower',cmap='jet')
plt.colorbar()
plt.clim([0, 1])
plt.savefig('ITCC_Psi1_100days_predictions_MLLR_nlevel=2_npc=150_norm=own_nic=9991_nensem=100_ntrain=400K_ntest=100K.png',dpi=100)
plt.close()
np.save('ITCC_Psi1_100days_predictions_MLLR_nlevel=2_npc=150_norm=own_nic=9991_nensem=100_ntrain=400K_ntest=100K',meanitcc)

####################################################################
# Metric 2b : Instantaneous spatial correlation coefficient (ISCC) #
####################################################################
iscc = np.zeros((n_ic, n_maxlead))

start = datetime.now()
if __name__ == "__main__":
    iscc = Parallel(n_jobs=num_cores, prefer="threads")(delayed(ICC_wrapper)(i,test_data,y_pred_100D,eofs,n_maxlead,'spatial') for i in tqdm(range(n_ic)))

print('Time taken:',datetime.now()-start)
meaniscc = np.mean(iscc, axis=0)
np.save('ISCC_Psi1_100days_predictions_MLLR_nlevel=2_npc=150_norm=own_nic=9991_nensem=100_ntrain=400K_ntest=100K',meaniscc)

"""
################################################################
##################### Long term forecasts ######################
################################################################
#### Load the predictions #####
f = 'psi1_pred_150PCs_tlength=400K_lead=200K_icond=1_inorm=1_nensembles=1.dat'
n_maxlead = 20000
n_ic = 1
y_pred = dlmread(f,n_ic*n_maxlead,n_pc,',') 
y_pred_200K = y_pred.reshape(n_ic,n_maxlead,n_pc,order='F')

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
    psi1_pred = y_pred_200K[kk,:,:].dot(eofs)
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
"""
# Climatology and variance of the predictions
plt.imshow(mean_psi1_climatology_pred.reshape(nx,ny), origin='lower', cmap='RdBu')
plt.clim(-20, 20)
plt.colorbar()
plt.savefig('Climatology_Psi1_200Kdays_predictions_MLLR_npc=150_norm=own_nic=1_nensem=1_ntrain=400K.png',dpi=100)
plt.close()
np.save('Climatology_Psi1_200Kdays_predictions_MLLR_npc=150_norm=own_nic=1_nensem=1_ntrain=400K',mean_psi1_climatology_pred)

plt.imshow(mean_psi1_variance_pred.reshape(nx,ny), origin='lower', cmap='jet')
plt.clim(0, 100000)
plt.colorbar()
plt.savefig('Variance_Psi1_200Kdays_predictions_MLLR_npc=150_norm=own_nic=1_nensem=1_ntrain=400K.png',dpi=100)
plt.close()
np.save('Variance_Psi1_200Kdays_predictions_MLLR_npc=150_norm=own_nic=1_nensem=1_ntrain=400K',mean_psi1_variance_pred)
"""
plt.imshow(np.array(mean_freq_pred).reshape(nx,ny), origin='lower', cmap='jet')
plt.colorbar()
plt.clim([0,4])
plt.savefig('Frequency_Psi1_200Kdays_predictions_MLLR_npc=150_norm=own_nic=1_nensem=1_ntrain=400K.png',dpi=100)
plt.close()
np.save('Frequency_Psi1_200Kdays_predictions_MLLR_npc=150_norm=own_nic=1_nensem=1_ntrain=400K',mean_freq_pred)
