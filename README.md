# Reduced-Order Ocean Modeling (ROOM)
We have employed several data-driven methods to obtain a low-cost data-driven upper ocean emulator for use in coupled ocean-atmosphere models.
The reference dataset is provided by the solution of a three-layer quasi-geostrophic ocean circulation model.
We model the top-150 PCs of the spatio-temporal solutions using different methods, obtain forecasts, project them back to the physical space using EOFs, and then assess the accuracy, stability, and computational cost of theie solutions.
We obtain the forecasts on both short- and long-timescales, equal to 100 days and 200K days, respectively.
The assessment metrics considered are RMSE, Anomaly cross-correlation (both on space and time), climatology, variance, Frequency map, forecast horizon, and running time complexity.

This repository provides the source codes (all written in Python) of all the methods employed.
The dataset used for the modelling and analyses can be found in the respective Zenodo repository.
