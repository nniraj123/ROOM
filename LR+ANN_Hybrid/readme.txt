#####################################################
A hybrid model for modelling LR residuals using ANN 
#####################################################

Setup ALPHA:

PCs considered: 150
Length of each PC: 50K data points, i.e., 500K days
Training Length: 40K data points
Test Length: 10K data points
Number of ICs for short-time forecasts: test length - forecast lead time
Number of ICs for long-time forecasts: 1
Normalization considered: Division by the standard deviation of the top PC

For the stochastic version:
Number of Ensembles for short-time forecasts: 100
Number of Ensembles for long-time forecasts: 100
Noise type: White

LR input: Model state at time t
LR output: Model tendency at t
ANN input: [Tendency residual at time t, model state at t]
ANN output: Tendency residual at time t+1