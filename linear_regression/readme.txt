#####################################################
A Linear Regression (LR) model - the baseline 
#####################################################

Setup ALPHA:

PCs considered: 150
Length of each PC: 50K data points, i.e., 500K days
Training Length: 40K data points
Test Length: 10K data points
Number of ICs for short-time forecasts: test length - forecast lead time
Number of ICs for long-time forecasts: 1
Normalization considered: Division by the standard deviation of the top PC

Model input: model state at t
Model output: model tendency at time t