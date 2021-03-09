#####################################################
A Multi-layer Linear Regression (ML-LR) model 
#####################################################

Setup ALPHA:

PCs considered: 150
Length of each PC: 50K data points, i.e., 500K days
Training Length: 40K data points
Test Length: 10K data points
Number of levels: 2 (including the main level)
Number of ICs for short-time forecasts: test length - forecast lead time + 1
Number of ICs for long-time forecasts: 1
Number of Ensembles for short-/long-time forecasts: 100
Normalization considered: Division by the standard deviation of the top PC

Model input: model state at t
Model output: model tendency at time t
Noise type: White