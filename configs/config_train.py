################################################################
#  configurations
################################################################
which_model = "Mod1"  # ['FNO1', 'UNet', 'Mod1']
which_loss = "Sobolev"  # ('L1', 'L2', 'Sobolev')

ntrain = 1000
ntest = 100
nvars = 3  # three variables in the system of PDEs

R = 13
grid_res = 2**R
sub = 2**5 #subsampling rate
h = grid_res // sub #total grid size divided by the subsampling rate
s = h

# FNO model parameters
modes = 16
width = 64

# Loss function parameters
lam = 0.000  # sobolev: weighting on gradient norm

# Optimizer and Schedule hyperparameters
epochs = 500
batch_size = 20
learning_rate = 0.001
step_size = 50
gamma = 0.5
weight_decay=1e-4