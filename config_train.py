################################################################
#  configurations
################################################################
which_model = "UNet"  # ['FNO1', 'UNet', 'Mod1']
which_loss = "Sobolev"  # ('L1', 'L2', 'Sobolev')

ntrain = 1000
ntest = 100
nvars = 3  # three variables in the system of PDEs

R = 10
grid_res = 2**R
sub = 2**3 #subsampling rate
h = grid_res // sub #total grid size divided by the subsampling rate
s = h

# FNO model parameters
modes = 16
width = 64

# Optimizer and Schedule hyperparameters
epochs = 500
batch_size = 20
learning_rate = 0.001
step_size = 50
gamma = 0.5
weight_decay=1e-4