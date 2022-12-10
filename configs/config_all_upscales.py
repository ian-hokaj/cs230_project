################################################################
#  configurations
################################################################
which_models = ['Mod1']  # ['FNO1', 'UNet', 'Mod1']
which_losses = ["L2", 'L1', 'Sobolev']  # ('L1', 'L2', 'Sobolev')

# Data parameters
ntrain = 1000
ntest = 100
R = 13
sub = 2**5 #subsampling rate
upscale_subs = [2**5, 2**4, 2**3, 2**2, 2**1, 2**0]

# training hyperparameters
epochs = 500
batch_size = 20
learning_rate = 0.005
gamma = 0.5

# FNO model parameters
modes = 16
width = 64

# Loss function parameters
lam = 0.01  # sobolev: weighting on gradient norm

nvars = 3
grid_res = 2**R
# s = grid_res // upscale_sub