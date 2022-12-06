################################################################
#  configurations
################################################################
which_model = "FNO1"  # ['FNO1', 'UNet', 'Mod1']
which_loss = "L2"  # ('L1', 'L2', 'Sobolev')

# Data parameters
ntrain = 1000
ntest = 100
R = 10
sub = 2**3 #subsampling rate
upscale_sub = 2**0

# training hyperparameters
epochs = 500
batch_size = 20
learning_rate = 0.001
gamma = 0.5

# FNO model parameters
modes = 16
width = 64

nvars = 3
grid_res = 2**R
s = grid_res // upscale_sub