import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from scripts.utilities import MatReader


################################################################
# configs
################################################################
from configs.config_eval import *

path = f'euler_R{R}_{which_model}_{which_loss}_sub{sub}_ep{epochs}_b{batch_size}_lr{learning_rate}_g{gamma}'
path_pred = 'pred/'+path+'.mat'
path_plot = 'plots/' + path + '.png'

################################################################
# Read Data & Plot
################################################################
mat = scipy.io.loadmat(path_pred)
train_l2s = mat['train_mses'][0]
test_l2s = mat['test_mses'][0]
arange_epochs = np.arange(epochs)

plt.plot(arange_epochs, train_l2s, label='Training Set MSE')
plt.plot(arange_epochs, test_l2s, label='Test Set MSE')
plt.yscale('log')
plt.title(f'Train/Test MSE Over {epochs} Epochs with {which_loss} Loss')
plt.xlabel('Epoch Number')
plt.ylabel('Average MSE')
plt.legend()
plt.savefig(path_plot)
plt.show()