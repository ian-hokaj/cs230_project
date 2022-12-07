import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Load configs and build file paths
from config_all_upscales import *

path = f'euler_sub{sub}_ep{epochs}_b{batch_size}_lr{learning_rate}_g{gamma}'
path_eval_upscales = 'eval/all_upscales/' + path + '.mat'
path_plot_upscales = 'plots/all_upscales/' + path + '.png'

################################################################
# Read Data & Plot
################################################################
mat = scipy.io.loadmat(path_eval_upscales)

models = mat['models']
losses = mat['losses']
# subs = mat['subs']
upscale_subs = mat['upscale_subs'][0]
upscale_avg_losses = mat['upscale_avg_losses'][0]
upscale_avg_mses = mat['upscale_avg_mses'][0]

print(models)
print(losses)
# Build dictionary of logical parameters
reformat = {}
for which_model in which_models:
    reformat[which_model] = {}
    for which_loss in which_losses:
        reformat[which_model][which_loss] = [[],[]]
for i in range(len(models)):
    model = models[i].strip()
    loss = losses[i].strip()  #remove white space bug
    reformat[model][loss][0].append(upscale_subs[i])
    reformat[model][loss][1].append(upscale_avg_mses[i])


for which_model in which_models:
    for which_loss in which_losses:
        plt.plot(reformat[which_model][which_loss][0], reformat[which_model][which_loss][1], label=f'Model: {which_model}, Loss: {which_loss}')
        plt.scatter(reformat[which_model][which_loss][0], reformat[which_model][which_loss][1])


plt.title(f'Train/Test Loss Over {epochs} Epochs')
plt.xlabel('Epoch Number')
plt.ylabel('Average L2 Norm')
plt.legend()
plt.savefig(path_plot_upscales)
plt.show()