# import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer

from euler_fourier_1d import FNO1d
from euler_u_net import UNet
from utilities3 import *
from Adam import Adam


torch.manual_seed(0)
np.random.seed(0)

################################################################
#  configurations
################################################################
# Flags
save = True  # save model and predictions

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
learning_rate = 0.0005
step_size = 50
gamma = 0.5
weight_decay=1e-4


# models
model = FNO1d(modes, width).cuda()
# model = UNet().cuda()

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
print("Model selected: ", model.name)
print("Model parameters; ", count_params(model))

path = f'euler_{model.name}_sub{sub}_ep{epochs}_b{batch_size}_lr{learning_rate}_g{gamma}'
path_model = 'model/' + path
path_pred = 'pred/' + path
path_plot = 'pred/' + path

######################################################################
# load & preprocess data
######################################################################
DATA_PATH = 'data/EulerData_not_in_structure.mat'
dataloader = MatReader(DATA_PATH)
x_data = dataloader.read_field('a')[:,::sub,:]  # index along variable dimension
y_data = dataloader.read_field('u')[:,::sub,:]

x_train = x_data[:ntrain,:,:]  # index along variable dimension
y_train = y_data[:ntrain,:,:]
x_test = x_data[-ntest:,:,:]
y_test = y_data[-ntest:,:,:]

# Normalize the data (might be wrong dimension...)
x_normalizer = EulerNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
x_normalizer.cuda()

y_normalizer = EulerNormalizer(y_train)
y_train = y_normalizer.encode(y_train)
y_test = y_normalizer.encode(y_test)  # loss in terms of normalized output
y_normalizer.cuda()

# y_train = x_normalizer()
# x_train = F.normalize(x_train, p=2.0, dim=0)  # normalize along spatial coordinates
# y_train = F.normalize(y_train, p=2.0, dim=0)
# x_test = F.normalize(x_test, p=2.0, dim=0)
# y_test = F.normalize(y_test, p=2.0, dim=0)

# x_train = x_train.reshape(ntrain,s,1)
# x_test = x_test.reshape(ntest,s,1)

# HOKAJ: need to make sure shuffle is along axis 0
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)


# ################################################################
# # training and evaluation
# ################################################################



# save evaluation metrics
train_losses = np.zeros(epochs)
test_losses = np.zeros(epochs)


myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        mse = F.mse_loss(out.contiguous().view(batch_size, -1), y.contiguous().view(batch_size, -1), reduction='mean')
        l2 = myloss(out.contiguous().view(batch_size, -1), y.contiguous().view(batch_size, -1))
        l2.backward() # use the l2 relative loss

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            test_l2 += myloss(out.contiguous().view(batch_size, -1), y.contiguous().view(batch_size, -1)).item()


    train_mse /= (nvars * len(train_loader))
    train_l2 /= (nvars * ntrain)
    test_l2 /= (nvars * ntrain)

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)
    # save performance metrics
    train_losses[ep] = train_l2
    test_losses[ep] = test_l2

save_flag = input("Save model? (y/n): ")
if save_flag == 'y':
    torch.save(model, path_model)
    print("Model saved as: ", path_model)

pred = torch.zeros(y_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x).contiguous().view(s, nvars)   # drop unused first dimension

        test_l2 += myloss(out.contiguous().view(1, -1), y.contiguous().view(1, -1)).item()
        print(index, test_l2)

        out = y_normalizer.decode(out)
        pred[index] = out

        index = index + 1

save_flag = input("Save predictions? (y/n): ")
if save_flag == 'y':    
    scipy.io.savemat(path_pred, mdict={'y_hat': pred.cpu().numpy(),
                                                             'y'    : y_test.cpu().numpy(),
                                                             'x'    : x_test.cpu().numpy(),
                                                            })
    print("Predictions saved as: ", path_pred)

