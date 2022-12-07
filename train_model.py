# import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer

from euler_fourier_1d import FNO1d
from euler_mod_fourier_1d import Mod1
from euler_u_net import UNet
from config_train import *
from utilities3 import *
from Adam import Adam


torch.manual_seed(0)
np.random.seed(0)


# Select model
if which_model == 'FNO1':
    model = FNO1d(modes, width).cuda()
elif which_model == 'UNet':
    model = UNet().cuda()
elif which_model == 'Mod1':
    model = Mod1(modes, width).cuda()
else:
    print(f"Model {which_model} is not a valid selection")
    exit()

# Select loss
if which_loss == "L1":
    loss = nn.L1Loss(reduction='mean')
elif which_loss == "L2":
    loss = nn.MSELoss(reduction='mean')
elif which_loss == "Sobolev":
    loss = SobolevLoss(h=20/s, lam=10)  # grid is (-10, 10) with s points
else:
    print(f"Loss {which_loss} is not a valid selection")
    exit()

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
print("Model selected: ", model.name)
print("Model parameters; ", count_params(model))

path = f'euler_R{R}_{which_model}_{which_loss}_sub{sub}_ep{epochs}_b{batch_size}_lr{learning_rate}_g{gamma}'
path_model = 'model/' + path
path_pred = 'pred/' + path + '.mat'
# path_plot = 'pred/' + path

######################################################################
# load & preprocess data
######################################################################
DATA_PATH = f'data/EulerData_R{R}.mat'
dataloader = MatReader(DATA_PATH)
x_data = dataloader.read_field('a')[:,::sub,:]  # index along variable dimension
y_data = dataloader.read_field('u')[:,::sub,:]

x_train = x_data[:ntrain,:,:]  # index along variable dimension
y_train = y_data[:ntrain,:,:]
x_test = x_data[-ntest:,:,:]
y_test = y_data[-ntest:,:,:]
# print(x_train.shape)

# Normalize the data (might be wrong dimension...)
x_normalizer = EulerNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
x_normalizer.cuda()

y_normalizer = EulerNormalizer(y_train)
y_train = y_normalizer.encode(y_train)
y_test = y_normalizer.encode(y_test)  # loss in terms of normalized output
y_normalizer.cuda()

# HOKAJ: need to make sure shuffle is along axis 0
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)


# ################################################################
# # training and evaluation
# ################################################################



# save evaluation metrics
train_mses = np.zeros(epochs)
train_losses = np.zeros(epochs)
test_losses = np.zeros(epochs)


for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_loss = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)
        print(out.shape)

        mse = F.mse_loss(out.contiguous().view(batch_size, -1), y.contiguous().view(batch_size, -1), reduction='mean')
        batch_loss = loss(out.contiguous().view(batch_size, -1), y.contiguous().view(batch_size, -1))
        batch_loss.backward()

        optimizer.step()
        train_mse += mse.item()
        train_loss += batch_loss.item()

    scheduler.step()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            test_loss += loss(out.contiguous().view(batch_size, -1), y.contiguous().view(batch_size, -1)).item()

    # Average because they have 'sum' reduction, so losses summed over batch
    train_mse /= (len(train_loader))
    train_loss /= len(train_loader)
    test_loss /= len(test_loader)#(ntrain * s * nvars)

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_loss, test_loss)

    # save performance metrics
    train_mses[ep] = train_mse
    train_losses[ep] = train_loss
    test_losses[ep] = test_loss

save_flag = input("Save model? (y/n): ")
if save_flag == 'y':
    torch.save(model, path_model)
    print("Model saved as: ", path_model)

pred = torch.zeros(y_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_loss = 0
        x, y = x.cuda(), y.cuda()

        out = model(x).contiguous().view(s, nvars)   # drop unused first dimension

        test_loss += loss(out.contiguous().view(1, -1), y.contiguous().view(1, -1)).item()
        print(index, test_loss)

        out = y_normalizer.decode(out)
        pred[index] = out

        index = index + 1

save_flag = input("Save predictions? (y/n): ")
if save_flag == 'y':    
    scipy.io.savemat(path_pred, mdict={'y_hat': pred.cpu().numpy(),
                                       'y'    : y_test.cpu().numpy(),
                                       'x'    : x_test.cpu().numpy(),
                                       'train_mses': train_mses,
                                       'train_losses': train_losses,
                                       'test_losses': test_losses,
                                      })
    print("Predictions saved as: ", path_pred)

