import torch.nn.functional as F
import matplotlib.pyplot as plt

from scripts.euler_fourier_1d import FNO1d
from scripts.euler_mod_fourier_1d import Mod1
from scripts.euler_u_net import UNet
from scripts.utilities import *

################################################################
#  Load Configurations
################################################################
# Set configurations in config_eval.py
from configs.config_eval import *

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
    loss = SobolevLoss(h=20/s, lam=1)  # grid is (-10, 10) with s points
else:
    print(f"Loss {which_loss} is not a valid selection")
    exit()

# Get filepaths from hyperparameters
TRAIN_PATH = f"data/EulerData_R{R}.mat"
TEST_PATH = TRAIN_PATH
path = f'euler_R{R}_{which_model}_{which_loss}_sub{sub}_ep{epochs}_b{batch_size}_lr{learning_rate}_g{gamma}'
path_model = 'model/' + path
path_pred = 'pred/' + path + '.mat'
path_eval = 'eval/' + f'upscale{upscale_sub}_' + path + '.mat'

# Load model from file
model = torch.load(path_model)
# print(model.count_params())

# Load the data
dataloader = MatReader(TEST_PATH)
x_train = dataloader.read_field('a')[:ntrain,::upscale_sub,:]
y_train = dataloader.read_field('u')[:ntrain,::upscale_sub,:]

dataloader = MatReader(TEST_PATH)
x_test = dataloader.read_field('a')[-ntest:,::upscale_sub,:]
y_test = dataloader.read_field('u')[-ntest:,::upscale_sub,:]

# Normalize
x_normalizer = EulerNormalizer(x_test)
x_train = x_normalizer.encode(x_train)
y_train = x_normalizer.encode(y_train)

x_test = x_normalizer.encode(x_test)
y_test = x_normalizer.encode(y_test)

x_normalizer.cuda()

# Outputs to save
pred = torch.zeros(y_test.shape)
test_losses = []
test_mses = []
avg_test_loss = 0
avg_test_mse = 0

################################################################
#  Tester
################################################################
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=1,
                                          shuffle=False,
                                         )
index = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        out = model(x)

        batch_mse = F.mse_loss(out.contiguous().view(1, -1), y.contiguous().view(1, -1), reduction='mean')
        batch_loss = loss(out.contiguous().view(1, -1), y.contiguous().view(1, -1)).item()
        # batch_loss /= (nvars * s)
        test_mses.append(batch_mse.item())
        test_losses.append(batch_loss)
        print(index, batch_mse.item(), batch_loss)

        # Store decoded output values for visualization
        out = x_normalizer.decode(out)
        pred[index] = out

        index = index + 1

test_losses = np.array(test_losses)
test_mses = np.array(test_mses)

print("Average Test loss: ", np.mean(test_losses))
print("Average Test MSE: ", np.mean(test_mses))


x_normalizer.cpu()
x_test = x_normalizer.decode(x_test)
y_test = x_normalizer.decode(y_test)

scipy.io.savemat(path_eval, mdict={'y_hat': pred.cpu().numpy(),
                                   'y'    : y_test.cpu().numpy(),
                                   'x'    : x_test.cpu().numpy(),
                                   'test_losses': test_losses,
                                   'test_mses': test_mses,
                                   })
