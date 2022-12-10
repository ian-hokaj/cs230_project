import torch.nn.functional as F
import matplotlib.pyplot as plt

from scripts.euler_fourier_1d import FNO1d
from scripts.euler_mod_fourier_1d import Mod1
from scripts.euler_u_net import UNet
# from euler_u_net import UNet
from scripts.utilities import *

################################################################
#  Load Configurations
################################################################
# Set configurations in config_eval.py
from configs.config_all_upscales import *

# Initialize script outputs
output_models = []
output_losses = []
output_subs = []
output_upscale_subs = []
output_upscale_avg_losses = []
output_upscale_avg_mses = []

path_eval_upscales = f'euler_R{R}_sub{sub}_ep{epochs}_b{batch_size}_lr{learning_rate}_g{gamma}'
path_eval_upscales = 'eval/all_upscales/' + path_eval_upscales + '.mat'

for which_model in which_models:
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

    

    # Iterate through increasing subsampling resolutions and test accuracy
    for upscale_sub in upscale_subs:
        s = grid_res // upscale_sub  # necessary for sobolev loss

        for which_loss in which_losses:
            # Select loss
            if which_loss == "L1":
                loss = nn.L1Loss(reduction='mean')
            elif which_loss == "L2":
                loss = nn.MSELoss(reduction='mean')
            elif which_loss == "Sobolev":
                loss = SobolevLoss(h=20/s, lam=lam)  # grid is (-10, 10) with s points
            else:
                print(f"Loss {which_loss} is not a valid selection")
                exit()

            # Get filepaths from hyperparameters
            TRAIN_PATH = f"data/EulerData_R{R}"
            TEST_PATH = TRAIN_PATH
            path = f'euler_R{R}_{which_model}_{which_loss}_sub{sub}_ep{epochs}_b{batch_size}_lr{learning_rate}_g{gamma}'
            path_model = 'model/' + path
            path_pred = 'pred/' + path + '.mat'

            # Load model from file
            print("Loading model: ", path_model)
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

            ################################################################
            #  Tester
            ################################################################
            test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                                    batch_size=1,
                                                    shuffle=False,
                                                    )
            index = 0
            batch_loss = 0
            batch_mse = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()
                    out = model(x)

                    batch_mse += F.mse_loss(out.contiguous().view(1, -1), y.contiguous().view(1, -1), reduction='mean').item()
                    batch_loss += loss(out.contiguous().view(1, -1), y.contiguous().view(1, -1)).item()
                    # batch_loss /= (nvars * s)
                    # test_mses.append(batch_mse.item())
                    # test_losses.append(batch_loss)
                    # print(index, batch_mse, batch_loss)

                    # Store decoded output values for visualization
                    # out = x_normalizer.decode(out)
                    # pred[index] = out

                    index = index + 1


            print(which_model, which_loss, upscale_sub)

            output_models.append(which_model)
            output_losses.append(which_loss)
            output_upscale_subs.append(upscale_sub)
            output_upscale_avg_losses.append(batch_loss / ntest)
            output_upscale_avg_mses.append(batch_mse / ntest)

scipy.io.savemat(path_eval_upscales, mdict={'models': output_models,
                                            'losses': output_losses,
                                            'upscale_subs': output_upscale_subs,
                                            'upscale_avg_losses': output_upscale_avg_losses,
                                            'upscale_avg_mses': output_upscale_avg_mses,
                                   })   