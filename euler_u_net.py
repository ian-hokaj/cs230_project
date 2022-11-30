import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from timeit import default_timer
from utilities3 import *
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding='same',
                               )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # add valid padding, stride of 2?
        x = self.conv2(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.encoder_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)])
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = self.max_pool(x)
        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.transpose_convs = nn.ModuleList([nn.ConvTranspose1d(channels[i],
                                                                 channels[i+1],
                                                                 kernel_size=2,
                                                                 stride=2,
                                                                 padding=0,
                                                                 output_padding=0,
                                                                 ) for i in range(len(channels) - 1)])
        self.decoder_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)])

    def forward(self, x, encoder_outputs):
        for i in range(len(self.channels) - 1):
            # print("x before padding: ", x.shape)
            x = self.transpose_convs[i](x)
            residual = encoder_outputs[i]
            # print("x after padding: ", x.shape)
            # print("residual shape: ", residual.shape)
            x = torch.cat([x, residual], dim=1)
            # NEED TO CONFIRM DIMENSION
            x = self.decoder_blocks[i](x)
        return x


class UNet(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super(UNet, self).__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.final_conv = nn.Conv1d(decoder_channels[-1], 3, 1)

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        x = self.decoder(encoder_outputs[-1], encoder_outputs[::-1][1:])
        x = self.final_conv(x)
        return x

# TESTING
# enc_block = Block(1, 64)
# x         = torch.randn(1, 1, 572)
# print("Block test: ", enc_block(x).shape)

# encoder = Encoder(channels=(3,64,128,256,512,1024))
# x    = torch.randn(1, 3, 1024)
# ftrs = encoder(x)
# for ftr in ftrs: print("Encoder test: ", ftr.shape)

# decoder = Decoder(channels=(1024, 512, 256, 128, 64))
# x = torch.randn(1, 1024, 64)
# decoder(x, ftrs[::-1][1:]).shape

# # channels=(1024, 512, 256, 128, 64)
# # i = 0
# # x = torch.randn(1, 1024, 28)
# # x = nn.ConvTranspose1d(channels[i],
# #                    channels[i+1],
# #                    kernel_size=2,
# #                    stride=2,
# #                    padding=0,
# #                    output_padding=0,
# #                    )(x)
# # print(x.shape)

# unet = UNet((3,64,128,256,512,1024), (1024, 512, 256, 128, 64))
# x    = torch.randn(1, 3, 1024)
# unet(x).shape



################################################################
#  configurations
################################################################
ntrain = 100
ntest = 10
nvars = 3  # three variables in the system of PDEs

grid_res = 2**10
sub = 2**1 #subsampling rate
h = grid_res // sub #total grid size divided by the subsampling rate
s = h

batch_size = 20
learning_rate = 0.001

epochs = 50
step_size = 50
gamma = 0.5

modes = 16
width = 64


################################################################
# read data
################################################################

DATA_PATH = 'data/EulerData_not_in_structure.mat'
dataloader = MatReader(DATA_PATH)
x_data = dataloader.read_field('a')[:,::sub,:]  # index along variable dimension
y_data = dataloader.read_field('u')[:,::sub,:]


x_train = x_data[:ntrain,:,:]  # index along variable dimension
y_train = y_data[:ntrain,:,:]
x_test = x_data[-ntest:,:,:]
y_test = y_data[-ntest:,:,:]

# Normalize the data (might be wrong dimension...)
x_train = F.normalize(x_train, p=2.0, dim=1)  # normalize along spatial coordinates
y_train = F.normalize(y_train, p=2.0, dim=1)
x_test = F.normalize(x_test, p=2.0, dim=1)
y_test = F.normalize(y_test, p=2.0, dim=1)

# Swap dimensions for convolusion to occur along spatial dimension (-1)
x_train = x_train.reshape(ntrain, nvars, s)
x_test = x_test.reshape(ntest, nvars, s)
y_train = y_train.reshape(ntrain, nvars, s)
y_test = y_test.reshape(ntest, nvars, s)

# HOKAJ: need to make sure shuffle is along axis 0
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# model
model = UNet(encoder_channels=(3,64,128,256,512,1024), decoder_channels=(1024, 512, 256, 128, 64)).cuda()
print("Model parameters: ", count_params(model))


################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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

        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
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
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()


    train_mse /= (nvars * len(train_loader))
    train_l2 /= (nvars * ntrain)
    test_l2 /= (nvars * ntrain)

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)
    # save performance metrics
    train_losses[ep] = train_l2
    test_losses[ep] = test_l2


# torch.save(model, 'model/ns_fourier_burgers')
pred = torch.zeros(y_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x).view(nvars, s)   # drop unused first dimension
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)
        index = index + 1