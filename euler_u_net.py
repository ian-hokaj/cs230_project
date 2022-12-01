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
    def __init__(self, encoder_channels=(3,64,128,256,512,1024), decoder_channels=(1024, 512, 256, 128, 64)):
        super(UNet, self).__init__()
        self.name = "UNet"
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.final_conv = nn.Conv1d(decoder_channels[-1], 3, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # convolve on spatial dimension
        encoder_outputs = self.encoder(x)
        x = self.decoder(encoder_outputs[-1], encoder_outputs[::-1][1:])
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
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

