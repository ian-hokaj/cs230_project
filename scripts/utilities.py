import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial

#################################################
# Utilities
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EulerNormalizer(object):
    def __init__(self, x):
        super(EulerNormalizer, self).__init__()
        lbs = [0.5, 10, 20000]
        ubs = [2, 100, 200000]
        self.lower_bounds = torch.tensor(lbs).reshape(1, 1, -1)
        self.upper_bounds = torch.tensor(ubs).reshape(1, 1, -1)
    
    def encode(self, x):
        x = (x - self.lower_bounds) / self.upper_bounds
        return x

    def decode(self, x):
        x = (x * self.upper_bounds) + self.lower_bounds
        return x

    def cuda(self):
        self.lower_bounds = self.lower_bounds.cuda()
        self.upper_bounds = self.upper_bounds.cuda()

    def cpu(self):
        self.lower_bounds = self.lower_bounds.cpu()
        self.upper_bounds = self.upper_bounds.cpu()


class SobolevLoss(object):
    def __init__(self, h, lam):
        self.p = 2
        self.h = h      # grid spacing
        self.lam = lam  # weighting gradient term 

    def __call__(self, x, y):
        N = x.shape[0]

        norm = torch.linalg.norm(x - y, dim=1)
        x = torch.reshape(x, (N, -1, 3))
        y = torch.reshape(y, (N, -1, 3))

        grad_x = (x[:, 2:, :] - 2*x[:, 1:-1, :] + x[:, :-2, :]) / self.h
        grad_y = (y[:, 2:, :] - 2*y[:, 1:-1, :] + y[:, :-2, :]) / self.h
        grad_x = torch.reshape(grad_x, (N, -1))
        grad_y = grad_y.reshape(N, -1)
        grad_norm = torch.linalg.norm(grad_x - grad_y, dim=1)

        # x = torch.reshape(x, (N, -1))
        # y = torch.reshape(y, (N, -1))
        # norm = torch.linalg.norm(x - y, dim=1)

        loss = torch.sum(torch.square(norm) + self.lam * torch.square(grad_norm))
        return loss

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c
