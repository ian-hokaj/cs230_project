from euler_fourier_1d import *
from utilities3 import *
import matplotlib.pyplot as plt

# Hyperparmeters
# model_name = 'UNet'
model_name = 'FNO1'
R = 10
sub = 2**3 #subsampling rate
epochs = 100
batch_size = 20
learning_rate = 0.0005
gamma = 0.5
ntest = 100

# Get filepaths from hyperparameters
TEST_PATH = "data/EulerData_not_in_structure.mat"
path = f"euler_{model_name}_sub{sub}_ep{epochs}_b{batch_size}_lr{learning_rate}_g{gamma}"
path_model = 'model/' + path
path_pred = 'pred/' + path
path_plot = 'pred/' + path

# Load model from file
model = torch.load(path_model)
# print(model.count_params())

# Load the data
dataloader = MatReader(TEST_PATH)
x_test = dataloader.read_field('a')[-ntest:,::sub,:]
y_test = dataloader.read_field('u')[-ntest:,::sub,:]

x_normalizer = EulerNormalizer(x_test)
x_test = x_normalizer.encode(x_test)
y_test = x_normalizer.encode(y_test)
x_normalizer.cuda()


# test
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
myloss = LpLoss(size_average=False)
pred = torch.zeros(y_test.shape)
index = 0
with torch.no_grad():
    test_l2 = 0
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        out = model(x)
        loss = myloss(out.view(1, -1), y.view(1, -1)).item()
        test_l2 += loss
        print(index, loss)

        out = x_normalizer.decode(out)
        pred[index] = out

        index = index + 1


# print(test_l2/ntest)

path = 'eval'
# scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy(), 'u': y_test.cpu().numpy()})

