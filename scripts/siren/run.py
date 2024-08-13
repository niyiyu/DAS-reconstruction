import argparse
import time

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Fitting GCI DAS data")
parser.add_argument("-f", "--file", required=True)
parser.add_argument("-c", "--starting_channel", type=int, required=True)
parser.add_argument("-g", "--gpu", type=int, required=True)

args = parser.parse_args()

filename = args.file
filetime = filename.split("_")[4]
channel = args.starting_channel
gpu_id = args.gpu

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import h5py
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader


class DASDataset(torch.utils.data.Dataset):
    def __init__(self, T, X, outputs):
        "Initialization"
        self.T = torch.Tensor(T)
        self.X = torch.Tensor(X)
        self.outputs = torch.Tensor(outputs)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.outputs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        t = self.T[index]
        x = self.X[index]
        y = self.outputs[index]

        return torch.concat([t, x], axis=-1), y


class SIREN(torch.nn.Module):
    def __init__(self, n_input, n_output, n_layers, n_units, omega):
        super().__init__()
        self.omega = omega
        self.n_layers = n_layers

        # input layer
        self.inputs = torch.nn.Linear(n_input, n_units, bias=False)
        self.inputs_bias = torch.nn.Parameter(torch.rand(n_units))

        # hidden layers
        for i in range(self.n_layers):
            setattr(self, f"ln{i+1}", torch.nn.Linear(n_units, n_units, bias=False))
            setattr(self, f"ln{i+1}_bias", torch.nn.Parameter(torch.rand(n_units)))

        # output layer
        self.outputs_bias = torch.nn.Parameter(torch.rand(n_output))
        self.outputs = torch.nn.Linear(n_units, n_output, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.sin(self.omega * self.inputs(x) + self.inputs_bias)
        for i in range(self.n_layers):
            layer = getattr(self, f"ln{i+1}")
            bias = getattr(self, f"ln{i+1}_bias")
            x = torch.sin(self.omega * layer(x) + bias)
        x = 2 * self.sigmoid(self.outputs(x) + self.outputs_bias) - 1

        return x


print(
    f"====REPORT: working on {filetime}, channel {channel}-{channel+1000}, GPU {gpu_id}===="
)

df = h5py.File(filename, "r")
data = df["/Acquisition/Raw[0]/RawData"][:, channel : channel + 1000].T
df.close()

nc, ns = data.shape
print(f"got {nc} channel and {ns} sample: {data.shape}")

data -= np.mean(data, axis=-1, keepdims=True)
data /= np.max(np.abs(data), axis=-1, keepdims=True)
data = torch.Tensor(data)

in_x, in_t = torch.meshgrid(torch.arange(nc), torch.arange(ns))
T = in_t / (ns - 1)
X = in_x / (nc - 1)

device = torch.device("cuda")

dataset = DASDataset(
    T.reshape([-1, 1]).to(device),
    X.reshape([-1, 1]).to(device),
    data.reshape([-1, 1]).to(device),
)

data_loader = DataLoader(dataset, batch_size=1024 * 16, shuffle=True, num_workers=0)

n_units = 128
n_layers = 20
n_input = 2
n_output = 1
omega = 30
model = SIREN(
    n_input=n_input, n_output=n_output, n_layers=n_layers, n_units=n_units, omega=omega
)

for name, mod in model.named_parameters():
    if "inputs" in name:  # for input layer
        if "bias" in name:
            mod.data.uniform_(-1 / np.sqrt(n_input), 1 / np.sqrt(n_input))
        elif "weight" in name:
            mod.data.uniform_(-1 / 2, 1 / 2)
    else:  # for hidden layer
        if "bias" in name:
            mod.data.uniform_(-1 / np.sqrt(n_units), 1 / np.sqrt(n_units))
        elif "weight" in name:
            mod.data.uniform_(
                -np.sqrt(6 / n_units) / omega, np.sqrt(6 / n_units) / omega
            )

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = MSELoss()

nepoch = 8
train_loss_log = []
t0 = time.time()
for t in range(nepoch):
    model.train()
    train_loss = 0

    for batch_id, batch in enumerate(data_loader):
        pred = model(batch[0].cuda())
        loss = loss_fn(pred, batch[1].cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(data_loader)

    train_loss_log.append(train_loss)
    print(f"{t}: train loss: %.6f" % train_loss)
    if train_loss < 1e-4:
        break
print(time.time() - t0)

torch.save(
    model.state_dict(), f"./weights/siren_{filetime}_{channel}-{channel+1000}.pt"
)
