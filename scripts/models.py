#
#     models for OOI DAS Compression Project
#
#     https://github.com/niyiyu/DASCompression
#
#     Yiyu Ni niyiyu@uw.edu
#     Nov. 29, 2022
#
##############################################

import torch
from torch.nn import Linear, Parameter, Sigmoid


class RandomFourierFeatureNetwork(torch.nn.Module):
    def __init__(self, nfeature, n_layers, n_outputs = 1):
        super().__init__()
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.inputs = Linear(2 * nfeature, nfeature)

        # hidden layers
        for i in range(self.n_layers):
            setattr(self, f"ln{i+1}", Linear(nfeature, nfeature))

        self.outputs = Linear(nfeature, self.n_outputs)
        self.sigmoid = Sigmoid()
        self.relu = torch.relu

    def forward(self, x):
        x = self.relu(self.inputs(x))
        for i in range(self.n_layers):
            layer = getattr(self, f"ln{i+1}")
            x = self.relu(layer(x))
        x = self.sigmoid(self.outputs(x))
        return x


class SIREN(torch.nn.Module):
    def __init__(self, n_input, n_output, n_layers, n_units, omega):
        super().__init__()
        self.omega = omega
        self.n_layers = n_layers

        # input layer
        self.inputs = Linear(n_input, n_units, bias=False)
        self.inputs_bias = Parameter(torch.rand(n_units))

        # hidden layers
        for i in range(self.n_layers):
            setattr(self, f"ln{i+1}", Linear(n_units, n_units, bias=False))
            setattr(self, f"ln{i+1}_bias", Parameter(torch.rand(n_units)))

        # output layer
        self.outputs_bias = Parameter(torch.rand(n_output))
        self.outputs = Linear(n_units, n_output, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = torch.sin(self.omega * self.inputs(x) + self.inputs_bias)
        for i in range(self.n_layers):
            layer = getattr(self, f"ln{i+1}")
            bias = getattr(self, f"ln{i+1}_bias")
            x = torch.sin(self.omega * layer(x) + bias)
        x = self.sigmoid(self.outputs(x) + self.outputs_bias)

        return x
