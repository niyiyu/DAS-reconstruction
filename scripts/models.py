import torch


class SHRED(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.1
        )
        self.sdn1 = torch.nn.Linear(hidden_size, output_size // 2)
        self.sdn3 = torch.nn.Linear(output_size // 2, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.lstm(x)[1][0][-1]  # should be -1
        x = self.relu(self.sdn1(x))
        x = self.sdn3(x)
        return


class RandomFourierFeature(torch.nn.Module):
    def __init__(self, nfeature, n_layers, n_outputs=1):
        super().__init__()
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.inputs = torch.nn.Linear(2 * nfeature, nfeature)

        for i in range(self.n_layers):
            setattr(self, f"ln{i+1}", torch.nn.Linear(nfeature, nfeature))

        self.outputs = torch.nn.Linear(nfeature, self.n_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.relu

    def forward(self, x):
        x = self.relu(self.inputs(x))
        for i in range(self.n_layers):
            layer = getattr(self, f"ln{i+1}")
            x = self.relu(layer(x))
        x = 2 * self.sigmoid(self.outputs(x)) - 1
        return x


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
