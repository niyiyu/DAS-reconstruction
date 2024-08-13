import numpy as np
import torch


class DASDataset(torch.utils.data.Dataset):
    """
    torch dataset class for SHallow REcurrent Network.
    """

    def __init__(self, inputs, outputs):
        if isinstance(inputs, torch.Tensor):
            self.inputs = inputs
            self.outputs = outputs
        else:
            self.inputs = inputs.astype(np.float32)
            self.outputs = outputs.astype(np.float32)

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        X = self.inputs[index, :]
        Y = self.outputs[index, :]

        return X, Y


class DASDataset2DOnTheFly(torch.utils.data.Dataset):
    """
    torch dataset class for Random Fourier Feature Network.
    """

    def __init__(self, B, T, X, outputs):
        self.B = torch.Tensor(B)
        self.T = torch.Tensor(T)
        self.X = torch.Tensor(X)
        self.outputs = torch.Tensor(outputs)

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        _t = self.T[index]
        _x = self.X[index]
        _Bv = torch.matmul(torch.Tensor(torch.concat([_t, _x], axis=-1)), self.B.T)
        X = torch.cat(
            [torch.cos(2 * torch.pi * _Bv), torch.sin(2 * torch.pi * _Bv)], axis=-1
        )
        Y = self.outputs[index, :]

        return X, Y


class DASDataset2D(torch.utils.data.Dataset):
    """
    torch dataset class for SIREN model.
    """

    def __init__(self, T, X, outputs):
        self.T = torch.Tensor(T)
        self.X = torch.Tensor(X)
        self.outputs = torch.Tensor(outputs)

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        # Select sample
        T = self.T[index]
        X = self.X[index]
        Y = self.outputs[index]

        return torch.concat([T, X], axis=-1), Y
