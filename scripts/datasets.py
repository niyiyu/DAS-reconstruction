#
#     torch dataset class for OOI DAS Compression Project
#
#     https://github.com/niyiyu/DASCompression
#
#     Yiyu Ni niyiyu@uw.edu
#     Nov. 29, 2022
#
##############################################

import torch 
import numpy as np

class OOIDASDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        'Initialization'
        if isinstance(inputs, torch.Tensor):
            self.inputs = inputs
            self.outputs = outputs
        else:
            self.inputs = inputs.astype(np.float32)
            self.outputs = outputs.astype(np.float32)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.outputs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.inputs[index, :]
        y = self.outputs[index, :]

        return X, y
