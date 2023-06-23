import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CustomTrajDataset(Dataset):
    def __init__(self, traj_df):
        self.inputs = torch.from_numpy(np.array(list(traj_df['position']))).type(torch.FloatTensor)
        self.forces = torch.from_numpy(np.array(list(traj_df['net_force']))).type(torch.FloatTensor)

        self.in_dim = self.inputs.shape[-1]

        self.input_shape = self.inputs.shape

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.forces[i]



def _get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return dataloader


def load_datasets(data_path, batch_size):
    train_df = pd.read_pickle(os.path.join(data_path, 'train.pkl'))
    val_df = pd.read_pickle(os.path.join(data_path, 'val.pkl'))
    test_df = pd.read_pickle(os.path.join(data_path, 'test.pkl'))
    train_dataset = CustomTrajDataset(train_df)
    valid_dataset = CustomTrajDataset(val_df)
    test_dataset = CustomTrajDataset(test_df)

    train_dataloader = _get_data_loader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = _get_data_loader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = _get_data_loader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader