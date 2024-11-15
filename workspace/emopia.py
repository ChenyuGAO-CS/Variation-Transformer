import torch
import numpy as np

class Emopia(torch.utils.data.Dataset):
    def __init__(self, path_data, pad_token):
        data = np.load(path_data)
        self.pieces = torch.from_numpy(data['x']).long()
        self.pad_token = pad_token

    def __getitem__(self, idx):
        x = self.pieces[idx][:-1]
        y = self.pieces[idx][1:]
        return x, y

    def __len__(self):
        return len(self.pieces)