import os
import pickle
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

class SyntheticScanDataset(Dataset):

    def __init__(self, pickles_path):
        self.pickles_path = pickles_path
        self.pickle_file_names = os.listdir(self.pickles_path)

    def __len__(self):
        return len(os.listdir(self.pickles_path))

    def __getitem__(self, idx):
        with open(os.path.join(self.pickles_path, self.pickle_file_names[idx]), 'rb') as p:
            scan = pickle.load(p)
            location_x, location_y = map(int, os.path.splitext(self.pickle_file_names[idx])[0].split('-'))
        sample = {'scan': Tensor(scan),
                  'location': Tensor((location_x, location_y))}
        return sample