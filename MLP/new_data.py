from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from torch import Tensor

import numpy as np
import os

class HELICoiD(Dataset):
    #Constructor for initially loading
    def __init__(self,path):
        #df = read_csv(path, header=None)
        #df = np.load(path + 'y_and_params/' + file)
    
        self.spectra_dir_path = path
        self.dataset_size = len(os.listdir(self.spectra_dir_path))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        filepath = os.path.join(self.spectra_dir_path, str(idx) + ".npz")
        #print("idx: ", idx, " ,filepath: ", filepath)
        npz_data = np.load(filepath) 
        spect = npz_data['spectra'].astype('float32')
        params = npz_data['params'].astype('float32')
        #print(spect)
        #print(params)
        return spect, params

