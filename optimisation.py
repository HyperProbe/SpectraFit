import os

import cvxpy as cp
import numpy as np
import scipy.io
import seaborn as sns
from tqdm import tqdm

import config
# print(os.getcwd())
# os.chdir('..')
import data_processing.preprocessing as preprocessing
import utils
from config import left_cut, right_cut


"""
Script the runs cvxpy optimisation, to find the param differences between two spectra
Last line saves the result for later use in NN training.
"""

sns.set()

# piglet name
pig_name = "pig1"

# Load and cut data
# img = scipy.io.loadmat(config.dataset_path+'LWP483_10Jan2017_SharedHyperProbe.mat')
img = scipy.io.loadmat('dataset/LWP483_10Jan2017_SharedHyperProbe.mat')

cut = 7830

wavelengths = img['wavelengths'].astype(float)
white_full = img['refSpectrum'].astype(float)
dark_full = img['DarkCount'].astype(float)
spectr = img['spectralDatameasured'].astype(float)

# dark/white normalisation
idx = (wavelengths >= left_cut) & (wavelengths <= right_cut)
wavelengths = wavelengths[idx]
spectr = spectr[idx.squeeze()]
dark_full = dark_full[idx.squeeze()]
white_full = white_full[idx.squeeze()]

os.mkdir(config.dataset_path + 'piglet_diffs/' + pig_name)


def optimisation(spectr1, spectr2, i):
    m = 4  # number of parameters (from 2 to 6)
    # np.random.seed(1)
    molecules, x = preprocessing.read_molecules(left_cut, right_cut, wavelengths)
    y_hbo2_f, y_hb_f, y_coxa, y_creda, _, _ = molecules
    M = np.transpose(np.vstack((np.asarray(y_hbo2_f),
                                np.asarray(y_hb_f),
                                np.asarray(y_coxa),
                                np.asarray(y_creda))))

    b = spectr2 / spectr1
    b = np.log(1 / np.asarray(b))
    X = cp.Variable(m)

    objective = cp.Minimize(cp.sum_squares(M @ X - b))
    constraints = [cp.abs(X[2] + X[3]) <= 0.01]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return -X.value, b


ref_spectr = (spectr[:, 0] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0])
ref_spectr[ref_spectr <= 0] = 0.0001

spectra_list = []
coef_list = []
for i in tqdm(range(1, cut + 1)):
    # if i not in [100,200,400,2000]: continue
    spectr2 = (spectr[:, i] - dark_full[:, 0]) / (white_full[:, 0] - dark_full[:, 0])
    spectr2[spectr2 <= 0] = 0.0001

    coef_diff, spect_diff = optimisation(ref_spectr, spectr2, i)

    spectra_list.append(spect_diff)
    coef_list.append(coef_diff)

# Save labled data
utils.save_optimization_data(ref_spectr, spectra_list, coef_list, folder=pig_name)
