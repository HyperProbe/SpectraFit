import pickle

import numpy as np
import scipy
import torch
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import config
import data_processing.preprocessing as prepro
import utils
from config import left_cut, right_cut


def load_Xwaves():
    """
    loads the available wavelengths from one example dataset (one piglet). !might not be consistent across all samples!
    """
    # path = "dataset/HSI_Human_Brain_Database_IEEE_Access/{}/raw.hdr".format('004-02')
    # x_waves = prepro.read_wavelengths(path)
    x_waves = scipy.io.loadmat(config.dataset_path + 'LWP483_10Jan2017_SharedHyperProbe.mat')['wavelengths'].astype(float)
    molecules, x = prepro.read_molecules(left_cut, right_cut, x_waves)
    return molecules, x


def save_data(syn_data, syn_params, name=""):
    """
    helper method to save synthetic dataset
    """
    with open(config.dataset_path + 'synthetic/spectra' + name + '.pkl', 'wb') as f:
        pickle.dump(torch.from_numpy(np.array(syn_data)).float(), f)
    with open(config.dataset_path + 'synthetic/params' + name + '.pkl', 'wb') as f:
        pickle.dump(torch.from_numpy(np.array(syn_params)).float(), f)


def get_params():
    """
    sample a random set of molecule concentrations (according to some rules)
    """
    #params = np.random.dirichlet(np.ones(config.molecule_count), size=1)[0]

    Hbb = np.random.dirichlet(np.ones(2), size=1)[0]
    oxyCCO = np.random.uniform(low=0.0, high=0.2)
    redCCO = np.random.uniform(low=0.0, high=0.2)
    Hbb *= (1 - (redCCO + oxyCCO))
    params = np.array([Hbb[0], Hbb[1], oxyCCO, redCCO])

    return params


def generate_dataset(n_samples=10000):
    """
    generates synthetic dataset of (spectrogram, param) pairs with n_samples
    """
    molecules, x = load_Xwaves()

    syn_data, syn_params = None, None
    for i in tqdm(range(n_samples)):
        params = get_params()
        if i == 0:
            syn_data = np.expand_dims(utils.beerlamb_multi(molecules, x, params, left_cut), axis=0)
            syn_params = np.expand_dims(params, axis=0)
        else:
            syn_data = np.concatenate([syn_data, np.expand_dims(utils.beerlamb_multi(molecules, x, params, left_cut), axis=0)])
            syn_params = np.concatenate([syn_params, np.expand_dims(params, axis=0)])
    save_data(syn_data, syn_params)


def generate_diff_dataset(n_samples=100000):
    """
    generates synthetic dataset of ((spectrogram - ref_spectrogram), params2-params) pairs with n_samples
    """
    molecules, x = load_Xwaves()

    syn_data, syn_params = None, None
    for i in tqdm(range(n_samples)):
        params = get_params()
        params2 = get_params()  # np.array([0.25,0.25,0.25,0.25])
        diff = np.expand_dims(np.expand_dims(np.squeeze(np.array(utils.beerlamb_multi(molecules, x, params, left_cut)))
                                             / np.squeeze(np.array(utils.beerlamb_multi(molecules, x, params2, left_cut))), axis=1), axis=0)
        if i == 0:
            syn_data = diff
            syn_params = np.expand_dims(params, axis=0) - np.expand_dims(params2, axis=0)
        else:
            syn_data = np.concatenate([syn_data, diff])
            syn_params = np.concatenate([syn_params, np.expand_dims(params, axis=0) - np.expand_dims(params2, axis=0)])
    save_data(syn_data, syn_params)


def generate_dataset_max_uniform(param_maxs, n_samples=10000):
    """
    generates synthetic dataset (spectrogram, param) pairs with n_samples.
    params are sampled from a uniform distribution with each parameter getting a max value from the array <param_maxs>
    """
    molecules, x = load_Xwaves()

    syn_data, syn_params = None, None
    for i in tqdm(range(n_samples)):
        params = np.array([np.random.uniform(0.0, m) for m in param_maxs])
        if i == 0:
            syn_data = np.expand_dims(utils.beerlamb_multi(molecules, x, params, left_cut), axis=0)
            syn_params = np.expand_dims(params, axis=0)
        else:
            syn_data = np.concatenate([syn_data, np.expand_dims(utils.beerlamb_multi(molecules, x, params, left_cut), axis=0)])
            syn_params = np.concatenate([syn_params, np.expand_dims(params, axis=0)])
    save_data(syn_data, syn_params)


def generate_spectrogram(params=None, interpolate=False):
    """
    generates one synthetic (spectrogram, params) pair
    """
    molecules, x = load_Xwaves()

    if params is None:
        params = get_params()
    syn_data = np.expand_dims(utils.beerlamb_multi(molecules, x, params, left_cut), axis=0)
    syn_params = np.expand_dims(params, axis=0)
    return x, torch.from_numpy(syn_data).float(), torch.from_numpy(syn_params).float()


def generate_and_compare_custom(real_spectrogram, param_list):
    """
    not very smart "brute force" method for finding optimal parameters.
    probably wont be used. if interested see notebook -> unused_code/dict_method.ipynb
    """
    molecules, x = load_Xwaves()

    syn_params, best, best_params = None, None, None
    mse_min = 1000000
    mse_list = []

    for i, params in enumerate(tqdm(param_list)):
        if i == 0:
            syn_spectrogram = utils.beerlamb_multi(molecules, x, params, left_cut)
            syn_params = np.expand_dims(params, axis=0)
        else:
            syn_spectrogram = utils.beerlamb_multi(molecules, x, params, left_cut)
            syn_params = np.concatenate([syn_params, np.expand_dims(params, axis=0)])
        syn_spectrogram = np.array(syn_spectrogram)
        syn_spectrogram /= np.max(syn_spectrogram)
        mse_new = mean_squared_error(real_spectrogram, syn_spectrogram)
        mse_list.append(mse_new)
        if mse_new < mse_min:
            best = syn_spectrogram
            best_params = params
            mse_min = mse_new

    return best, best_params, mse_min, real_spectrogram

# uncomment the next line to generate a new dataset
# generate_diff_dataset()
