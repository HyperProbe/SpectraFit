import os
import numpy as np
from preprocessing import bands_lin_interpolation

spectra_path = os.path.join(os.path.dirname(__file__), '../datasets/spectra/')

def read_spectrum(file_name):
    """
    reads spectra from txt file
    input: file_name
    output: mu, bands as np arrays
    """
    data = np.loadtxt(os.path.join(spectra_path, file_name)).T
    bands = data[0]
    mu = data[1]
    return mu, bands

def get_extinctions(band_range):
    """
    returns extinction spectra for fat (cm^-1), water (cm^-1), hemoglobin(cm^-1 mM^-1), cytochromes (cm^-1 mM^-1)
    input:  band_range, range of the extinction spectra bands
    output: extinction_dict, dictionary with extinction spectra
    """

    extinction_dict = {}

    # cytochrome c oxidised
    cyt_c_ox, _ = bands_lin_interpolation(*read_spectrum('cyt_c_ox_500_1000.txt'), band_range)
    extinction_dict['cyt_c_ox'] = cyt_c_ox
    # cytochrome c reduced
    cyt_c_red, _ = bands_lin_interpolation(*read_spectrum('cyt_c_red_500_1000.txt'), band_range)
    extinction_dict['cyt_c_red'] = cyt_c_red
    # cytochrome c difference
    cyt_c_diff = cyt_c_ox - cyt_c_red
    extinction_dict['cyt_c_diff'] = cyt_c_diff
    # cytochrome b oxidised
    cyt_b_ox, _ = bands_lin_interpolation(*read_spectrum('cyt_b_ox_402_997.txt'), band_range)
    extinction_dict['cyt_b_ox'] = cyt_b_ox
    # cytochrome b reduced
    cyt_b_red, _ = bands_lin_interpolation(*read_spectrum('cyt_b_red_402_997.txt'), band_range)
    extinction_dict['cyt_b_red'] = cyt_b_red
    # cytochrome b difference
    cyt_b_diff = cyt_b_ox - cyt_b_red
    extinction_dict['cyt_b_diff'] = cyt_b_diff
    # cytochrome oxibase oxidised
    cyt_oxi_ox, _ = bands_lin_interpolation(*read_spectrum('cyt_oxi_ox_520_999.txt'), band_range)
    extinction_dict['cyt_oxi_ox'] = cyt_oxi_ox
    # cytochrome oxibase reduced
    cyt_oxi_red, _ = bands_lin_interpolation(*read_spectrum('cyt_oxi_red_520_999.txt'), band_range)
    extinction_dict['cyt_oxi_red'] = cyt_oxi_red
    # cytochrome oxibase difference
    cyt_oxi_diff = cyt_oxi_ox - cyt_oxi_red
    extinction_dict['cyt_oxi_diff'] = cyt_oxi_diff

    # hemoglobin
    hb_1, _ = bands_lin_interpolation(*read_spectrum('hb_450_630.txt'),[450,630])
    hb_2, _ = bands_lin_interpolation(*read_spectrum('hb_600_800.txt'),[631,650])
    hb_3, _ = bands_lin_interpolation(*read_spectrum('hb_650_1042.txt'),[651,1042])
    hb, _ = bands_lin_interpolation(np.concatenate((hb_1*2.3025851,hb_2*2.3025851,hb_3*10000)), np.arange(450,1043), band_range)
    extinction_dict['hb'] = hb
    # oxyhemoglobin
    hbo2_1, _ = bands_lin_interpolation(*read_spectrum('hbo2_450_606.txt'),[450,606])
    hbo2_2, _ = bands_lin_interpolation(*read_spectrum('hbo2_600_800.txt'),[607,650])
    hbo2_3, _ = bands_lin_interpolation(*read_spectrum('hbo2_650_1042.txt'),[651,1042])
    hbo2, _ = bands_lin_interpolation(np.concatenate((hbo2_1*2.3025851,hbo2_2*2.3025851,hbo2_3*10000)), np.arange(450,1043), band_range)
    extinction_dict['hbo2'] = hbo2

    # water
    water_1, _ = bands_lin_interpolation(*read_spectrum('water_380_727.txt'), [380,600])
    water_2, _ = bands_lin_interpolation(*read_spectrum('water_600_1050.txt'), [601,1050])
    water, _ = bands_lin_interpolation(np.concatenate((water_1/100,water_2*2.3025851)), np.arange(380,1051), band_range)
    extinction_dict['water'] = water

    # fat
    fat, _ = bands_lin_interpolation(*read_spectrum('fat_429_1098.txt'), band_range)
    extinction_dict['fat'] = fat /100

    return extinction_dict
