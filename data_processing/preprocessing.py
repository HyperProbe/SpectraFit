import bisect
import os
import numpy as np

from config import spectra_path


# load spectra from txt files into dictionary
def load_spectra():
    path_absorp = os.getcwd() + spectra_path
    path_dict = {}
    path_dict["cytoa_oxy"] = path_absorp + "moody cyt aa3 oxidised.txt"
    path_dict["cytoa_red"] = path_absorp + "moody cyt aa3 reduced.txt"
    path_dict["hbo2"] = path_absorp + "hb02.txt"
    path_dict["hbo2_450"] = path_absorp + "z_adult_hbo2_450_630.txt"
    path_dict["hbo2_600"] = path_absorp + "z_adult_hbo2_600_800.txt"
    path_dict["hb"] = path_absorp + "hb.txt"
    path_dict["hb_450"] = path_absorp + "z_adult_hb_450_630.txt"
    path_dict["hb_600"] = path_absorp + "z_adult_hb_600_800.txt"
    path_dict["water"] = path_absorp + "matcher94_nir_water_37.txt"
    path_dict["fat"] = path_absorp + "fat.txt"
    return path_dict


### reading cpectra from .txt
def read_spectra(file_name):
    with open(file_name, 'r') as data:
        x, y = [], []
        for line in data:
            p = line.split()
            if not p[0] == '\x00':
                x.append(float(p[0]))
                y.append(float(p[1]))
    return np.array(x), np.array(y)



def cut_spectra(x, y, left_cut, right_cut):
    """
    cuts off spectrogram according to cut values
    """
    ix_left = np.where(x == left_cut)[0][0]
    ix_right = np.where(x == right_cut)[0][0]
    return y[ix_left:ix_right + 1]


def wave_interpolation(y, x, mol_list, x_waves):
    """
    interpolate spectrogram values according to x_waves
    """
    lower_bound, upper_bound = x[0], x[-1]
    new_x = np.asarray([i for i in x_waves if lower_bound < i < upper_bound])

    new_y = {}
    for i in mol_list:
        new_y[i] = np.interp(new_x, x, y[i])

    return new_y, new_x


def read_molecules(left_cut, right_cut, x_waves=None):
    path_dict = load_spectra()

    # read spectra for: cytochrome oxydised/reduced, oxyhemoglobin, hemoglobin, water, fat
    mol_list = ["cytoa_oxy", "cytoa_red", "hbo2", "hbo2_450", "hbo2_600", "hb", "hb_450", "hb_600", "water", "fat"]
    x, y = {}, {}
    for i in mol_list:
        x[i], y[i] = read_spectra(path_dict[i])

    # from extinction to absorption
    # TODO check if water spectra was in extinction 
    y_list = ['hb_450', 'hb_600', 'hbo2_450', 'hbo2_600', 'cytoa_oxy', 'cytoa_red', 'water']
    for i in y_list:
        y[i] *= 2.3025851

    # from mm and micromole to cm and minimole, get rid of mole
    y["hbo2"] *= 10 * 1000 #/ 10
    y["hb"] *= 10 * 1000 #/ 10

    #y["cytoa_oxy"] /= 1000
    #y["cytoa_red"] /= 1000

    # from m to cm
    y["fat"] /= 100

    # interpolate till 800nm as the data have value for every second nm, 400,402,404...
    for i in ["hbo2", "hb"]:
        xvals = np.array([i for i in range(int(x[i + "_450"][0]), int(x[i + "_600"][-1]) + 1)])
        yinterp = np.interp(xvals, np.concatenate([x[i + "_450"], x[i + "_600"]]), np.concatenate([y[i + "_450"], y[i + "_600"]]))
        x[i] = np.concatenate([xvals, x[i][151:]])
        y[i] = np.concatenate((yinterp, np.asarray(y[i][151:])), axis=None)

    # cutting all spectra to the range [left_cut, right_cut] nm
    x_new = x["cytoa_oxy"][bisect.bisect_left(x["cytoa_oxy"], left_cut):bisect.bisect_right(x["cytoa_oxy"], right_cut)]
    mol_list = ["hbo2", "hb", "cytoa_oxy", "cytoa_red", "water", "fat"]
    for i in mol_list:
        y[i] = cut_spectra(x[i], y[i], left_cut, right_cut)

    if x_waves is not None:
        y, x_new = wave_interpolation(y, x_new, mol_list, x_waves)

    return [y[i] for i in mol_list], x_new


def read_wavelengths(path):
    f = open(path, "r")
    txt = f.read().split("{")[-1].strip("}")
    txt = txt.split(",")
    x = [float(i) for i in txt]
    return x
