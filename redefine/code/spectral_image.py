import spectral as sp
from spectral import BandInfo
import numpy as np

class SpectralImage(np.ndarray):
    def __new__(cls, img, params=None, metadata=None, bands=None):
        # Check if img is of type spectral.io.bilfile.BilFile
        if isinstance(img, sp.io.bilfile.BilFile):
            img_data = img.asarray()
            bands = img.bands
            params = img.params()
            metadata = img.metadata
        elif isinstance(img, np.ndarray):
            img_data = img
            bands = bands
            metadata = metadata
            params = params
        elif isinstance(img, __main__.SpectralImage):
            return img.copy()
        else:
            raise ValueError("Unsupported input type")

        obj = np.asarray(img_data).view(cls)
        obj.metadata = metadata
        obj.processing_steps = []
        obj.bands = bands
        obj.nbands = params.nbands
        obj.nrows = params.nrows
        obj.ncols = params.ncols
        obj.dtype = params.dtype
        return obj

    @staticmethod
    def open_image(filename):
        img = sp.open_image(filename)
        return SpectralImage(img)

    def __repr__(self):
        return f"SpectralImage(shape={self.shape}, dtype={self.dtype}, nbands={self.params.nbands}, nrows={self.params.nrows}, ncols={self.params.ncols}, bands={self.bands})"


def calibrage_img(img, white_ref, dark_ref, average_ref_pixels=False):
    '''
    Calibrate the image using the white and dark references.
    input:
        img: image to calibrate, SpectralImage
        white_ref: white reference, np.array or SpectralImage
        dark_ref: white reference, np.array or SpectralImage
        average_ref_pixels: if True, average the white and dark references over the input image before calibration
    output:
        calibrated image as SpectralImage
    '''
    img_cal = SpectralImage(img)
    if average_ref_pixels:
        img_cal.processing_steps.append("Calibration with averaged references")
        white_ref = np.mean(white_ref, axis=(0,1))
        dark_ref = np.mean(dark_ref, axis=(0,1))
    else:
        img_cal.processing_steps.append("Calibration")
    img_cal = np.divide(
        np.subtract(img, dark_ref),
        np.subtract(white_ref, dark_ref))
    return 


def normalize_band_wise(img, class_wise=False, gt_map=None):
    '''

    '''
    img_norm = SpectralImage(img)
    if class_wise:
        img_norm.processing_steps.append("Band-wise normalization with class-wise mean and std")
        if gt_map is None:
            raise ValueError("gt_map must be provided when class_wise is True")
        classes = np.unique(gt_map)
        for class_id in classes:
            mask = np.where(gt_map[:,:,0] == class_id)
            class_std = np.std(img_corr[mask], axis=0)
            class_mean = np.mean(img_corr[mask], axis=0)
            img_corr[mask] = (img[mask] - class_mean) / class_std + class_mean
    else:
        img_norm.processing_steps.append("Band-wise normalization with global mean and std")
        img_norm = (img - np.mean(img, axis=(0,1))) / np.std(img, axis=(0,1)) + np.mean(img, axis=(0,1))
    return 


import spectral as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data_folder = "helicoid/005-01"
img = sp.open_image(data_folder + "/raw.hdr")

test_img = SpectralImage(img)
print(test_img.nrows)
print(np.mean(test_img))
print(type(test_img))
norm = normalize_band_wise(test_img) 
print(type(norm))

for i in range(4):
    img = i