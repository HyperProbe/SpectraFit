import torch
import numpy as np
import spectral as sp
import scipy
from scipy.ndimage import convolve1d
from pysptools.detection.detect import CEM

def get_array(img):
    '''
    Returns the image as a numpy array.
    input:
        img: image to convert, SpyFile or array-like
    output:
        image as numpy array as float32
    '''
    if isinstance(img, np.ndarray):
        return img.astype(np.float32)
    if isinstance(img, sp.io.bilfile.BilFile):
        img = img.asarray().astype(np.float32)
    else:
        try:
            img = np.asarray(img).astype(np.float32)
        except:
            raise ValueError("Unsupported input type")
    return img

def project_img(img, white_ref, dark_ref, device="cpu"):
    '''
    Project the image onto the subspace orthogonal to the illumination spectrum.
    input:
        img: image to project, shape (...,k) where k is the number of bands and ... are the spatial or time dimensions
        white_ref: white reference, shape (...,k) where m is the number of white reference pixels
        dark_ref: white reference, shape (...,k) where n is the number of dark reference pixels
    output:
        projected image as np.array
    '''
    img, white_ref, dark_ref = get_array(img), get_array(white_ref), get_array(dark_ref)
    # calculate reflectance
    R = np.subtract(img, dark_ref, dtype=np.float32)
    # calculate illumination spectrum E
    E = np.mean(np.subtract(white_ref, dark_ref, dtype=np.float32), axis=-2).squeeze()
    E = smooth_spectral(E, 5)
    # get mapping to subspace orthogonal to E
    P_E = np.eye(E.shape[0]) - np.outer(E, E)/np.dot(E, E)
    
    # apply mapping to data
    # create torch tensors and move to gpu
    R = torch.from_numpy(R).to(device).float()
    P_E = torch.from_numpy(P_E).to(device).float()

    # R_E = np.einsum('ij,klj->kli', P_E, R)
    R_E = torch.einsum('ij,...j->...i', P_E, R) 

    # convert back to numpy array
    R_E = R_E.cpu().numpy()

    return R_E

def project_absorbance(abs, endmembers_proj, endmembers_unmix, device="cpu"):
    '''
    Project the image and endmembers_unmix onto the subspace orthogonal to the spectra in endmembers_proj.
    input:
        abs: absorbance array to project, shape (...,k) where k is the number of bands and ... are the spatial or time dimensions
        endmembers: endmember spectra, shape (n, k), where n is the number of endmembers
    output:
        projected absorbance array perpendicular to all spectra, np.array
        projected endmembers_unmix, np.array
    '''
    P = np.eye(endmembers_proj.shape[1]) - endmembers_proj.T @ np.linalg.pinv(endmembers_proj).T
    # convert to torch tensors
    P = torch.from_numpy(P).to(device).float()
    abs = torch.from_numpy(abs).to(device).float()
    endmembers_unmix = torch.from_numpy(endmembers_unmix).to(device).float()
    # project data
    abs_proj = torch.einsum('ik,...k->...i', P, abs).cpu().numpy()
    endmembers_unmix_proj = torch.einsum('ik,nk->ni', P, endmembers_unmix).cpu().numpy()
    return abs_proj, endmembers_unmix_proj

def osp(abs, endmembers_proj, endmember_target, device="cpu"):
    '''
    Project the image spectra in the subspace orthogonal to the spectra in endmembers_proj.
    input:
        abs: absorbance array to project, shape (...,k) where k is the number of bands and ... are the spatial or time dimensions
        endmembers_proj: endmember spectra to remove, shape (n, k), where n is the number of endmembers
        endmember_target: target endmember to detect, shape (k,)
    output:
        abs_proj: target heatmap, shape (...)
    '''
    # make sure input is torch tensor
    was_numpy = isinstance(abs, np.ndarray)
    abs = torch.tensor(abs, device=device).float()
    endmembers_proj = torch.tensor(endmembers_proj, device=device).float()
    endmember_target = torch.tensor(endmember_target, device=device).float()

    # perform OSP
    P = torch.eye(endmembers_proj.shape[1], device=device) - endmembers_proj.T @ torch.pinverse(endmembers_proj).T
    abs_proj = torch.einsum('ik,...k->...i', P, abs)
    abs_proj = torch.einsum('k,...k->...', endmember_target, abs_proj)

    if was_numpy:
        abs_proj = abs_proj.cpu().numpy()

    return abs_proj

def icem(spectr, t, lmda=0, R=None, device='cpu'):
    '''
    Improved constrained energy minimization (ICEM) algorithm.
    input:
        spectr: image to unmix, shape (...,k) where k is the number of bands and ... are the spatial or time dimensions
        t: endmember to unmix, shape (k,)
        lmda: regularization parameter, float
    output:
        heatmap, np.array
    '''
    was_numpy = isinstance(spectr, np.ndarray)
    spectr = torch.tensor(spectr, device=device).float()
    t = torch.tensor(t, device=device).float()

    input_shape = spectr.shape
    M = spectr.reshape(-1, input_shape[-1])

    N, p = M.shape
    if R is None:
        R = torch.mm(M.T, M) / N
    else:
        R = torch.tensor(R, device=device).float()
    R_hat = R + lmda * torch.eye(p, device=device)
    Rinv = torch.inverse(R_hat)
    t_Rinv = torch.mv(Rinv.T, t)
    denom = torch.dot(t, t_Rinv)
    y = torch.mv(M, t_Rinv) / denom

    if was_numpy:
        y =  y.cpu().numpy()

    return y.reshape(input_shape[:-1])

def cosine_similarity(abs, spectr):
    '''
    Calculate the cosine similarity between the absorbance and the endmember spectra.
    input:
        abs: absorbance array, shape (...,k) where k is the number of bands and ... are the spatial or time dimensions
        spectr: endmember spectra, shape (n, k), where n is the number of endmembers
    output:
        cosine similarity, np.array of shape (...)
    '''
    return np.einsum("...k,k->...", abs, spectr) / (np.linalg.norm(abs, axis=2) * np.linalg.norm(spectr))

def similarity(abs, spectr):
    '''
    Calculate the similarity between the absorbance and the endmember spectra.
    input:
        abs: absorbance array, shape (...,k) where k is the number of bands and ... are the spatial or time dimensions
        spectr: endmember spectra, shape (n, k), where n is the number of endmembers
    output:
        similarity, np.array of shape (...)
    '''
    return np.einsum("...k,k->...", abs, spectr)

def calibrate_img(img, white_ref, dark_ref):
    '''
    Calibrate the image using the white and dark references.
    input:
        img: image to calibrate, shape (...,k) where k is the number of bands and ... are the spatial or time dimensions
        white_ref: white reference, shape (...,k) where m is the number of white reference pixels
        dark_ref: white reference, shape (...,k) where n is the number of dark reference pixels
    output:
        calibrated image as np.array
    '''
    img, white_ref, dark_ref = get_array(img), get_array(white_ref), get_array(dark_ref)
    # calculate reflectance
    R = np.subtract(img, dark_ref, dtype=np.float32)
    # calculate illumination spectrum E
    # first subract dark reference from white reference pixel-wise to get rid of pixel differences, then average over pixels to minimize noise
    E = np.mean(np.subtract(white_ref, dark_ref, dtype=np.float32), axis=-2, keepdims=True)
    img_calibrated = np.divide(R,E)
    return img_calibrated

def band_removal(img, new_range, orig_bands=None):
    '''
    Remove bands from the image that are not in the specified range.
    input:
        img: image to convert, shape (...,k) where k is the number of bands and ... are the spatial or time dimensions
        new_range: range of bands to keep, list containing min and max band
        orig_bands: original band centers, list
    output:
        image as numpy array
    '''
    if isinstance(img, sp.io.bilfile.BilFile):
        orig_bands = img.bands.centers
    if orig_bands is None:
        raise ValueError("orig_bands must be provided when img is not a SpyFile")
    img = get_array(img)
    orig_bands = np.array(orig_bands).squeeze()
    idx = (orig_bands >= new_range[0]) & (orig_bands <= new_range[1])
    img_cropped = img[...,idx]
    new_bands = orig_bands[idx]
    return img_cropped, new_bands

def calibrate_img_advanced(img, white_ref, dark_ref, average_ref_pixels=False):
    '''
    Calibrate the image using the white and dark references using normalization from https://link.springer.com/chapter/10.1007/978-3-031-18256-3_43.
    input:
        img: image to calibrate, np.array or SpyFile
        white_ref: white reference, np.array or SpyFile
        dark_ref: white reference, np.array or SpyFile
        average_ref_pixels: if True, average the white and dark references over the input image before calibration
    output:
        calibrated image as np.array
    '''
    img, white_ref, dark_ref = get_array(img), get_array(white_ref), get_array(dark_ref)
    if average_ref_pixels:
        white_ref = np.mean(white_ref, axis=(0,1))
        dark_ref = np.mean(dark_ref, axis=(0,1))
        white_ref = np.tile(white_ref, (img.shape[1], 1))
        dark_ref = np.tile(dark_ref, (img.shape[1], 1))
    alpha = np.subtract(img, dark_ref, dtype=np.float32)
    beta = np.subtract(white_ref, dark_ref, dtype=np.float32) - np.min(alpha)
    beta_hat = np.divide(beta, np.max(beta, axis=(0,1)))
    alpha_hat = np.divide(alpha - np.min(alpha), beta_hat)
    img_calibrated = np.divide(alpha_hat, np.max(alpha_hat, axis=(0,1))) * 100
    return img_calibrated

def l1_normalize(img):
    '''
    Normalize the image spectral signatures pixel-wise to unit L1 norm.
    Spectral dimension must be the last dimension.
    input:
        img: image to normalize, np.array or SpyFile
    output:
        image as np.array
    '''
    img = get_array(img)
    img_norm = np.divide(img, np.linalg.norm(img, ord=1, axis=-1, keepdims=True))
    return img_norm

def normalize_bands_std(img, class_wise=False, gt_map=None):
    '''
    Normalize the image spectral signatures band-wise to unit variance.
    '''
    img, gt_map = get_array(img), get_array(gt_map)
    img_norm = np.zeros_like(img)
    if class_wise:
        if gt_map is None:
            raise ValueError("gt_map must be provided when class_wise is True")
        classes = np.unique(gt_map)
        for class_id in classes:
            mask = np.where(gt_map[:,:,0] == class_id)
            class_std = np.std(img[mask], axis=0)
            class_mean = np.mean(img[mask], axis=0)
            img_norm[mask] = (img[mask] - class_mean) / class_std + class_mean
    else:
        img_norm = (img - np.mean(img, axis=(0,1))) / np.std(img, axis=(0,1)) + np.mean(img, axis=(0,1))
    return img_norm

def smooth_spectral(img, window_size=5):
    '''
    Smooth the image using a mean filter.
    input:
        img: image to smooth, np.array or SpyFile
        window_size: size of the smoothing window, int
    output:
        smoothed image as np.array
    '''
    img = get_array(img)
    kernel = np.ones(window_size)/window_size
    img_smooth = convolve1d(img, kernel, mode='nearest', axis=-1)
    return img_smooth

def bands_lin_interpolation(spectr, bands_old, range_new):
    """
    interpolate spectrogram values to new bands
    input:  spectr, shape (...,k) where k is the number of bands and ... are the spatial or time dimensions
            bands_old, bands of the original spectrogram
            range_new, new bands range
    output: mu_new, dictionary with interpolated spectrogram values
            bands_new, new bands with stepsize 1nm
    """
    spectr = get_array(spectr)
    bands_new = np.arange(range_new[0], range_new[1] + 1)
    if bands_old[0] > bands_new[0] or bands_old[-1] < bands_new[-1]:
        raise ValueError("Interpolation range is out of bounds")
    if spectr.ndim == 1:
        spectr_new = np.interp(bands_new, bands_old, spectr)
    if spectr.ndim == 2:
        spectr_new = np.zeros((spectr.shape[0], bands_new.shape[0]))
        for i in range(spectr.shape[0]):
            spectr_new[i,:] = np.interp(bands_new, bands_old, spectr[i,:])
    if spectr.ndim == 3:
        spectr_new = np.zeros((spectr.shape[0], spectr.shape[1], bands_new.shape[0]))
        for i in range(spectr.shape[0]):
            for j in range(spectr.shape[1]):
                spectr_new[i,j,:] = np.interp(bands_new, bands_old, spectr[i,j,:])
    return spectr_new, bands_new

def to_absorbance(img):
    '''
    Convert image spectral signatures to absorbance.
    A(lambda) = -log(img(lambda)).
    input:
        img: image to convert, np.array or SpyFile
    output:
        image as np.array
    '''
    img = get_array(img)
    img_absorbance = -np.log(img)
    return img_absorbance

def normalize_spectral_interval(img):
    '''
    Normalize the image spectral signatures pixel-wise to the interval [0,1].
    input:
        img: image to normalize, np.array or SpyFile
    output:
        image as np.array
    '''
    img = get_array(img)
    img_min = np.min(img, axis=-1)[..., np.newaxis]
    img_max = np.max(img, axis=-1)[..., np.newaxis]
    img_spectral_norm = np.divide(
        np.subtract(img, img_min),
        np.subtract(img_max, img_min))
    return img_spectral_norm

def normalize_spectral_interval_mean(img, class_wise=False, gt_map=None):
    '''
    Normalize the image spectral signatures mean to [0,1].
    If class_wise is True, normalize each class separately.
    input:
        img: image to normalize, np.array or SpyFile
        class_wise: if True, normalize each class separately, bool
        gt_map: ground truth map, np.array or SpyFile
    output:
        image as np.array
    '''
    img = get_array(img)
    if class_wise:
        img_spectral_norm = np.zeros_like(img)
        if gt_map is None:
            raise ValueError("gt_map must be provided when class_wise is True")
        gt_map = get_array(gt_map)
        classes = np.unique(gt_map)
        for class_id in classes:
            mask = np.where(gt_map[:,:,0] == class_id)
            class_mean = np.mean(img[mask], axis=0)
            class_mean_min = np.min(class_mean, axis=0)
            class_mean_max = np.max(class_mean, axis=0)
            img_spectral_norm[mask] = np.divide(
                np.subtract(img[mask], class_mean_min),
                np.subtract(class_mean_max, class_mean_min))
    else:
        spectral_mean = np.mean(img, axis=(0,1))
        mean_min = np.min(spectral_mean)
        mean_max = np.max(spectral_mean)
        img_spectral_norm = np.divide(
            np.subtract(img, mean_min),
            np.subtract(mean_max, mean_min))
    return img_spectral_norm