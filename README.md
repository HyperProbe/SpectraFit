# Setup

1. Clone the framework repository
2. Create virtual environment
3. Execute [setup.py](http://setup.py) (also installs requirements)
4. Add `fat.txt` to `/dataset/UCL-NIR-Spectra/spectra` folder
5. Add `LWP483_10Jan2017_SharedHyperProbe.mat` to `/dataset` folder

# Framework structure

## Data Processing

`convert_dataset.py` 

Converts and saves Helicoid `.hdr` files into pytorch tensors

---

`generate_dataset.py`

Contains different methods for generating synthetic spectra datasets and samples (2d and 1d, and coefficient distributions)

---

`datasets.py`

Different torch Dataset classes. with [Spectrum, coefficient] pairs and [Spectrum, segmentation label] pairs

---

`preprocessing.py`

Read molecule specific spectra, Read wavelengths of helicoid dataset, interpolation

## Training and Optimisation Framework

`models.py`

contains MLP (1d input) and CNN (2d input) model for finding coefficients

---

`optimisation.py`

Loads dataset and runs convex optimisation

---

`train.py` & `train_cnn.py`

Runs MLP and CNN training respectively (with synthetic data)

---

`train_helicoid.py`

Runs training with Helicoid dataset

---

`utils.py`

Utilization methods (e.g. Beer-Lambert calculation)

---

`config.py`

Stores global variables used in different scripts

## Notebooks

`model_test.ipynb`

Tests CNN model by plotting predicted and gt spectrograms

---

`optimization.ipynb`

Tests optimisation on Helicoid data (doesn't give good results)

---

`dict_method.py`

Implements brute force method for finding coefficients (generates many possibilities for parameter tuples and matches the best combination to groundtruth, by comparing the resulting spectra)

---

# Approach

1. Find coefficients of real spectra (by optimisation/bruteforce)
2. Use resulting labels to train DL model
3. Use DL model to obtain coefficients quickly