# SpectraFit

Code used for spectral unmixing and molecular mapping of HELICOID hyperspectral images.

This branch contains the implementation used for the paper experiments, including different approaches for modeling the wavelength-dependent optical pathlength.

## Overview

The pipeline estimates changes in molecular concentrations and tissue scattering parameters from hyperspectral measurements using the differential modified Beer-Lambert law (dMBLL).

For HELICOID samples, a blood pixel is selected as the reference spectrum and the change in attenuation (`delta_A`) is computed relative to this reference. Molecular concentration changes and scattering parameters are then estimated pixelwise using least-squares optimization.

The main quantities estimated include:

* HbO2
* HHb
* oxidized CCO
* reduced CCO
* water
* fat
* scattering coefficient and scattering power

## Pathlength models

Several pathlength models are implemented in `src/scattering_model/scatter_optim.py`.

Available `--pathlength-mode` options are:

* `ones` — unitary pathlength (`Lp = 1`)
* `pathlength_from_wavelength` — wavelength-dependent pathlength interpolated from a predefined gray-matter pathlength spectrum
* `gray_matter_delta_p1` — wavelength-dependent pathlength computed for gray matter using the delta-P1 approximation
* `gray_matter_jacques` — wavelength-dependent pathlength computed using the Jacques model
* `pixelwise_delta_p1_from_unitary` — pixelwise delta-P1 pathlength 
* `pixelwise_jacques_from_unitary` — pixelwise Jacques pathlength
  
The pathlength-related model files are stored under:

```text
data/pathlengths/
├── gray_matter_pl_interpolated_spline.txt
├── carp.pickle
├── jacques.pickle
└── m_parameters.pickle
```

## Main script

The main script for generating molecular maps is:

```text
scripts/plot_helicoid_molecular_maps.py
```

It:

1. loads the selected HELICOID samples,
2. selects a random blood pixel as the reference,
3. computes `delta_A`,
4. estimates reference scattering parameters,
5. performs pixelwise spectral unmixing,
6. caches the estimated coefficients and scattering parameters, and
7. generates molecular and scattering maps.

By default, the spectral unmixing is performed between 530 and 750 nm.

## Example

For example, the gray-matter Jacques pathlength model can be run with:

```bash
python scripts/plot_helicoid_molecular_maps.py \
    --pathlength-mode gray_matter_jacques \
    --output-dir plots/tail_stats \
    --results-dir results/tmp_pixelwise_pl_debug \
    --pathlength-debug-dir data/pathlengths
```

Paths can be changed depending on the local setup.

Other pathlength models can be tested by changing `--pathlength-mode`.

For example:

```bash
--pathlength-mode ones
--pathlength-mode pathlength_from_wavelength
--pathlength-mode gray_matter_delta_p1
--pathlength-mode gray_matter_jacques
--pathlength-mode pixelwise_delta_p1_from_unitary
--pathlength-mode pixelwise_jacques_from_unitary
```

The pixelwise models require previously computed unitary-pathlength results.

## Output

Computed results are stored in the specified `--results-dir`. The pipeline caches the molecular coefficients and scattering parameters for each sample as NumPy files so that they can be reused for further analysis and visualization.

Generated figures are written to the specified `--output-dir`.

The repository also contains plotting utilities for comparing molecular maps obtained with different pathlength models.

## Data

The HELICOID dataset is not included in this repository.

The dataset location can be specified using:

```bash
--data-root /path/to/helicoid/data
```

## Installation

The Python dependencies are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```
