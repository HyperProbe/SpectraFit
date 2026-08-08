# HSI-Biopsy Dataset Usage Guide

This guide explains how to work with the HSI-Biopsy dataset, including loading data, accessing samples, and visualizing hyperspectral images.

## Basic Dataset Usage

The `Biopsy2Dataset` class provides a convenient interface for working with hyperspectral imaging (HSI) data from biopsy samples. Each sample in the dataset represents a single HSI cube with its associated metadata.

### Initializing the Dataset

```python
from src.dataset.biopsy2_dataset import Biopsy2Dataset

# Initialize dataset (reads paths from .env file)
ds = Biopsy2Dataset()

# You can also specify custom paths
# ds = Biopsy2Dataset(hsi_data_dir='/path/to/hsi_files', metadata_csv_path='/path/to/metadata.csv')
```

### Dataset Information

```python
# Get number of HSI cubes in the dataset
num_samples = len(ds)
print(f"Total HSI cubes in dataset: {num_samples}")
```

### Loading Samples

There are multiple ways to access samples in the dataset:

#### 1. By Index

```python
# Load a sample by its index
sample = ds[0]  # First sample in the dataset

# Access sample contents
print(sample['id'])      # Unique ID in format "patient_number_fov"
print(sample['patient_id'])       # Original patient ID (e.g., "S1.2") 
print(sample['fov'])              # FOV number
print(sample['hsi_cube'].shape)   # HSI cube data shape (typically [bands, height, width])
print(sample['metadata'])         # Associated metadata dictionary
```

#### 2. By ID

```python
# Get by ID (e.g., "1.2_3" for patient S1.2, FOV 3)
sample = ds.get_sample_by_id("1.2_3")
```

#### 3. By Patient ID

```python
# Get all FOVs for a specific patient
patient_samples = ds.get_samples_by_patient_id("S1.2")

# Iterate through all FOVs for this patient
for sample in patient_samples:
    print(f"FOV {sample['fov']}, cube shape: {sample['hsi_cube'].shape}")
```

#### 4. By Patient ID and FOV

```python
# Get a specific patient and FOV combination
specific_sample = ds.get_sample_by_patient_and_fov("S1.2", "3")
```

### Using Patient ID Constants

The `ALL_PATIENT_IDS` class provides autocomplete-friendly access to patient IDs:

```python
from src.constants import ALL_PATIENT_IDS

# Access a specific patient ID (with IDE autocomplete support)
patient_id = ALL_PATIENT_IDS.S1_2  # Returns "S1.2"

# Get samples for this patient
samples = ds.get_samples_by_patient_id(patient_id)
```

You can also filter patients by tumor type:

```python
# Get all glioma patients
glioma_patients = ALL_PATIENT_IDS.get_by_type("Glioma")

# Get all meningioma patients
meningioma_patients = ALL_PATIENT_IDS.get_by_type("Mening")
```

## Visualization

The project includes utilities for visualizing HSI data.

### Creating RGB Images

```python
from src.visualize_data import create_rgb

# Create an RGB representation of an HSI cube
rgb_image = create_rgb(sample['hsi_cube'], r_band=650, g_band=550, b_band=450)

# Display the image
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(rgb_image)
plt.title(f"Patient {sample['patient_id']}, FOV {sample['fov']}")
plt.axis('off')
plt.show()
```

### Interactive Spectral Analysis

```python
from src.visualize_data import show_interactive_image_with_spectrum

# Create an interactive visualization (works in Jupyter notebooks)
%matplotlib widget  # For Jupyter notebooks
show_interactive_image_with_spectrum(sample['hsi_cube'])
```

### Displaying Patient Data with Metadata

```python
from src.visualize_data import display_with_metadata

# Display all samples for a patient with their metadata
display_with_metadata(ds, "S1.2")

# Optionally save the figures to a directory
display_with_metadata(ds, "S1.2", save_to_path='data/output_images')
```

## Example Workflow

A complete example workflow might look like this:

```python
from src.dataset.biopsy2_dataset import Biopsy2Dataset
from src.constants import ALL_PATIENT_IDS
from src.visualize_data import display_with_metadata

# Initialize the dataset
ds = Biopsy2Dataset()

# Get all glioma patients
glioma_patients = ALL_PATIENT_IDS.get_by_type("Glioma")
print(f"Found {len(glioma_patients)} glioma patients")

# Select a patient to analyze
patient_id = glioma_patients[0]  # First glioma patient

# Display and optionally save all FOVs for this patient
display_with_metadata(ds, patient_id, save_to_path='data/output_images')
```

For more detailed examples, see the Jupyter notebooks in the `notebooks/` directory, particularly [example_analysis.ipynb](../notebooks/example_analysis.ipynb).
