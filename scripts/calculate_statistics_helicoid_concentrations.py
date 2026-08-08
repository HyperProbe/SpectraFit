import os
import sys

# Add project root to path
sys.path.append("..")
from pathlib import Path

from tqdm import tqdm
from src.constants import HELICOID_VAL_SET, HELICOID_TEST_SET
from src.dataset.concentrations_dataset import ConcentrationsDataset

import numpy as np

from src.dataset.dataset_utils import split_dataset_with_ids
from src.wavelength_selection.enums import SampleType

reduced_wl_dir = (
    "../results/helicoid_spectral_unmixing/coarseness_1_left_530_right_750_reduced"
)
gt_dir = "../results/helicoid_spectral_unmixing/coarseness_1_left_530_right_750"

dataset = ConcentrationsDataset(
    reduced_wl_dir=reduced_wl_dir,
    gt_dir=gt_dir,
    sample_type=SampleType.HELICOID,
    center_crop_size=(224, 224),  # Example crop size, adjust as needed
)
print(f"Successfully created dataset with {len(dataset)} samples")

trainset, _, _ = split_dataset_with_ids(
    dataset, val_ids=HELICOID_VAL_SET, test_ids=HELICOID_TEST_SET
)

# Define output directories for saving statistics
# Create the main statistics directory with subdirectories for gt and reduced_wl
stats_base_dir = "../results/statistics/helicoid"
gt_stats_dir = os.path.join(stats_base_dir, "gt")
reduced_wl_stats_dir = os.path.join(stats_base_dir, "reduced_wl")

# Create directories if they don't exist
Path(gt_stats_dir).mkdir(parents=True, exist_ok=True)
Path(reduced_wl_stats_dir).mkdir(parents=True, exist_ok=True)

print(f"Calculating channel-wise statistics for {len(trainset)} samples...")

# Initialize lists to collect all concentration values per channel
reduced_wl_channels = []
gt_channels = []

# Get the number of channels from the first sample
sample = trainset[0]
n_channels = sample["reduced_wl"]["coef_list"].shape[2]
print(f"Number of channels (molecules): {n_channels}")

# Initialize lists for each channel
for i in range(n_channels):
    reduced_wl_channels.append([])
    gt_channels.append([])

a_reduced = []
b_reduced = []
a_gt = []
b_gt = []
# Collect all concentration values across all samples
for i, sample in enumerate(tqdm(trainset)):
    try:
        reduced_conc = sample["reduced_wl"]["coef_list"]
        scatter_params_reduced = sample["reduced_wl"]["scatter_params"]
        gt_conc = sample["gt"]["coef_list"]
        scatter_params_gt = sample["gt"]["scatter_params"]

        # For each channel, flatten the spatial dimensions and collect values
        for channel_idx in range(n_channels):
            # Flatten spatial dimensions (H, W) to get all pixel values for this channel
            reduced_channel_values = reduced_conc[:, :, channel_idx].flatten()
            gt_channel_values = gt_conc[:, :, channel_idx].flatten()

            reduced_wl_channels[channel_idx].extend(reduced_channel_values)
            gt_channels[channel_idx].extend(gt_channel_values)

        a_reduced.append(scatter_params_reduced[:, 0].flatten())
        b_reduced.append(scatter_params_reduced[:, 1].flatten())
        a_gt.append(scatter_params_gt[:, 0].flatten())
        b_gt.append(scatter_params_gt[:, 1].flatten())

    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        continue

print("Calculating statistics and saving to files...")

# Calculate mean and std for each channel and save to files
molecule_names = [
    "HbO2",
    "Hb",
    "oxCCO",
    "redCCO",
    "Cyt-c_oxy",
    "Cyt-c_red",
    "Cyt-b_oxy",
    "Cyt-b_red",
    "Water",
    "Fat",
]

for channel_idx in range(n_channels):
    # Convert to numpy arrays for efficient computation
    reduced_values = np.array(reduced_wl_channels[channel_idx])
    gt_values = np.array(gt_channels[channel_idx])

    # Calculate statistics
    reduced_mean = np.mean(reduced_values)
    reduced_std = np.std(reduced_values)
    gt_mean = np.mean(gt_values)
    gt_std = np.std(gt_values)

    # Save reduced wavelength statistics
    np.save(os.path.join(reduced_wl_stats_dir, f"mean_{channel_idx}.npy"), reduced_mean)
    np.save(os.path.join(reduced_wl_stats_dir, f"std_{channel_idx}.npy"), reduced_std)

    # Save ground truth statistics
    np.save(os.path.join(gt_stats_dir, f"mean_{channel_idx}.npy"), gt_mean)
    np.save(os.path.join(gt_stats_dir, f"std_{channel_idx}.npy"), gt_std)

    # Print summary
    molecule_name = (
        molecule_names[channel_idx]
        if channel_idx < len(molecule_names)
        else f"Channel_{channel_idx}"
    )
    print(f"Channel {channel_idx} ({molecule_name}):")
    print(f"  Reduced WL - Mean: {reduced_mean:.6f}, Std: {reduced_std:.6f}")
    print(f"  Ground Truth - Mean: {gt_mean:.6f}, Std: {gt_std:.6f}")
    print(f"  Sample count: {len(reduced_values):,} pixels")

# Save scatter parameters statistics
a_reduced_mean = np.mean(a_reduced)
a_reduced_std = np.std(a_reduced)
b_reduced_mean = np.mean(b_reduced)
b_reduced_std = np.std(b_reduced)
a_gt_mean = np.mean(a_gt)
a_gt_std = np.std(a_gt)
b_gt_mean = np.mean(b_gt)
b_gt_std = np.std(b_gt)

np.save(os.path.join(reduced_wl_stats_dir, "scatter_mean_0.npy"), a_reduced_mean)
np.save(os.path.join(reduced_wl_stats_dir, "scatter_std_0.npy"), a_reduced_std)
np.save(os.path.join(reduced_wl_stats_dir, "scatter_mean_1.npy"), b_reduced_mean)
np.save(os.path.join(reduced_wl_stats_dir, "scatter_std_1.npy"), b_reduced_std)

np.save(os.path.join(gt_stats_dir, "scatter_mean_0.npy"), a_gt_mean)
np.save(os.path.join(gt_stats_dir, "scatter_std_0.npy"), a_gt_std)
np.save(os.path.join(gt_stats_dir, "scatter_mean_1.npy"), b_gt_mean)
np.save(os.path.join(gt_stats_dir, "scatter_std_1.npy"), b_gt_std)

print(f"\nStatistics saved to:")
print(f"  Base directory: {stats_base_dir}")
print(f"  Reduced WL: {reduced_wl_stats_dir}")
print(f"  Ground Truth: {gt_stats_dir}")

# Verify files were created
print(f"\nFiles created in {reduced_wl_stats_dir}:")
for file in sorted(os.listdir(reduced_wl_stats_dir)):
    print(f"  {file}")

print(f"\nFiles created in {gt_stats_dir}:")
for file in sorted(os.listdir(gt_stats_dir)):
    print(f"  {file}")

print(f"\nNormalization directory structure:")
print(f"{stats_base_dir}/")
print(f"├── gt/")
print(f"│   ├── mean_0.npy, std_0.npy")
print(f"│   ├── mean_1.npy, std_1.npy")
print(f"│   └── ...")
print(f"└── reduced_wl/")
print(f"    ├── mean_0.npy, std_0.npy")
print(f"    ├── mean_1.npy, std_1.npy")
print(f"    └── ...")
