# HSI Brain Segmentation

A comprehensive deep learning project for brain segmentation using hyperspectral imaging (HSI) data, featuring domain adaptation techniques and multiple neural network architectures.

## Project Overview

This project implements advanced deep learning techniques for brain segmentation from hyperspectral imaging data. It includes:

- **Hyperspectral Image Processing**: Working with 826-channel hyperspectral brain images
- **Domain Adaptation**: Feature Adaptation and Domain Adversarial (FADA) methods for cross-domain segmentation
- **Multiple Architectures**: Support for U-Net, U-Net++, LinkNet, FPN, PSPNet, DeepLabV3+, and more
- **Dimensionality Reduction**: Autoencoders, Gaussian channel reduction, and convolutional reducers
- **Ensemble Methods**: Model averaging and majority voting techniques
- **Extensive Evaluation**: Comprehensive metrics including Dice score, IoU, precision, recall

## Project Structure

```
├── 📁 src/                           # Source code
│   ├── 📁 dataset/                   # Dataset handling
│   │   └── dataset.py                # Dataset classes and data loaders
│   ├── 📁 model/                     # Neural network models
│   │   ├── 📁 FADA/                  # Feature Adaptation Domain Adversarial methods
│   │   │   ├── classifier.py         # Classification head
│   │   │   ├── discriminator.py      # Domain discriminator
│   │   │   ├── feature_extractor.py  # Feature extraction modules
│   │   │   └── segmentation_model.py # FADA segmentation models
│   │   ├── 📁 HSI_models/            # Hyperspectral-specific models
│   │   │   ├── HSI_Net.py            # HSI neural network architectures
│   │   │   └── ensemble_model.py     # Ensemble learning methods
│   │   └── 📁 dimensionality_reduction/ # Channel reduction techniques
│   │       ├── autoencoder.py        # Autoencoder-based reduction
│   │       ├── gaussian.py           # Gaussian channel reduction
│   │       ├── conv_reducer.py       # Convolutional reducers
│   │       ├── cycle_GAN.py          # CycleGAN for domain adaptation
│   │       └── window_reducer.py     # Wavelength windowing
│   ├── 📁 training/                  # Training pipelines
│   │   ├── domain_adaptation_training.py  # Domain adaptation training
│   │   └── supervised_domain_adaptation.py # Supervised domain adaptation
│   └── 📁 util/                      # Utility functions
│       ├── constants.py              # Project constants
│       ├── segmentation_util.py      # Segmentation utilities
│       └── visualize_data.py         # Data visualization tools
├── 📁 scripts/                       # Training and sweep scripts
│   ├── FIVES_sweep.py                # FIVES dataset hyperparameter sweeps
│   ├── sweep_autoencoder.py          # Autoencoder hyperparameter sweeps
│   ├── train_sweep_FADA_supervised.py # Supervised FADA sweeps
│   └── train_sweep_FADA_unsupervised.py # Unsupervised FADA sweeps
├── � notebooks/                     # Jupyter notebooks for analysis
│   ├── brain-segmentation.ipynb     # Brain segmentation experiments
│   ├── fundus-segmentation.ipynb    # Fundus segmentation (baseline)
│   ├── domain-adaptation.ipynb      # Domain adaptation experiments
│   ├── dataset-analysis.ipynb       # Dataset analysis and visualization
│   └── search-channels.ipynb        # Optimal channel selection
├── 📁 config/                        # Configuration files for experiments
├── 📁 metrics/                       # Custom evaluation metrics
├── 📁 data/                          # Dataset storage
├── 📁 models/                        # Trained model storage
├── 📋 requirements.txt               # Python dependencies
├── 🔧 .env.example                   # Environment variables template
└── 🔧 .env                           # Environment variables (create from .env.example)
```

## Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TimMachTUM/hsi-brain-segmentation.git
   cd hsi-brain-segmentation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env file to set your data paths
   nano .env  # or use your preferred editor
   ```

4. **Configure data directories**
   
   Edit the `.env` file to set the following paths:
   ```bash
   # FIVES dataset paths
   FIVES_DIR=/path/to/your/FIVES/dataset
   FIVES_RANDOM_CROPS_DIR=/path/to/your/FIVES_RANDOM_CROPS/dataset
   
   # HELICoiD HSI Brain Database paths
   HELICOID_DIR=/path/to/your/HELICoiD/HSI_Human_Brain_Database_IEEE_Access
   HELICOID_WITH_LABELS_DIR=/path/to/your/HELICoiD/with_labels
   
   # Model storage path
   MODELS_DIR=/path/to/your/trained/models
   
   # Ring labels for post-processing
   RING_LABELS_DIR=/path/to/your/ring/labels
   ```

### Environment Setup Guide

The project uses environment variables to manage data paths and configuration. Follow these steps to set up your environment:

1. **Copy the environment template**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file with your specific paths**
   Open the `.env` file and update the following variables:

   ```bash
   # FIVES Dataset Paths
   # Download from: https://figshare.com/articles/dataset/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169
   FIVES_DIR=/absolute/path/to/FIVES/dataset
   FIVES_RANDOM_CROPS_DIR=/absolute/path/to/FIVES_RANDOM_CROPS/dataset

   # HELICoiD HSI Brain Database Paths
   # Main hyperspectral brain database
   HELICOID_DIR=/absolute/path/to/HELICoiD/HSI_Human_Brain_Database_IEEE_Access
   # Version with ground truth labels
   HELICOID_WITH_LABELS_DIR=/absolute/path/to/HELICoiD/with_labels

   # Model Storage
   # Directory where trained models will be saved
   MODELS_DIR=/absolute/path/to/trained/models

   # Ring Labels (optional)
   # For post-processing ring artifact removal
   RING_LABELS_DIR=/absolute/path/to/ring/labels
   ```

3. **Example directory structure**
   
   Your data directories should be organized as follows:
   ```
   FIVES/
   ├── train/
   │   ├── Original/
   │   └── GroundTruth/
   ├── validation/
   │   ├── Original/
   │   └── GroundTruth/
   └── test/
       ├── Original/
       └── GroundTruth/

   HELICoiD/HSI_Human_Brain_Database_IEEE_Access/
   ├── 004-02/
   │   ├── raw.hdr
   │   ├── gtMap.hdr
   │   ├── image.jpg
   │   ├── darkReference.hdr
   │   └── whiteReference.hdr
   ├── 012-02/
   │   └── ...
   └── ...


   ```

### Basic Usage

1. **Train a basic segmentation model**
   ```python
   from src.dataset.dataset import build_hsi_dataloader
   from src.util.segmentation_util import build_segmentation_model, model_pipeline
   
   # Load data
   trainloader, validationloader, testloader = build_hsi_dataloader(
       batch_size=8, window=(500, 600)
   )
   
   # Build model
   model = build_segmentation_model(
       encoder="timm-regnetx_320",
       architecture="Unet",
       in_channels=1
   )
   
   # Train
   model_pipeline(model, trainloader, validationloader, testloader)
   ```

2. **Run domain adaptation**
   ```python
   from src.model.FADA.train_FADA_supervised import model_pipeline
   
   # Configure domain adaptation training
   config = {
       "architecture": "Linknet",
       "encoder": "timm-regnetx_320",
       "learning_rate_fea": 0.001,
       "learning_rate_cls": 0.001,
       "learning_rate_dis": 0.001
   }
   
   model_pipeline(trainloader_source, validationloader_source, 
                  testloader_source, trainloader_target, testloader_target, 
                  config, project="domain-adaptation")
   ```


## Key Features

### 1. Hyperspectral Image Processing
- **826-channel processing**: Full spectral range from 400-1000nm
- **Wavelength windowing**: Flexible spectral band selection
- **RGB simulation**: Convert HSI to RGB using optimal channel selection
- **Normalization**: Dark/white reference correction

### 2. Domain Adaptation
- **FADA (Feature Adaptation Domain Adversarial)**: State-of-the-art domain adaptation
- **Supervised/Unsupervised**: Both labeled and unlabeled target domain support
- **Cross-domain validation**: Evaluate on different imaging conditions

### 3. Neural Network Architectures
- **U-Net variants**: U-Net, U-Net++, MA-Net
- **Modern architectures**: LinkNet, FPN, PSPNet, DeepLabV3+, PAN
- **Encoders**: ResNet, RegNet, SENet, ResNeXt, and more
- **Custom layers**: Hyperspectral-specific processing layers

### 4. Dimensionality Reduction
- **Autoencoder-based**: Learnable channel reduction
- **Gaussian reduction**: Statistical channel combination
- **Convolutional reducers**: 1x1 convolutions for channel reduction
- **CycleGAN**: Generative domain adaptation

### 5. Advanced Training Features
- **Weights & Biases integration**: Automatic experiment tracking
- **Hyperparameter sweeps**: Automated hyperparameter optimization
- **Custom loss functions**: Dice, BCE, Focal loss, clDice
- **Data augmentation**: Rotation, flipping, affine transformations

## Datasets

### Primary Dataset: HELICoiD HSI Brain Database
- **826 spectral channels** (400-1000nm)
- **High spatial resolution** brain tissue images
- **Ground truth segmentation** masks
- **Multiple subjects** and imaging conditions

### Auxiliary Dataset: FIVES
- **Fundus images** for baseline comparison
- **Retinal vessel segmentation** tasks
- **Domain adaptation experiments**

## Configuration

The project uses YAML configuration files for different experiments:

```yaml
# Example config file
name: FADA-RGB-Sweep
method: bayes
parameters:
  architecture:
    value: Linknet
  encoder:
    value: timm-regnetx_320
  in_channels:
    value: 3
  learning_rate_fea:
    min: 0.001
    max: 0.01
```

## Evaluation Metrics

- **Dice Score**: Harmonic mean of precision and recall
- **Precision/Recall**: Pixel-level accuracy metrics
- **clDice**: Connectivity-aware Dice score
- **F1 Score**: Balanced accuracy measure

## Experiments

### Key Notebook Examples

1. **`notebooks/brain-segmentation.ipynb`**: Main HSI brain segmentation experiments
2. **`notebooks/domain-adaptation.ipynb`**: Cross-domain adaptation experiments
3. **`notebooks/search-channels.ipynb`**: Optimal spectral channel selection
4. **`notebooks/dataset-analysis.ipynb`**: Data exploration and visualization

### Hyperparameter Sweeps

The project includes automated hyperparameter optimization:

```bash
# FIVES dataset sweep
python scripts/FIVES_sweep.py

# Autoencoder sweep
python scripts/sweep_autoencoder.py

# Domain adaptation sweeps
python scripts/train_sweep_FADA_supervised.py
python scripts/train_sweep_FADA_unsupervised.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact

In case of questions, please contact Tim Mach or Ivan Ezhov
Technical University of Munich (TUM)
