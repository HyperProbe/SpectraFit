# Autoencoder Model Pipeline

This module provides a complete pipeline for training and validating convolutional autoencoders on concentration maps from spectral unmixing results.

## Features

- **Configurable Architecture**: Flexible encoder/decoder configuration
- **Multiple Optimizers**: Support for Adam, SGD, RMSprop
- **Learning Rate Scheduling**: StepLR, ReduceLROnPlateau
- **Early Stopping**: Prevent overfitting with configurable patience
- **Checkpointing**: Save and resume training progress
- **Evaluation Metrics**: MSE, MAE, RMSE
- **Tensorboard Logging**: Monitor training progress
- **Data Splitting**: Patient-wise train/validation/test splits

## Quick Start

### 1. Basic Usage

```python
from src.models import ModelPipeline, create_default_config

# Create configuration
config = create_default_config(
    reduced_wl_dir="/path/to/reduced_wavelength_results",
    gt_dir="/path/to/ground_truth_results",
    experiment_name="my_autoencoder",
    output_dir="./results"
)

# Train model
pipeline = ModelPipeline(config)
pipeline.train_and_evaluate()

# Evaluate
test_metrics = pipeline.evaluate(with_wandb=False)
print(f"Test MSE: {test_metrics['mse']:.6f}")
```

### 2. Using Configuration Files

```python
# Create custom config
config = ModelConfig(
    in_channels=10,
    encoder_channels=[64, 128, 256],
    batch_size=16,
    learning_rate=1e-3,
    num_epochs=100,
    # ... other parameters
)

# Save config
config.save_json("my_config.json")

# Load and use config
pipeline = ModelPipeline("my_config.json")
pipeline.train()
```
## Configuration Options

### Model Architecture
- `in_channels`: Number of input channels (molecules)
- `encoder_channels`: List of encoder layer output channels
- `decoder_channels`: List of decoder layer output channels (auto-computed if None)
- `kernel_size`, `stride`, `padding`: Convolution parameters
- `activation`: Activation function (ReLU, LeakyReLU, etc.)
- `final_activation`: Final layer activation (optional)

### Training Parameters
- `batch_size`: Training batch size
- `learning_rate`: Initial learning rate
- `num_epochs`: Maximum number of epochs
- `optimizer`: Optimizer type (Adam, SGD, RMSprop)
- `scheduler`: Learning rate scheduler (StepLR, ReduceLROnPlateau, None)
- `loss_function`: Loss function (MSE, L1, Huber)

### Data Configuration
- `reduced_wl_dir`: Path to reduced wavelength concentration maps
- `gt_dir`: Path to ground truth concentration maps
- `split_ratios`: Train/validation/test split ratios
- `random_seed`: Random seed for reproducibility

### Training Options
- `early_stopping_patience`: Early stopping patience epochs
- `save_best_model`: Save best validation model
- `checkpoint_interval`: Checkpoint saving interval

## Data Requirements

The pipeline expects concentration map data in the following structure:

```
results_directory/
├── sample1_directory/
│   ├── coef_list.npy          # Concentration coefficients (H, W, num_molecules)
│   ├── scatter_params.npy     # Scattering parameters (H, W, 2)
│   └── errors_scatter.npy     # Optimization errors (H, W) [optional]
├── sample2_directory/
│   └── ...
└── ...
```

Each `.npy` file should contain:
- `coef_list.npy`: Shape (H, W, C) where C is number of molecules
- `scatter_params.npy`: Shape (H, W, 2) for scattering parameters
- `errors_scatter.npy`: Shape (H, W) for optimization errors (optional)

## Examples

### Example 1: Simple Training

```python
from src.models import ModelPipeline, ModelConfig

config = ModelConfig(
    reduced_wl_dir="../results/reduced_wl_experiment",
    gt_dir="../results/gt_experiment", 
    experiment_name="simple_autoencoder",
    batch_size=8,
    num_epochs=50,
    learning_rate=1e-3
)

pipeline = ModelPipeline(config)
pipeline.train()
```

### Example 2: Custom Architecture

```python
config = ModelConfig(
    # Larger model
    encoder_channels=[64, 128, 256, 512],
    kernel_size=5,
    activation="LeakyReLU",
    
    # Advanced training
    optimizer="Adam",
    scheduler="ReduceLROnPlateau",
    scheduler_params={"patience": 10, "factor": 0.5},
    loss_function="L1",
    
    # Other settings...
)
```

### Example 3: Resume Training

```python
# Resume from checkpoint
pipeline = ModelPipeline(config)
pipeline.train(resume_from_checkpoint="./results/experiment/checkpoints/best_model.pth")
```

### Example 4: Load Trained Model

```python
# Load and evaluate trained model
config = ModelConfig.from_json("./results/experiment/config.json")
pipeline = ModelPipeline(config)
pipeline.setup_all()
pipeline.load_checkpoint("./results/experiment/checkpoints/best_model.pth")

# Evaluate
metrics = pipeline.evaluate()
print(f"Loaded model performance: {metrics}")
```

## Monitoring Training

### Training Curves
Access training history programmatically:

```python
pipeline.train()

# Plot training curves
import matplotlib.pyplot as plt
plt.plot(pipeline.train_losses, label='Train')
plt.plot(pipeline.val_losses, label='Validation') 
plt.legend()
plt.show()
```

## File Structure

After training, the following files are created:

```
results/experiment_name/
├── config.json                    # Saved configuration
├── checkpoints/
│   ├── best_model.pth             # Best validation model
│   ├── checkpoint_epoch_10.pth    # Regular checkpoints
│   └── ...
└── logs/                          # Tensorboard logs
    └── events.out.tfevents.*
```

## Notes

- The pipeline automatically handles device selection (CUDA/CPU)
- Patient-wise data splitting prevents data leakage
- Early stopping prevents overfitting
- Checkpointing allows resuming interrupted training
- All configurations are saved for reproducibility

## Jupyter Notebook Demo

See `notebooks/autoencoder_training_demo.ipynb` for an interactive demonstration of the complete pipeline.
