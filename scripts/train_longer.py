import sys


# Add src to path
sys.path.append('..')

from src.models.pipeline.model_pipeline import ModelPipeline
from src.models.pipeline.config import ModelConfig

config = ModelConfig.from_json(
    "../results/models/sweep/helicoid-autoencoder/unet_12_wl_adversarial_image_norm/distinctive-sweep-29/config.json"
)
config.experiment_name = "unet_adv_12_wl_random_crops_image_norm_longer_training"
config.output_dir = "../results/models"
config.num_epochs = 100
config.device = "cuda:0"
config.device_ids = [0]
config.evaluate_full_resolution = True
config.early_stopping_patience = 100


# Create pipeline
pipeline = ModelPipeline(config)
pipeline.setup_all()
print(f"\nPipeline ready!")
print(f"Model parameters: {sum(p.numel() for p in pipeline.model.parameters()):,}")
print(f"Training batches: {len(pipeline.train_loader)}")
print(f"Validation batches: {len(pipeline.val_loader)}")
print(f"Test batches: {len(pipeline.test_loader)}")

pipeline.start_pipeline()