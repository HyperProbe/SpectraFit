import wandb
import torch
import sys
sys.path.append("..")
from src.util.segmentation_util import train_sweep

log_file = open("./out/hyperparameter_logs.txt", "w")
sys.stdout = log_file
wandb.login()

sweep_config = {
    "name": "ClDice-Alpha-Random-Sweep",
    "method": "random",
    "parameters": {
        "encoder": {"value": "timm-regnetx_320"},
        "learning_rate": {"value": 0.012948375271687074},
        "epochs": {"value": 15},
        "loss": {"value": "ClDice"},
        "optimizer": {"value": "Adam"},
        "batch_size": {"value": 8},
        "proportion_augmented_data": {"value": 0.1},
        "architecture": {
            "value": "Linknet",
        },
        "channels": {"value": 1},
        "width": {"value": 512},
        "height": {"value": 512},
        "iter": {"value": 10},
        "smooth": {"value": 1.0},
        "alpha": {"min": 0.1, "max": 0.9},
    },
    "metric": {"name": "dice_score", "goal": "maximize"},
}

sweep_config["parameters"]["device"] = {
    "value": "cuda:3" if torch.cuda.is_available() else "cpu"
}
sweep_id = wandb.sweep(sweep_config, project="fundus-segmentation")
wandb.agent(sweep_id, function=train_sweep, count=10)

sys.stdout = sys.__stdout__
log_file.close()
