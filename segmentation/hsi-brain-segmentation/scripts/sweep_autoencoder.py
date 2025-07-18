import wandb
import torch
import sys
sys.path.append("..")
from src.training.train_autoencoder import train_sweep  
import yaml

log_file = open("./out/hyperparameter_logs.txt", "w")
sys.stdout = log_file
wandb.login()

with open("./config/gaussian_autoencoder.yaml", "r") as file:
    sweep_config = yaml.safe_load(file)

sweep_config["parameters"]["device"] = {
    "value": "cuda:7" if torch.cuda.is_available() else "cpu"
}
sweep_id = wandb.sweep(sweep_config, project="hsi-autoencoder")
wandb.agent(sweep_id, function=train_sweep, count=20)

sys.stdout = sys.__stdout__
log_file.close()