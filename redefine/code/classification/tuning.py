import argparse
import numpy as np
import lightning.pytorch as pl

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping

from model import ClassificationModel
from dataloader import HelicoidDataModule

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, help="Training mode: baseline, baseline_reduced, heatmap or heatmap_only", choices=["baseline", "heatmap", "heatmap_nomc", "baseline_reduced", "heatmap_only"])
# add mandatory argument for log_dir
parser.add_argument("--log_dir", type=str, required=True, help="Directory to save logs")
args = parser.parse_args()


def train(config, dm):
    logger = TensorBoardLogger(config["log_dir"], name="my_model")
    model = ClassificationModel(input_dim=dm.sample_size(), output_dim=dm.num_classes(), loss_weight=1/dm.class_distribution(), config=config)

    early_stop_callback = EarlyStopping(monitor="val/val_loss", min_delta=0.0, patience=5, verbose=False, mode="min")
    trainer = pl.Trainer(logger=logger, max_epochs=config["num_epochs"], devices=1, callbacks=[early_stop_callback])

    trainer.fit(model, dm.train_dataloader(batch_size=config["batch_size"]), dm.val_dataloader())

    # log hyperparameters and metrics
    best_val_loss = early_stop_callback.best_score
    params = {
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "last_layer_dim": config["last_layer_dim"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "batch_size": config["batch_size"],
        # "num_epochs": model.current_epoch-5,
    }
    metrics = {"best_val_loss": best_val_loss}
    logger.log_hyperparams(params, metrics)

    return model

def main():

    # data to use as the model input
    if args.mode == "baseline":
        files = ["preprocessed.npy"]
    elif args.mode == "heatmap":
        files = ["preprocessed.npy", "osp_absolute.npy", "osp_rel_lit.npy", "osp_rel_mc.npy", "cem_absolute.npy", "cem_rel_lit.npy", "cem_rel_mc.npy"]
    elif args.mode == "heatmap_nomc":
        files = ["preprocessed.npy", "osp_absolute.npy", "osp_rel_lit.npy", "cem_absolute.npy", "cem_rel_lit.npy"]
    elif args.mode == "heatmap_only":
        files = ["osp_absolute.npy", "osp_rel_lit.npy", "osp_rel_mc.npy", "cem_absolute.npy", "cem_rel_lit.npy", "cem_rel_mc.npy"]


    dm = HelicoidDataModule(files=files, fold="fold3")
    dm.setup("fit")

    # train 50 randomly sampled configurations
    np.random.seed(0)
    for i in range(50):
        hidden_dim = np.random.randint(4, 64)
        num_layers = np.random.randint(0, 4)
        last_layer_dim = np.random.randint(4, hidden_dim+1)
        exp = np.random.uniform(-5, -1)
        weight_decay = 10**exp
        exp = np.random.uniform(-6, -4)
        lr = 10**exp
        batch_size = 2**np.random.randint(5, 7)

        config = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "last_layer_dim": last_layer_dim,
            "lr": lr,
            "weight_decay": weight_decay,
            "num_epochs": 100,
            "log_dir": args.log_dir,
            "batch_size": batch_size
        }

        print(config)
        train(config, dm)


if __name__ == "__main__":
    main()