import argparse
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from model import ClassificationModel
from dataloader import HelicoidDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, help="Training mode: baseline, baseline_reduced, heatmap or heatmap_only", choices=["baseline", "heatmap", "heatmap_nomc", "baseline_reduced", "heatmap_only"])
# add mandatory argument for log_dir
parser.add_argument("--log_dir", type=str, required=True, help="Directory to save logs")
parser.add_argument("--folds", nargs='+', type=str, required=True, help="Fold to use for training", choices=["fold1", "fold2", "fold3", "fold4", "fold5"])
parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension of the model")
parser.add_argument("--num_layers", type=int, required=True, help="Number of hidden layers")
parser.add_argument("--last_layer_dim", type=int, required=True, help="Dimension of the last hidden layer")
parser.add_argument("--lr", type=float, required=True, help="Learning rate")
parser.add_argument("--weight_decay", type=float, required=True, help="Weight decay")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
args = parser.parse_args()


def train(config, dm):
    logger = TensorBoardLogger(config["log_dir"], name="my_model")
    model = ClassificationModel(input_dim=dm.sample_size(), output_dim=dm.num_classes(), loss_weight=1/dm.class_distribution(), config=config)

    early_stop_callback = EarlyStopping(monitor="val/val_loss", mode="min", min_delta=0.0, patience=config["patience"], verbose=False)
    checkpoint_callback = ModelCheckpoint(monitor="val/val_loss", mode="min", save_top_k=1, dirpath=config["log_dir"], filename=f"{dm.get_fold()}")
    trainer = pl.Trainer(logger=logger, max_epochs=config["num_epochs"], devices=1, callbacks=[early_stop_callback, checkpoint_callback])

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
        "patience": config["patience"],
    }
    metrics = {"best_val_loss": best_val_loss}
    logger.log_hyperparams(params, metrics)

    return model


def main():

    # data to use as the model input
    if args.mode == "baseline":
        files = ["preprocessed.npy"]
    elif args.mode == "heatmap":
        files = ["preprocessed.npy", "osp_absolute.npy", "osp_rel_mc.npy", "osp_rel_lit.npy", "cem_absolute.npy", "cem_rel_lit.npy", "cem_rel_mc.npy"]
    elif args.mode == "heatmap_nomc":
        files = ["preprocessed.npy", "osp_absolute.npy", "osp_rel_lit.npy", "cem_absolute.npy", "cem_rel_lit.npy"]
    elif args.mode == "heatmap_only":
        files = ["osp_absolute.npy", "osp_rel_mc.npy", "osp_rel_lit.npy", "cem_absolute.npy", "cem_rel_lit.npy", "cem_rel_mc.npy"]

    # train model for each fold
    for fold in args.folds:
        dm = HelicoidDataModule(files=files, fold=fold)
        dm.setup("fit")

        config = {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "last_layer_dim": args.last_layer_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_epochs": 150,
            "log_dir": args.log_dir,
            "patience": 5,
            "batch_size": args.batch_size,
        }

        train(config, dm)

if __name__ == "__main__":
    main()

