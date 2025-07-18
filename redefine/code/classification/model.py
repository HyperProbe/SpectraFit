import torch
import lightning.pytorch as pl
from torch import optim, nn, Tensor
from torchmetrics import Accuracy 
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy

class ClassificationModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, loss_weight, config):
        super().__init__()
        self.save_hyperparameters(logger=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_weight = Tensor(loss_weight).to(device)
        self.accuracy = Accuracy(task="multiclass", num_classes=output_dim, top_k=1)
        self.multiclass_accuracy = MulticlassAccuracy(num_classes=output_dim, average=None)
        self.f1_score = MulticlassF1Score(num_classes=output_dim, average="macro")

        self.weight_decay = config["weight_decay"]
        self.lr = config["lr"]
        self.num_layers = config["num_layers"]
        self.hidden_dim = config["hidden_dim"]
        self.last_layer_dim = config["last_layer_dim"]

        input_layer = nn.Linear(input_dim, self.hidden_dim)
        hidden_layers = []
        for _ in range(self.num_layers):
            hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            hidden_layers.append(nn.BatchNorm1d(self.hidden_dim))
            hidden_layers.append(nn.ReLU())
        last_layer = nn.Linear(self.hidden_dim, self.last_layer_dim)
        output_layer = nn.Linear(self.last_layer_dim, output_dim)
        self.layers = nn.Sequential(nn.BatchNorm1d(input_dim), input_layer, nn.BatchNorm1d(self.hidden_dim), nn.ReLU(), *hidden_layers, last_layer, nn.BatchNorm1d(self.last_layer_dim), nn.ReLU(), output_layer)

        self.epoch_logits = []
        self.epoch_labels = []
        self.best_epoch = 0

    def forward(self, x):
        return self.layers(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y, weight=self.loss_weight)
        accuracy = self.accuracy(logits, y)
        tumor_accuracy = self.multiclass_accuracy(logits, y)[1]
        self.log("train/train_loss", loss)
        self.log("train/train_accuracy", accuracy)
        self.log("train/tumor_accuracy", tumor_accuracy)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.eval()
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y, weight=self.loss_weight)
        accuracy = self.accuracy(logits, y)
        tumor_accuracy = self.multiclass_accuracy(logits, y)[1]
        self.log("val/val_loss", loss, on_epoch=True, on_step=False)
        self.log("val/val_accuracy", accuracy, on_epoch=True, on_step=False)
        self.log("val/tumor_accuracy", tumor_accuracy, on_epoch=True, on_step=False)

        self.epoch_logits.append(logits)
        self.epoch_labels.append(y)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y, weight=self.loss_weight)
        accuracy = self.accuracy(logits, y)
        self.log("test/test_loss", loss, on_epoch=True, on_step=False)
        self.log("test/test_accuracy", accuracy, on_epoch=True, on_step=False)

        self.epoch_logits.append(logits)
        self.epoch_labels.append(y)

    def on_validation_epoch_end(self):
        # calculate f1 score
        epoch_logits = torch.cat(self.epoch_logits, dim=0)
        epoch_labels = torch.cat(self.epoch_labels, dim=0)
        f1_score = self.f1_score(epoch_logits, epoch_labels)
        self.log("val/f1_score", f1_score, sync_dist=True)
        self.epoch_logits.clear()
        self.epoch_labels.clear()

    def on_test_epoch_end(self):
        # calculate f1 score
        epoch_logits = torch.cat(self.epoch_logits, dim=0)
        epoch_labels = torch.cat(self.epoch_labels, dim=0)
        f1_score = self.f1_score(epoch_logits, epoch_labels)
        self.log("test/f1_score", f1_score, sync_dist=True)
        self.epoch_logits.clear()
        self.epoch_labels.clear()

    def on_train_epoch_end(self):
        i = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                i += 1
                self.logger.experiment.add_histogram(f"layer_{i}/weight", layer.weight, self.current_epoch)
                self.logger.experiment.add_histogram(f"layer_{i}/bias", layer.bias, self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)
        return optimizer