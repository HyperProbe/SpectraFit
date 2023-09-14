import os
import time

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data_processing.datasets import PigletDataset, SpectraDataset
from neuralnet.model import SpectraMLP
import config


def train_piglet(n_layers=4, layer_width=1024):
    """
    created to train on datasets in dataset/piglet_diffs
    results are saved in the results folder, for each n_layers/layer_width setup seperately
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lr = config.lr

    name = str(n_layers) + "_" + str(layer_width)
    os.mkdir("results/{}".format(name))
    save_path = 'results/{}/best_model.pth'.format(name)

    model = SpectraMLP(config.molecule_count, n_layers=n_layers, layer_width=layer_width)
    model.to(device)
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    loss = 0.0

    # train_set = PigletDataset(path_to_data, version="_train")
    # val_set = PigletDataset(path_to_data, version="_test")
    # train_set = SpectraDataset(config.dataset_path+"synthetic/")
    train_set = PigletDataset(config.dataset_path+"piglet_diffs/pig2/")
    val_set = PigletDataset(config.dataset_path+"piglet_diffs/pig3/")

    train_dl = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=32, shuffle=False)

    best_loss, best_age = 100, 0
    ground_truth, predictions = None, None
    start = time.time()
    for epoch in range(config.epochs):
        print('Epoch {}/{}'.format(epoch + 1, config.epochs))
        print('-' * 10)
        model.train()
        # Iterate through training data loader
        for i, (inputs, targets) in enumerate(tqdm(train_dl)):
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            inputs = torch.squeeze(inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(targets, outputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

        val_loss = 0
        model.eval()
        ground_truth, predictions = None, None
        for i, (inputs, targets) in enumerate(tqdm(val_dl)):
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            inputs = torch.squeeze(inputs)
            outputs = model(inputs)
            if ground_truth is None:
                ground_truth = targets
                predictions = outputs
            else:
                ground_truth = torch.cat((ground_truth, targets), 0)
                predictions = torch.cat((predictions, outputs), 0)

            val_loss += criterion(outputs, targets)
        val_loss = val_loss / len(val_dl)

        print("validation loss at epoch {}:".format(epoch), val_loss.item())
        if best_loss > val_loss:
            best_loss = val_loss
            best_age = 0
        else:
            best_age += 1
            print("patience: {}/{}".format(best_age, config.patience))
            if best_age == int(config.patience / 2):
                print("50% patience reached, decrease learning rate")
                lr /= 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    print("New learning rate: ", param_group['lr'])
            if best_age >= config.patience:
                print("patience reached, stop training")
                break

        torch.save(model.state_dict(), save_path)

    time_delta = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_delta // 60, time_delta % 60
    ))

    print(ground_truth[100])
    print(predictions[100])
    utils.plot_pred(ground_truth, predictions, name)


nn_params = [(4,1024)]#[(4, 1024), (3, 1024), (2, 1024), (1, 1024),
            # (4, 512), (3, 512), (2, 512), (1, 512),
            # (4, 256), (3, 256), (2, 256), (1, 256),
            # (4, 128), (3, 128), (2, 128), (1, 128)]

for p in nn_params:
    train_piglet(n_layers=p[0], layer_width=p[1])
