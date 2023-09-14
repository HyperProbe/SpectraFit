from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torch.optim import Adam
from torch.nn import MSELoss
import time

from data_processing.datasets import SpectraDataset
from model import SpectraMLP
import config


"""
    - created to train on the synthetic datasets
    - saves best model an result summary in the results folder
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lr = config.lr
continous = False

save_path = 'results/model_checkpoints/best_model.pth'

model = SpectraMLP(config.molecule_count)
model.to(device)
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=lr)
loss = 0.0

if continous:
    spectra_dataset = SpectraDataset(config.dataset_path, 50000)
    val_dl = DataLoader(spectra_dataset, batch_size=512, shuffle=False)
else:
    spectra_dataset = SpectraDataset(config.dataset_path)
    train_set, val_set = train_test_split(spectra_dataset, test_size=0.2)
    train_dl = DataLoader(train_set, batch_size=512, shuffle=False)
    val_dl = DataLoader(val_set, batch_size=16, shuffle=False)

best_loss, best_age = 100, 0
start = time.time()
for epoch in range(config.epochs):
    print('Epoch {}/{}'.format(epoch + 1, config.epochs))
    print('-' * 10)
    model.train()
    if continous: train_dl = DataLoader(spectra_dataset, batch_size=512, shuffle=False)
    # Iterate through training data loader
    for i, (inputs, targets) in enumerate(tqdm(train_dl)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        preds = outputs

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

    val_loss = 0
    model.eval()
    for i, (inputs, targets) in enumerate(tqdm(val_dl)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
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

    torch.save(model, save_path)

time_delta = time.time() - start
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_delta // 60, time_delta % 60
))
