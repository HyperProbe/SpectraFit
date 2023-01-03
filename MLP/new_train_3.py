from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from tqdm import tqdm

from torch.nn import Module
from torch.optim import Adam
from torch.nn import L1Loss
from torch.optim import lr_scheduler
import time

from new_data_3 import HELICoiD
from new_MLP import SpectraMLP

def rmae(outputs, targets):
    loss = torch.abs(outputs - targets) / torch.abs(targets)
    loss = loss.mean(0) 
    return loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_inputs = 301 
n_outputs = 3
epochs = 100
lr = 0.01

save_path='HELICoiD_3params_best_model.pth'

model = SpectraMLP(n_inputs, n_outputs)
model.to(device)
criterion = L1Loss()
optimizer = Adam(model.parameters(), lr=lr)
loss = 0.0

path_to_data = "./syn_data_3_params_w_hbb_cox/y_and_params/"
HELICoiD_dataset = HELICoiD(path_to_data)

train_size_percent = 0.9
val_size_percent = 1 - train_size_percent
datset_size= HELICoiD_dataset.__len__()
train_set, val_set = random_split(HELICoiD_dataset, [round(datset_size*train_size_percent), round(datset_size*val_size_percent)])

train_dl = DataLoader(train_set, batch_size=16448, shuffle=True)
val_dl = DataLoader(val_set, batch_size=1024, shuffle=False)

start = time.time()
print(next(model.parameters()).is_cuda)
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch+1, epochs))
    print('-' * 10)
    model.train()

    # Iterate through training data loader
    for i, (inputs, targets) in enumerate(tqdm(train_dl)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        preds = outputs
        
        
        #rmae = torch.abs(outputs - targets) / torch.abs(targets)
        #rmae = rmae.mean(0) 
        loss = rmae(outputs, targets)  
        print(loss)  

        loss.sum().backward()
        optimizer.step()
    
    val_loss = 0
    model.eval()
    for i, (inputs, targets) in enumerate(tqdm(val_dl)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        val_loss += rmae(outputs, targets)

    print("validation loss at epoch {}:".format(epoch), val_loss/round(datset_size*val_size_percent))

    torch.save(model, save_path)

time_delta = time.time() - start
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_delta // 60, time_delta % 60
))

    