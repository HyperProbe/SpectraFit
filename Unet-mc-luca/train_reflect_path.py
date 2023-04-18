import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from dataset import Dataset 
from utils import *
import albumentations as albu

import torch
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils #as smp_utils


x_train_dir = "train" #os.path.join(DATA_DIR, 'train')
y_train_dir = "train" #os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = "val" #os.path.join(DATA_DIR, 'val')
y_valid_dir = "val" #os.path.join(DATA_DIR, 'valannot')

#x_test_dir = os.path.join(DATA_DIR, 'test')
#y_test_dir = os.path.join(DATA_DIR, 'testannot')

ENCODER = 'resnext50_32x4d'
ENCODER_WEIGHTS = None #'imagenet'
Slice = 80
ACTIVATION = None # sigmoid
DEVICE = 'cuda'
Mode = "absorption" # "pathlength" or "absorption" 
if Mode=="absorption":
    CLASSES = 6
else:
    CLASSES = 1

# create model with pretrained encoder
model = smp.Unet(
    in_channels=1,
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=CLASSES, 
    activation=ACTIVATION,
)
print(model)
#preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(),
    slice=Slice,
    mode=Mode # "pathlength" or "absorption" 
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(),
    slice=Slice,
    mode=Mode # "pathlength" or "absorption" 
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

loss = smp.utils.losses.L1Loss()
metrics = [
    smp.utils.losses.L1Loss(),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# train model for N epochs

N = 100
score = 1e5

for i in range(0, N):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    print(valid_logs)
    if score > valid_logs['l1_loss']:
        score = valid_logs['l1_loss']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

torch.save(model, './last_model.pth')