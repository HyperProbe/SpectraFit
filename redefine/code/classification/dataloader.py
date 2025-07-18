import os
import json
import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import PCA

class Helicoid_Dataset_Loader():
    def __init__(self, files, n_dim=None):
        self.files = files
        self.n_dim = n_dim
        if n_dim is not None:
            self.pca = PCA(n_components=n_dim)

    def load_data(self, patient_folders, mode='labeled', return_img_shape=False):
        data = []
        labels = []
        for patient_folder in patient_folders:
            print(f"loading image {patient_folder}")
            patient_folder = os.path.join('/mnt/Drive3/ivan/martin_ivan/datasets/npj_database/', patient_folder)
            img_data = []
            img_labels = np.load(os.path.join(patient_folder, 'gtMap.npy')).astype(int)
            img_shape = img_labels.squeeze().shape
            for file in self.files:
                img_data_all = np.load(os.path.join(patient_folder, file))
                if mode == 'labeled':
                    img_data.append(img_data_all[(img_labels !=0)])
                elif mode == 'all':
                    img_data.append(img_data_all.reshape(-1, img_data_all.shape[-1]))
                else:
                    raise ValueError("Unknown mode")
            img_data = np.concatenate(img_data, axis=1)
            data.append(img_data)
            if mode == 'labeled':
                labels.append(img_labels[(img_labels !=0)])
            elif mode == 'all':
                labels.append(img_labels.reshape(-1))
            else:
                raise ValueError("Unknown mode")
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0) - 1
        if return_img_shape:
            return data, labels, img_shape
        return data, labels
    
    def balance_dataset(self, data, labels, per_class_count=3000):
        class_counts = np.unique(labels, return_counts=True)[1]
        data_balanced = []
        labels_balanced = []
        for i in range(len(class_counts)):
            idx_class = np.where(labels == i)[0]
            np.random.seed(0)
            random_idx = np.random.choice(idx_class, per_class_count, replace=False)
            data_balanced.append(data[random_idx])
            labels_balanced.append(labels[random_idx])
        data_balanced = np.concatenate(data_balanced, axis=0)
        labels_balanced = np.concatenate(labels_balanced, axis=0)
        return data_balanced, labels_balanced
    
    def to_device(self, data, labels):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        data = torch.tensor(data, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        return data, labels

    def get_train_dataset(self, train_patient_folders, balance_dataset=False):
        data, labels = self.load_data(train_patient_folders, mode='labeled')
        if balance_dataset:
            data, labels = self.balance_dataset(data, labels)
        if self.n_dim is not None:
            data, labels = self.pca.fit_transform(data)
        data, labels = self.to_device(data, labels)
        return TensorDataset(data, labels)
    
    def get_val_dataset(self, val_patient_folders):
        data, labels = self.load_data(val_patient_folders, mode='labeled')
        if self.n_dim is not None:
            data, labels = self.pca.transform(data)
        data, labels = self.to_device(data, labels)
        return TensorDataset(data, labels)
    
    def get_test_dataset(self, test_patient_folders):
        data, labels = self.load_data(test_patient_folders, mode='labeled')
        if self.n_dim is not None:
            data, labels = self.pca.transform(data)
        data, labels = self.to_device(data, labels)
        return TensorDataset(data, labels)
    
    def get_predict_datasets(self, test_patient_folders):
        img_datasets = []
        img_shapes = []
        for test_patient_folder in test_patient_folders:
            data, labels, img_shape = self.load_data([test_patient_folder], mode='all', return_img_shape=True)
            if self.n_dim is not None:
                data, labels = self.pca.transform(data)
            data, labels = self.to_device(data, labels)
            img_datasets.append(TensorDataset(data, labels))
            img_shapes.append(img_shape)
        return img_datasets, img_shapes

class HelicoidDataModule(pl.LightningDataModule):
    def __init__(self, files, fold="fold1"):
        super().__init__()
        self.fold = fold
        self.files = files
        self.setup()
        self.dataset_loader = Helicoid_Dataset_Loader(files)

    def setup(self, stage=None):
        with open('folds_new.json') as f:
            folds = json.load(f)

        if stage=="fit":
            self.dataset_train = self.dataset_loader.get_train_dataset(folds[self.fold]["train"])
            self.dataset_val = self.dataset_loader.get_val_dataset(folds[self.fold]["val"])
        if stage=="val":
            self.dataset_val = self.dataset_loader.get_val_dataset(folds[self.fold]["val"])
        if stage=="test":
            self.dataset_test = self.dataset_loader.get_test_dataset(folds[self.fold]["test"])
        if stage=="predict":
            self.datasets_predict, self.test_img_shapes = self.dataset_loader.get_predict_datasets(folds[self.fold]["test"])
            self.image_ids = folds[self.fold]["test"]

    def train_dataloader(self, batch_size=64):
        return DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    def val_dataloader(self, batch_size=1024):
        return DataLoader(self.dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    def test_dataloader(self, batch_size=1024): 
        return DataLoader(self.dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    def predict_dataloader(self, batch_size=1024):
        dataloaders = []
        for dataset in self.datasets_predict:
            dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False))
        return dataloaders, self.test_img_shapes, self.image_ids

    def sample_size(self):
        size = self.dataset_train.tensors[0].shape[1]
        print(f"------------- sample size: {size} -------------")
        return size
    
    def class_distribution(self):
        dist =  torch.unique(self.dataset_train.tensors[1], return_counts=True)[1].float()
        print(f"------------- class distribution: {dist} -------------")
        return dist
    
    def num_classes(self):
        num = len(torch.unique(self.dataset_train.tensors[1]))
        print(f"------------- number of classes: {num} -------------")
        return num
    
    def get_fold(self):
        return self.fold