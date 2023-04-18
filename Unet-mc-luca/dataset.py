from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import os
import cv2
import numpy as np
import scipy
import random
from PIL import Image
import imageio.v3 as iio



class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to output folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. cropping)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. cast as tensors, shape manipulation)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            slice=None, 
            augmentation=None, 
            preprocessing=None,
            mode=None
    ):
        #self.ids = os.listdir(images_dir)
        self.images_dir =  images_dir #[os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        #self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.slice = slice
        self.mode = mode

    
    def __getitem__(self, i):
        
        # read data

        image = scipy.io.loadmat('./train/185_185_121_1/hypercube_hypoxia.mat')
        base = scipy.io.loadmat('./train/185_185_121_1/hypercube_baseline.mat')

        abs_coefficients_brain = np.asarray([2325.6*0.0375, 2325.6*(1-0.0375), 70, 10, 4, 1])     
        abs_coefficients_vessels = np.asarray([2325.6*0.0375*0.85, 2325.6*0.0375*(1-0.85), 50, 1, 0, 0])     
        abs_spectra = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        if self.images_dir == "train":
            s = random.randint(0, self.slice)
            #print(s)
        else:
            s = random.randint(self.slice, image['hypercube'].shape[-1]-1)
            #print(s)

        image = (image['hypercube'][:,:,s]-base['hypercube'][:,:,s]).astype('float')
        image = np.expand_dims(image, axis=2)
        #print("image", image.shape)

        if self.mode=="pathlength":
            mask = scipy.io.loadmat('./train/185_185_121_1/PPLvessel_hypoxia.mat') #PPLvessel_baseline
            mask = mask['ppl_2'][:,:,s].astype('float')
            mask = np.expand_dims(mask, axis=2)
        else:
            
            label = cv2.imread("./train/185_185_121_1/Mouse_brain_segmented.png",0)
            label = label[10:,:]
            label = cv2.resize(label, (185, 185), interpolation = cv2.INTER_NEAREST)
            label = cv2.flip(label, 0)

            # 2D
            # mask = np.zeros_like(label)
            # mask[label==127] = np.sum(abs_coefficients_brain*abs_spectra)
            # mask[label==255] = np.sum(abs_coefficients_vessels*abs_spectra)
            # mask = np.expand_dims(mask, axis=2)
            

            # 3D
            # Define the size of the images and the number of images
            image_size = (label.shape[0], label.shape[1])
            num_images = len(abs_spectra)

            # Create an empty array to store the stacked images
            stacked_images = np.empty((num_images,) + image_size)

            # Load the images and stack them
            for i in range(num_images):
   
                stacked_images[i][label == 127] = abs_coefficients_brain[i]*abs_spectra[i]
                stacked_images[i][label == 255] = abs_coefficients_vessels[i]*abs_spectra[i]
            mask = stacked_images.transpose(1, 2, 0).astype('float32')
            #print("mask", np.unique(label))
            
           
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        if self.images_dir == "train":
            return 4000
        else:
            return 80