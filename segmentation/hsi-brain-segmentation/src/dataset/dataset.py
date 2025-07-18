from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import os
import spectral
import numpy as np
import cv2
import random
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from torchvision.transforms import (
    Compose,
    ToTensor,
    Grayscale,
    Resize,
    v2,
    Normalize,
    InterpolationMode,
    CenterCrop,
    GaussianBlur,
)
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.util.constants import FIVES_DIR, FIVES_RANDOM_CROPS_DIR, HELICOID_DIR, HELICOID_WITH_LABELS_DIR

spectral.settings.envi_support_nonlowercase_params = True


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        image_transform=None,
        label_transform=None,
        augmentation=None,
    ):
        """
        Args:
            image_dir (string): Directory with all the original images.
            label_dir (string): Directory with all the labels.
            image_transform (callable, optional): Optional transform to be applied on a sample.
            label_transform (callable, optional): Optional transform to be applied on a sample.
            augmentation (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.augmentation = augmentation
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(
            self.label_dir, self.images[idx]
        )  # Assuming label and image files are named the same

        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name).convert(
            "L"
        )  # Convert label image to grayscale if needed

        if self.augmentation:
            image, label = self.augmentation(image, label)

        if self.image_transform:
            image = self.image_transform(image)

        if self.label_transform:
            label = self.label_transform(label)

        return image, label


class HSIDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_transform=None,
        window=None,
        with_gt=False,
        exclude_labeled_data=False,
        augmentation=None,
        with_img=True,
        rgb=False,
        rgb_channels=(425, 192, 109),
        ring_label_dir=None,
        classes=1,
    ):
        """
        Initialize the dataset with the path to the data.
        Args:
        root_dir (str): Path to the directory containing subdirectories with the HSI data and labels.
        image_transform (callable, optional): Optional transform to be applied on a sample.
        window (tuple, optional): The start and end wavelengths of the window to use.
        """
        self.root_dir = root_dir
        self.data_paths = []  # To store paths of the hyperspectral images and labels
        self.image_transform = image_transform
        self.window = window
        self.with_gt = with_gt
        self.with_img = with_img
        self.augmentation = augmentation
        self.rgb = rgb
        self.rgb_channels = rgb_channels
        self.ring_label_dir = ring_label_dir
        self.classes = classes
        labeled_data = ["004-02", "012-02", "021-01", "027-02", "030-02"]

        subdirs = os.listdir(root_dir)
        if exclude_labeled_data:
            subdirs = [subdir for subdir in subdirs if subdir not in labeled_data]

        # Iterate through each subdirectory in the root directory
        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                raw_path = os.path.join(subdir_path, "raw.hdr")
                gt_path = os.path.join(subdir_path, "gtMap.hdr")
                if with_gt:
                    gt_path = os.path.join(subdir_path, "gtMap.png")
                img_path = os.path.join(subdir_path, "image.jpg")
                dark_path = os.path.join(subdir_path, "darkReference.hdr")
                white_path = os.path.join(subdir_path, "whiteReference.hdr")

                if (
                    os.path.exists(raw_path)
                    and (os.path.exists(gt_path) or ring_label_dir is not None)
                    and os.path.exists(img_path)
                ):
                    self.data_paths.append(
                        (raw_path, gt_path, img_path, dark_path, white_path)
                    )

        self.data_paths.sort()

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        Retrieve the HSI image and corresponding label at the index `idx`.
        Args:
        idx (int): The index of the item.

        Returns:
        tuple: (image, label) where `image` is a hyperspectral image and `label` is the corresponding ground truth map.
        """
        raw_path, gt_path, img_path, dark_path, white_path = self.data_paths[idx]

        # Load the hyperspectral image and label
        hsi_image = spectral.open_image(raw_path).load()
        dark_reference = spectral.open_image(dark_path).load()
        white_reference = spectral.open_image(white_path).load()

        dark_full = np.tile(dark_reference, (hsi_image.shape[0], 1, 1))
        white_full = np.tile(white_reference, (hsi_image.shape[0], 1, 1))

        if self.with_gt:
            label = Image.open(gt_path).convert("L")
            label = (np.array(label) >= 128).astype(int)
            label = transforms.ToTensor()(label)
        elif self.ring_label_dir:
            image_dir = os.path.basename(os.path.dirname(white_path))
            ring_label_filename = f"{image_dir}.png"
            ring_label_path = os.path.join(self.ring_label_dir, ring_label_filename)
            if os.path.exists(ring_label_path):
                label = Image.open(ring_label_path).convert("L")
            else:
                label = Image.new("L", (hsi_image.shape[0], hsi_image.shape[1]), 0)
            label = transforms.ToTensor()(label)

        else:
            label = spectral.open_image(gt_path).load()
            label = (label.transpose(2, 0, 1) == 3).astype(int)
            label = torch.tensor(label, dtype=torch.int8)

        hsi_image = (hsi_image - dark_full) / (white_full - dark_full)
        hsi_image[hsi_image <= 0] = 10**-2
        hsi_image = hsi_image.transpose(2, 0, 1)

        if self.rgb:
            red, green, blue = self.rgb_channels
            hsi_image = np.stack(
                [hsi_image[red, :, :], hsi_image[green, :, :], hsi_image[blue, :, :]],
                axis=0,
            )
            hsi_image = (hsi_image - np.min(hsi_image)) / (
                np.max(hsi_image) - np.min(hsi_image)
            )

        if self.window is not None:
            channels = self.get_window_from_wavelengths(self.window)
            hsi_image_median = np.median(
                hsi_image[channels[0] : channels[1], :, :], axis=0
            )
            hsi_image = (hsi_image_median - np.min(hsi_image_median)) / (
                np.max(hsi_image_median) - np.min(hsi_image_median)
            )
            hsi_image = np.expand_dims(hsi_image, axis=0)

        # Convert to PyTorch tensors
        hsi_image = torch.tensor(hsi_image, dtype=torch.float32)

        if self.augmentation:
            hsi_image, label = self.augmentation(hsi_image, label)

        if self.image_transform and self.label_transform:
            hsi_image = self.image_transform(hsi_image)
            label = self.label_transform(label)

        if self.with_img:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = img[:, :, [2, 1, 0]]
            img = self.center_crop(img, (224, 224))
            return hsi_image, label, img

        return hsi_image, label

    def get_window_from_wavelengths(self, wavelengths):
        """
        Get the window indices corresponding to the given wavelengths.
        Args:
        wavelengths (tuple): The start and end wavelengths of the window.

        Returns:
        tuple: The start and end indices of the window.
        """
        img = spectral.open_image(self.data_paths[0][0])
        wavelength_array = np.array(img.metadata["wavelength"]).astype(float)
        indices = np.where(
            (wavelength_array >= wavelengths[0]) & (wavelength_array <= wavelengths[1])
        )[0]
        start_idx, end_idx = indices[0], indices[-1]
        return (start_idx, end_idx)

    def get_mean_std(self):
        """
        Compute the mean and standard deviation of the dataset.
        """
        if os.path.exists("./normalization_coefficients/mean.npy") and os.path.exists(
            "./normalization_coefficients/std.npy"
        ):
            mean = np.load("./normalization_coefficients/mean.npy")
            std = np.load("./normalization_coefficients/std.npy")
            return mean, std

        # Initialize variables to store the sum and sum of squares
        sum_ = 0
        sum_sq = 0
        num_pixels = 0

        # Iterate through each image in the dataset
        for i in range(len(self)):
            image, _ = self[i]
            sum_ += image.sum()
            sum_sq += (image**2).sum()
            num_pixels += image.numel()

        # Compute the mean and standard deviation
        mean = sum_ / num_pixels
        std = ((sum_sq / num_pixels) - (mean**2)) ** 0.5
        np.save("./normalization_coefficients/mean.npy", mean)
        np.save("./normalization_coefficients/std.npy", std)

        return mean, std

    def crop_dataset(self):
        """
        Normalize the dataset using the given mean and standard deviation.
        Args:
        """

        self.image_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
            ]
        )
        label_transform_list = [transforms.CenterCrop(224)]
        if self.classes > 1:
            label_transform_list.append(OneHotEncodeTransform(self.classes))
            if self.ring_label_dir is not None:
                label_transform_list.append(CustomLabelTransform())
        self.label_transform = transforms.Compose(label_transform_list)

    def center_crop(self, image, crop_size):
        """
        Perform a center crop on the image.

        Args:
        image (numpy array): The input image.
        crop_size (tuple): The size of the crop (height, width).

        Returns:
        numpy array: The cropped image.
        """
        height, width, _ = image.shape
        new_height, new_width = crop_size

        # Calculate the coordinates for the center crop
        start_x = width // 2 - (new_width // 2)
        start_y = height // 2 - (new_height // 2)

        # Perform the crop
        cropped_image = image[
            start_y : start_y + new_height, start_x : start_x + new_width
        ]

        return cropped_image


class SegmentationDatasetWithRandomCrops(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        image_transform=None,
        label_transform=None,
        crop_height=512,
        crop_width=512,
        threshold=0.1,
    ):
        """
        Args:
            image_dir (string): Directory with all the original images.
            label_dir (string): Directory with all the labels.
            image_transform (callable, optional): Optional transform to be applied on a sample.
            label_transform (callable, optional): Optional transform to be applied on a sample.
            crop_height (int): Desired height of the cropped image.
            crop_width (int): Desired width of the cropped image.
            threshold (float): Minimum required vessel ratio in the cropped image.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.threshold = threshold
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(self.label_dir, self.images[idx])

        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name).convert("L")  # Convert label image to grayscale

        image = np.array(image)
        label = np.array(label)
        label = (label > 0).astype(np.uint8)  # Binarize the label image

        # Apply the random cropping with condition
        cropped_image, cropped_label = self.random_crop_with_condition(
            image, label, self.crop_height, self.crop_width, self.threshold
        )

        if cropped_image is None or cropped_label is None:
            # Handling case where no valid crop is found, you could choose to not augment or use the full image as a fallback
            cropped_image = image
            cropped_label = label

        # Convert numpy arrays back to PIL images
        cropped_image = Image.fromarray(cropped_image)
        cropped_label[cropped_label == 1] = 255

        if self.image_transform:
            cropped_image = self.image_transform(cropped_image)

        if self.label_transform:
            cropped_label = self.label_transform(cropped_label)

        return cropped_image, cropped_label

    def calculate_vessel_ratio(self, mask):
        return np.sum(mask == 1) / np.prod(mask.shape)

    def random_crop_with_condition(
        self, image, mask, crop_height, crop_width, threshold=0.1
    ):
        assert (
            image.shape[:2] == mask.shape[:2]
        ), "Image and mask must have the same dimensions."

        height, width = image.shape[:2]
        for _ in range(1000):  # Try up to 100 times to find a valid crop
            x = random.randint(0, width - crop_width)
            y = random.randint(0, height - crop_height)
            cropped_image = image[y : y + crop_height, x : x + crop_width]
            cropped_mask = mask[y : y + crop_height, x : x + crop_width]

            if self.calculate_vessel_ratio(cropped_mask) >= threshold:
                return cropped_image, cropped_mask

        # If no valid crop is found, return a center crop
        center_x = (width - crop_width) // 2
        center_y = (height - crop_height) // 2
        center_cropped_image = image[
            center_y : center_y + crop_height, center_x : center_x + crop_width
        ]
        center_cropped_mask = mask[
            center_y : center_y + crop_height, center_x : center_x + crop_width
        ]

        return center_cropped_image, center_cropped_mask


# Custom one-hot encoding transform
class OneHotEncodeTransform:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, label):
        """
        label: Tensor of shape [1, H, W] after ToTensor(),
               where each pixel is in {0, 1, ..., num_classes-1}.
        """
        # Squeeze out the channel dimension to get shape [H, W]
        label = label.squeeze(0).long()  # (H, W)

        # One-hot encode to [H, W, num_classes]
        label = F.one_hot(label, num_classes=self.num_classes)

        # Rearrange to [num_classes, H, W]
        label = label.permute(2, 0, 1).float()

        return label


class CustomLabelTransform:
    """
    Custom label transform to convert [0, 1, 0] to [0, 0, 1]
    """

    def __call__(self, label):
        mask = (label[0] == 0) & (label[1] == 1) & (label[2] == 0)

        # For those locations, set the second channel to 0 and the third channel to 1.
        label[1, mask] = 0
        label[2, mask] = 1

        return label


def build_FIVES_random_crops_dataloaders(
    batch_size=8,
    num_channels=1,
    proportion_bloodvessels=0.1,
    width=512,
    height=512,
    load_from_path=FIVES_RANDOM_CROPS_DIR,
    kernel_size=None,
    sigma=None,
    classes=1,
):
    train_image_path = FIVES_DIR / "train" / "Original"
    train_label_path = FIVES_DIR / "train" / "GroundTruth"
    test_image_path = FIVES_DIR / "test" / "Original"
    test_label_path = FIVES_DIR / "test" / "GroundTruth"

    np.random.seed(42)

    transforms_list = [Grayscale(num_output_channels=1), ToTensor()]
    if num_channels == 1:
        if kernel_size and sigma:
            transforms_list.append(GaussianBlur(kernel_size, sigma))
        image_transform = Compose(transforms_list)

    else:
        image_transform = Compose([ToTensor()])

    label_transform = (
        Compose([ToTensor()])
        if classes == 1
        else Compose([ToTensor(), OneHotEncodeTransform(classes)])
    )

    if load_from_path is None:
        random_crop_dataset = SegmentationDatasetWithRandomCrops(
            train_image_path,
            train_label_path,
            image_transform,
            label_transform,
            crop_width=width,
            crop_height=height,
            threshold=proportion_bloodvessels,
        )

        testset = SegmentationDatasetWithRandomCrops(
            test_image_path,
            test_label_path,
            image_transform,
            label_transform,
            crop_width=width,
            crop_height=height,
            threshold=proportion_bloodvessels,
        )

        # Prepare DataLoader
        train_size = int(0.9 * len(random_crop_dataset))
        train_indices = np.random.choice(
            len(random_crop_dataset), train_size, replace=False
        )
        val_indices = np.setdiff1d(np.arange(len(random_crop_dataset)), train_indices)

        train_dataset = Subset(random_crop_dataset, train_indices)
        val_dataset = Subset(random_crop_dataset, val_indices)

    else:
        if isinstance(load_from_path, str):
            load_from_path = Path(load_from_path)
            
        train_path = load_from_path / "train"
        val_path = load_from_path / "validation"
        test_path = load_from_path / "test"

        train_dataset = SegmentationDataset(
            train_path / "Original",
            train_path / "GroundTruth",
            image_transform,
            label_transform,
        )

        val_dataset = SegmentationDataset(
            val_path / "Original",
            val_path / "GroundTruth",
            image_transform,
            label_transform,
        )

        testset = SegmentationDataset(
            test_path / "Original",
            test_path / "GroundTruth",
            image_transform,
            label_transform,
        )
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    validationloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    print(
        f"Number of samples in the training set: {len(train_dataset)}, validation set: {len(val_dataset)}"
    )
    print(f"Number of samples in the test set: {len(testset)}")

    return trainloader, validationloader, testloader


def build_FIVES_dataloaders(
    batch_size=8,
    proportion_augmented_data=0.1,
    num_channels=1,
    width=512,
    height=512,
    cropped=False,
    classes=1,
):
    train_image_path = FIVES_DIR / "train" / "Original"
    train_label_path = FIVES_DIR / "train" / "GroundTruth"
    test_image_path = FIVES_DIR / "test" / "Original"
    test_label_path = FIVES_DIR / "test" / "GroundTruth"
    np.random.seed(42)

    # Define transformations for images
    # normalization = (
    #     Normalize(mean=[0.3728, 0.1666, 0.0678], std=[0.1924, 0.0956, 0.0395])
    #     if num_channels == 3
    #     else Normalize(mean=[0.2147], std=[0.1163])
    # )
    random_crop_transform_list = []
    transforms_list = []
    if num_channels == 1:
        transforms_list.append(Grayscale(num_output_channels=1))
        random_crop_transform_list.append(Grayscale(num_output_channels=1))

    random_crop_transform_list.append(ToTensor())
    transforms_list.append(
        (
            CenterCrop((width, height))
            if cropped
            else Resize((width, height), interpolation=InterpolationMode.BICUBIC)
        ),  # Resize images to 512x512
    )
    transforms_list.append(ToTensor())
    image_transform = Compose(transforms_list)

    # Define transformations for labels, if needed
    label_transform_list = [
        (
            CenterCrop((width, height))
            if cropped
            else Resize((width, height), interpolation=InterpolationMode.NEAREST_EXACT)
        ),
        ToTensor(),  # Convert label to a tensor
    ]

    augmentation = v2.RandomApply(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=90),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    )

    random_crop_image_transform = Compose(random_crop_transform_list)
    random_crop_label_transform_list = [ToTensor()]

    if classes > 1:
        label_transform_list.append(OneHotEncodeTransform(classes))
        random_crop_label_transform_list.append(OneHotEncodeTransform(classes))

    random_crop_label_transform = Compose(random_crop_label_transform_list)
    label_transform = Compose(label_transform_list)

    random_crop_dataset = SegmentationDatasetWithRandomCrops(
        train_image_path,
        train_label_path,
        random_crop_image_transform,
        random_crop_label_transform,
        crop_width=width,
        crop_height=height,
    )

    dataset = SegmentationDataset(
        train_image_path,
        train_label_path,
        image_transform,
        label_transform,
    )

    testset = SegmentationDataset(
        test_image_path, test_label_path, image_transform, label_transform
    )

    augmented_dataset = SegmentationDataset(
        train_image_path,
        train_label_path,
        image_transform,
        label_transform,
        augmentation,
    )

    # Prepare DataLoader
    train_size = int(0.9 * len(dataset))
    train_indices = np.random.choice(len(dataset), train_size, replace=False)
    val_indices = np.setdiff1d(np.arange(len(dataset)), train_indices)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    augmented_dataset = Subset(
        augmented_dataset,
        train_indices[: int(proportion_augmented_data * len(train_indices))],
    )
    random_crop_dataset = Subset(
        random_crop_dataset,
        train_indices[: int(proportion_augmented_data * len(train_indices))],
    )

    train_dataset = ConcatDataset(
        [train_dataset, random_crop_dataset, augmented_dataset]
    )
    print(
        f"Number of samples in the training set: {len(train_dataset)}, validation set: {len(val_dataset)}"
    )
    print(f"Number of samples in the test set: {len(testset)}")

    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    validationloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    return trainloader, validationloader, testloader


def create_montage(dataset, num_images=10, show_windowed=False):
    # Define the number of images you want to show in the montage
    num_images = min(num_images, len(dataset))

    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
    cmap = None

    for i in range(num_images):
        sample = dataset[i]
        if len(sample) == 2 or show_windowed:
            image, label = sample[0], sample[1].float()
            image = image.cpu().numpy().squeeze()
            label = label.cpu().numpy().squeeze()

            if image.shape[0] == 1:  # Grayscale (single channel)
                image = np.stack(
                    [image, image, image], axis=-1
                )  # Convert to 3-channel grayscale RGB
                cmap = "gray"
            elif image.shape[0] == 3:  # RGB image
                image = image.transpose(1, 2, 0)  # Convert to HWC format
        else:
            image, label = sample[2], sample[1]

            # Convert image and label to numpy arrays for plotting
            if isinstance(image, torch.Tensor):
                image = image.numpy().transpose(1, 2, 0)
            if isinstance(label, torch.Tensor):
                label = label.numpy().squeeze()
                overlay = np.zeros_like(image)
                overlay[label == 1] = [0, 255, 0]

        # Plot the image
        axes[i, 0].imshow(image, cmap=cmap)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Image {i}".format(i=i))

        # Plot the label
        axes[i, 1].imshow(label, cmap="gray")
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Label")

    plt.tight_layout()
    plt.show()


def build_hsi_dataloader(
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    batch_size=8,
    window=None,
    exclude_labeled_data=False,
    augmented=False,
    rgb=False,
    rgb_channels=(425, 192, 109),
    ring_label_dir=None,
    classes=1,
):
    assert not (rgb and window is not None), "If rgb=True, window must be None."
    assert not (window is not None and rgb), "If window is set, rgb must be False."

    path = HELICOID_DIR

    dataset = HSIDataset(
        path,
        window=window,
        exclude_labeled_data=exclude_labeled_data,
        with_img=False,
        rgb=rgb,
        rgb_channels=rgb_channels,
        ring_label_dir=ring_label_dir,
        classes=classes,
    )
    dataset.crop_dataset()

    total_samples = len(dataset)

    # Assert that the splits sum up to exactly 1
    if not abs((train_split + val_split + test_split) - 1.0) < 1e-6:
        raise ValueError("train_split, val_split, and test_split must sum up to 1.")

    # Calculate the number of samples for each split
    train_size = int(train_split * total_samples)
    val_size = int(val_split * total_samples)
    test_size = total_samples - train_size - val_size  # Ensure all samples are used

    augmentation = v2.RandomApply(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=90),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    )
    augmented_dataset = HSIDataset(
        path,
        window=window,
        exclude_labeled_data=exclude_labeled_data,
        augmentation=augmentation,
        with_img=False,
        rgb=rgb,
        rgb_channels=rgb_channels,
        ring_label_dir=ring_label_dir,
        classes=classes,
    )
    augmented_dataset.crop_dataset()

    # Generate shuffled indices for the dataset
    indices = list(range(total_samples))
    random.shuffle(indices)

    # Split the indices based on calculated sizes
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create subsets for each split
    trainset = Subset(dataset, train_indices)
    valset = Subset(dataset, val_indices)
    testset = Subset(dataset, test_indices)

    # Create DataLoaders for each subset
    if augmented:
        trainset = ConcatDataset([trainset, augmented_dataset])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, validationloader, testloader


def build_hsi_testloader(
    window=None,
    batch_size=1,
    rgb=False,
    rgb_channels=(425, 192, 109),
    classes=1,
    ring_label_dir=None,
    choose_indices=[0, 1, 2, 3, 4],
):
    assert not (rgb and window is not None), "If rgb=True, window must be None."
    assert not (window is not None and rgb), "If window is set, rgb must be False."

    path = HELICOID_WITH_LABELS_DIR
    testset = HSIDataset(
        path,
        with_gt=True,
        window=window,
        with_img=False,
        rgb=rgb,
        rgb_channels=rgb_channels,
        classes=classes,
        ring_label_dir=None,
    )
    testset.crop_dataset()
    testloader_target = DataLoader(
        Subset(testset, choose_indices), batch_size=batch_size, shuffle=False
    )
    if ring_label_dir is not None:
        testset_with_ring_labels = HSIDataset(
            path,
            with_gt=False,
            window=window,
            with_img=False,
            rgb=rgb,
            rgb_channels=rgb_channels,
            classes=classes,
            ring_label_dir=ring_label_dir,
            exclude_labeled_data=False,
        )
        testset_with_ring_labels.crop_dataset()
        testloader_ring = DataLoader(
            Subset(testset_with_ring_labels, choose_indices),
            batch_size=batch_size,
            shuffle=False,
        )
        return testloader_target, testloader_ring

    return testloader_target


def _get_ratio(label):
    count_ones = torch.sum(label == 1).item()
    total_elements = label.numel()
    return count_ones / total_elements


def save_random_crops_dataset_to_path(trainset, valset, testset, path, threshold):
    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "validation")
    test_path = os.path.join(path, "test")

    for i, data in enumerate(trainset):
        bloodvessel_ratio = _get_ratio(data[1])
        if bloodvessel_ratio >= threshold:
            img = transforms.ToPILImage()(data[0])
            img.save(f"{train_path}/Original/{i}.png")
            label = transforms.ToPILImage()(data[1])
            label.save(f"{train_path}/GroundTruth/{i}.png")
            print(f"Bloodvessel ratio: {bloodvessel_ratio}, Image {i} saved")

    for i, data in enumerate(valset):
        bloodvessel_ratio = _get_ratio(data[1])
        if bloodvessel_ratio >= threshold:
            img = transforms.ToPILImage()(data[0])
            img.save(f"{val_path}/Original/{i}.png")
            label = transforms.ToPILImage()(data[1])
            label.save(f"{val_path}/GroundTruth/{i}.png")
            print(f"Bloodvessel ratio: {bloodvessel_ratio}, Image {i} saved")

    for i, data in enumerate(testset):
        bloodvessel_ratio = _get_ratio(data[1])
        if bloodvessel_ratio >= threshold:
            img = transforms.ToPILImage()(data[0])
            img.save(f"{test_path}/Original/{i}.png")
            label = transforms.ToPILImage()(data[1])
            label.save(f"{test_path}/GroundTruth/{i}.png")
            print(f"Bloodvessel ratio: {bloodvessel_ratio}, Image {i} saved")


def get_wavelengths_from_metadata(data_path=HELICOID_DIR / "004-02/raw.hdr"):
    img = spectral.open_image(data_path)
    wavelength_array = np.array(img.metadata["wavelength"]).astype(float)

    return wavelength_array
