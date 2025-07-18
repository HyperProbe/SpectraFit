import os
import torch
import segmentation_models_pytorch as smp
import wandb
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.dataset.dataset import build_FIVES_dataloaders, build_hsi_testloader
from sklearn.metrics import precision_score
import cv2
from typing import Literal
from ipywidgets import interact, FloatSlider, fixed
import torch.nn.functional as F
from monai.losses.cldice import SoftDiceclDiceLoss
from metrics.cldice import clDice

from src.model.FADA.feature_extractor import BaseFeatureExtractorWithDimReduction
from src.util.constants import RING_LABELS_DIR


def train_and_validate(
    model,
    trainloader,
    validationloader,
    criterion,
    optimizer,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
    with_overlays=False,
    save_to_wandb=False,
):
    """
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
        :param model_name: Model file name (default=None)
    Returns
        train_losses, val_losses: List of losses per epoch
    """
    train_losses, val_losses = [], []
    highest_dice = 0
    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device).float()
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loss += loss.item()

            if (
                i + 1
            ) % batch_print == 0:  # Adjust the condition based on your preference
                print(
                    f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / batch_print:.4f}"
                )
                running_loss = 0.0  # Reset running loss after printing

        # Calculate and print the average loss per epoch
        train_loss = train_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train/loss": train_loss}, step=epoch + 1)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(validationloader):
                inputs, labels = data[0].to(device), data[1].to(device).float()
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                if with_overlays and i == 0:
                    log_segmentation_example(model, data, device, epoch)
        val_loss = val_running_loss / len(validationloader)
        val_losses.append(val_loss)

        precision, _, _, _, dice_score = evaluate_model(
            model, validationloader, device, with_wandb=False
        )
        if model_name:
            if dice_score > highest_dice:
                highest_dice = dice_score
                torch.save(model.state_dict(), f"./models/{model_name}.pth")
                if save_to_wandb:
                    model_artifact = wandb.Artifact(f"{model_name}", type="model")
                    model_artifact.add_file(f"./models/{model_name}.pth")
                    wandb.log_artifact(model_artifact)

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "validation/loss": val_loss}, step=epoch + 1)
        wandb.log({"epoch": epoch + 1, "precision": precision}, step=epoch + 1)
        wandb.log({"epoch": epoch + 1, "dice_score": dice_score}, step=epoch + 1)

    return train_losses, val_losses


def log_segmentation_example(
    model,
    data,
    device,
    epoch,
    title="Validation Overlay",
    channel_reducer=None,
    ring_data=None,
    with_postprocessing=False,
):
    inputs, labels = data[0][0], data[1][0].to(device).float()
    prediction = predict(model, inputs, device)
    if with_postprocessing:
        if ring_data is not None:
            prediction = remove_rings_from_predictions(prediction, ring_data)
        prediction = apply_morphological_operations(prediction)

    # Convert the labels and prediction to integer masks
    class_labels = {
        0: "Background",
        1: "Blood Vessel",
    }
    if labels.shape[0] == 1:
        labels = labels.cpu().numpy().squeeze()
    else:
        labels = labels.argmax(dim=0).cpu().numpy()
        class_labels[2] = "Black Ring"  # Hardcoded for contrastive loss approach

    # Assuming inputs are in a suitable format (e.g., normalized between 0 and 1 or uint8)
    if channel_reducer:
        inputs = channel_reducer(inputs.unsqueeze(0).to(device)).squeeze(0)

    if isinstance(model.feature_extractor, BaseFeatureExtractorWithDimReduction):
        inputs = model.feature_extractor.forward_transform(
            inputs.unsqueeze(0).to(device)
        ).squeeze(0)

    input_image = inputs.cpu().numpy().squeeze()  # Convert to HWC format
    if input_image.shape[0] == 1:  # Grayscale (single channel)
        input_image = np.stack(
            [input_image, input_image, input_image], axis=-1
        )  # Convert to 3-channel grayscale RGB
    elif input_image.shape[0] == 3:  # RGB image
        input_image = input_image.transpose(1, 2, 0)  # Convert to HWC format

    image_min = input_image.min()
    image_max = input_image.max()

    # Apply min-max normalization to scale values to [0, 255]
    input_image = 255 * (input_image - image_min) / (image_max - image_min)
    input_image = input_image.astype(np.uint8)  # Convert to uint8 after scaling

    prediction = prediction.squeeze(0).cpu().numpy()

    # Prepare the masks for wandb logging
    mask_data = {
        "predictions": {
            "mask_data": prediction,  # The predicted mask
            "class_labels": class_labels,
        },
        "ground_truth": {
            "mask_data": labels,  # The ground truth mask
            "class_labels": class_labels,
        },
    }

    # Log the image and masks using wandb
    wandb.log(
        {title: wandb.Image(input_image, masks=mask_data, caption=f"Epoch {epoch+1}")},
        step=epoch + 1,
    )


def dice_coefficient(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def remove_rings_from_predictions(predictions, rings):
    """
    Set predictions in the regions specified by `rings` to 0 (background).

    Args:
        predictions (torch.Tensor): The model predictions (binary or multiclass).
        rings (torch.Tensor): A mask of the same spatial size as predictions indicating regions to ignore.

    Returns:
        torch.Tensor: Postprocessed predictions.
    """
    predictions[rings == 1] = 0
    return predictions


def apply_morphological_operations(predictions):
    """
    Apply morphological operations to smooth the predictions.

    Args:
        predictions (torch.Tensor): The binary model predictions.

    Returns:
        torch.Tensor: Smoothed predictions after morphological operations.
    """
    smoothed_predictions = []
    for prediction in predictions.cpu().numpy():
        prediction = prediction.astype(np.uint8)  # Convert to uint8 for OpenCV
        # Apply morphological closing (dilation followed by erosion)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE, kernel)
        # Apply morphological opening (erosion followed by dilation)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        smoothed_predictions.append(opened)
    smoothed_predictions = np.array(smoothed_predictions)
    return torch.tensor(smoothed_predictions, device=predictions.device)


def evaluate_model_with_postprocessing(
    model, dataloader, dataloader_with_rings, device, with_wandb=True, threshold=0.5
):
    model.eval()
    recall = 0
    precision = 0
    f1_score = 0
    accuracy = 0
    dice_score = 0
    cl_dice_score = 0

    with torch.no_grad():
        for (i, data), (_, ring_data) in zip(
            enumerate(dataloader), enumerate(dataloader_with_rings)
        ):
            inputs, labels = data[0].to(device), data[1].to(device).int()
            rings = ring_data[1].to(device).int()

            outputs = model(inputs)
            predictions = (outputs > threshold).long()
            predictions = remove_rings_from_predictions(predictions, rings)
            predictions = apply_morphological_operations(predictions)

            tp, fp, fn, tn = smp.metrics.get_stats(
                predictions, labels, mode="binary", threshold=threshold
            )

            recall += smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
            precision += smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
            f1_score += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

            dice_score += dice_coefficient(predictions, labels)
            pred_2d = predictions.squeeze().cpu().numpy()
            labels_2d = labels.squeeze().cpu().numpy()
            cl_dice_score += clDice(pred_2d, labels_2d)

    precision /= len(dataloader)
    recall /= len(dataloader)
    f1_score /= len(dataloader)
    accuracy /= len(dataloader)
    dice_score /= len(dataloader)
    cl_dice_score /= len(dataloader)

    if with_wandb:
        wandb.log(
            {
                "test/precision_postprocessed": precision,
                "test/recall_postprocessed": recall,
                "test/f1_score_postprocessed": f1_score,
                "test/accuracy_postprocessed": accuracy,
                "test/dice_score_postprocessed": dice_score,
                "test/cl_dice_score_postprocessed": cl_dice_score,
            }
        )

    print(
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, Dice Score: {dice_score:.4f}, CL Dice Score: {cl_dice_score:.4f}, Accuracy: {accuracy:.4f}"
    )
    return precision, recall, f1_score, accuracy, dice_score, cl_dice_score


def evaluate_model(model, dataloader, device, with_wandb=True, threshold=0.5):
    model.eval()
    recall = 0
    precision = 0
    f1_score = 0
    accuracy = 0
    dice_score = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data[0].to(device), data[1].to(device).int()
            outputs = model(inputs)
            num_classes = outputs.shape[1]
            if num_classes == 1:
                tp, fp, fn, tn = smp.metrics.get_stats(
                    outputs, labels, mode="binary", threshold=threshold
                )
            else:
                pred_multiclass = torch.argmax(outputs, dim=1)
                pred_multiclass = F.one_hot(
                    pred_multiclass, num_classes=num_classes
                ).permute(0, 3, 1, 2)
                tp, fp, fn, tn = smp.metrics.get_stats(
                    pred_multiclass.long(),
                    labels,
                    mode="multiclass",
                    num_classes=num_classes,
                )
                labels = torch.argmax(labels, dim=1)
                pred_multiclass = torch.argmax(pred_multiclass, dim=1)
                pred_multiclass[pred_multiclass == 2] = 0

            recall += smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
            precision += smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
            f1_score += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

            pred = outputs > 0.5 if num_classes == 1 else pred_multiclass

            dice_score += dice_coefficient(pred, labels)

    precision /= len(dataloader)
    recall /= len(dataloader)
    f1_score /= len(dataloader)
    accuracy /= len(dataloader)
    dice_score /= len(dataloader)
    if with_wandb:
        wandb.log(
            {
                "test/precision": precision,
                "test/recall": recall,
                "test/f1_score": f1_score,
                "test/accuracy": accuracy,
                "test/dice_score": dice_score,
            }
        )
    print(
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, Dice Score: {dice_score:.4f}, Accuracy: {accuracy:.4f}"
    )
    return precision, recall, f1_score, accuracy, dice_score


def model_pipeline(
    model,
    trainloader,
    validationloader,
    testloader,
    criterion,
    optimizer,
    config,
    project,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
    evaluate=True,
    with_overlays=False,
    save_to_wandb=False,
):
    with wandb.init(project=project, config=config, name=model_name):
        config = wandb.config
        train_loss, val_loss = train_and_validate(
            model,
            trainloader,
            validationloader,
            criterion,
            optimizer,
            epochs,
            model_name,
            device,
            batch_print,
            with_overlays,
            save_to_wandb,
        )
        if evaluate:
            if model_name:
                model.load_state_dict(torch.load(f"./models/{model_name}.pth"))
            evaluate_model(model, testloader, device)
        return model, train_loss, val_loss


def predict(model, data, device, with_sigmoid=True, threshold=0.5):
    model.to(device)
    model.eval()
    data = data.unsqueeze(0).to(device)  # Add batch dimension (B, C, H, W)
    with torch.no_grad():
        logits = model(data)
        out_channels = logits.shape[1]
        if out_channels == 1:
            probs = torch.sigmoid(logits) if with_sigmoid else logits
            prediction = (probs > threshold).float()
            prediction = prediction.squeeze(1)
        else:
            # -- Multi-class segmentation --
            probs = F.softmax(logits, dim=1)  # (B, out_channels, H, W)
            prediction = torch.argmax(probs, dim=1)  # (B, H, W)
    return prediction


def show_overlay(
    model,
    data,
    device,
    with_sigmoid=True,
    title=None,
    with_precision=True,
    with_postprocessing=True,
    threshold=0.5,
):
    prediction = predict(
        model, data[0], device, with_sigmoid=with_sigmoid, threshold=threshold
    )
    image = data[2]
    labels = data[1]

    prediction_np = prediction.cpu().numpy().squeeze(0)

    _plot_overlay(
        title, with_precision, with_postprocessing, image, labels, prediction_np
    )


def show_interactive_overlay(model, data, device, title, ring_data=None):
    image = get_image_from_hsi(data)

    labels = data[1]
    model.to(device)
    model.eval()
    data = data[0].unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(data)

    def apply_slider(
        threshold,
        prediction,
        title,
        with_sigmoid=True,
        with_postprocessing=True,
        with_precision=True,
    ):
        with torch.no_grad():
            if with_sigmoid:
                prediction = torch.sigmoid(prediction)
            prediction = (prediction > threshold).float()
        prediction = prediction.squeeze(1)

        prediction_np = prediction.cpu().numpy().squeeze(0)
        if ring_data is not None:
            prediction_np[ring_data.cpu().squeeze(0) == 1] = 0
        _plot_overlay(
            title, with_precision, with_postprocessing, image, labels, prediction_np
        )

    threshold_slider = FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Threshold:",
        readout_format=".2f",
    )
    interact(
        apply_slider,
        threshold=threshold_slider,
        prediction=fixed(prediction),
        title=title,
    )


def get_image_from_hsi(data):
    image = data[0].cpu().numpy().squeeze()
    # Example: picking 3 channels by index (replace with your real channel selection or Gaussian)
    if data[0].shape[0] == 1:
        input_image = np.stack(
            [image, image, image], axis=-1
        )  # Convert to 3-channel grayscale RGB
    else:
        input_image = np.stack(
            [
                image[425, :, :],
                image[192, :, :],
                image[109, :, :],
            ],
            axis=-1,
        )  # Convert to 3-channel grayscale

    image_min, image_max = input_image.min(), input_image.max()
    input_image = 255 * (input_image - image_min) / (image_max - image_min)
    input_image = input_image.astype(np.uint8)
    return input_image


def _plot_overlay(
    title, with_precision, with_postprocessing, image, labels, prediction_np
):
    if with_postprocessing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        prediction_np = cv2.morphologyEx(prediction_np, cv2.MORPH_OPEN, kernel)

    labels_np = labels.cpu().numpy().squeeze(0)

    overlay = np.zeros_like(image)
    # Green for prediction
    overlay[prediction_np == 1] = [0, 255, 0]
    # Blue for ground truth
    overlay[labels_np == 1] = [0, 0, 255]
    # Cyan for intersection (both prediction and ground truth are 1)
    overlay[(prediction_np == 1) & (labels_np == 1)] = [0, 255, 255]
    combined = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    if with_precision:
        precision = precision_score(labels_np.flatten(), prediction_np.flatten())
        dice_score = dice_coefficient(prediction_np, labels_np)
        if title:
            title += f" - Precision: {precision:.2f}"
            title += f" - Dice Score: {dice_score:.2f}"
        else:
            title = f"Precision: {precision:.2f}"
            title += f" - Dice Score: {dice_score:.2f}"

    plt.figure(figsize=(10, 10))
    plt.imshow(combined)
    plt.axis("off")  # Turn off axis numbers and ticks
    if title:
        plt.suptitle(title)
    plt.show()


def show_training_step(output, image):
    prediction = torch.sigmoid(output)
    prediction = (prediction > 0.5).float()
    img = image.squeeze().cpu().numpy()
    img = img[:, :, [2, 1, 0]]
    prediction = prediction.squeeze()
    overlay = np.zeros_like(img)
    overlay[prediction.cpu().numpy() == 1] = [0, 255, 0]
    combined = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(combined)
    plt.axis("off")  # Turn off axis numbers and ticks
    plt.show()


def get_model_paths(directory):
    """
    Given a directory structure, return a dictionary where the keys are the image indices
    and the values are the paths to the corresponding model files.

    Args:
        directory (str): The base directory containing the models.

    Returns:
        dict: A dictionary mapping image indices to model file paths.
    """
    model_paths = {}

    for subdir, _, files in os.walk(directory):
        for file in files:
            # Extract the image index from the directory name
            parts = os.path.normpath(subdir).split(os.sep)
            if len(parts) >= 2:  # Ensure the directory structure is as expected
                try:
                    image_index = int(parts[-1].replace("image", ""))
                    model_paths[image_index] = os.path.join(subdir, file)
                except ValueError:
                    # Skip directories that don't match the expected format
                    continue

    return dict(sorted(model_paths.items()))


def cross_testing(model_paths, device, build_func, model_params, window=None):
    total_precision = 0
    total_recall = 0
    total_accuracy = 0
    total_dice_score = 0
    total_cl_dice_score = 0
    for image_index, model_path in model_paths.items():
        test_indices = list({0, 1, 2, 3, 4} - {image_index})
        testloader, testloader_ring_dir = build_hsi_testloader(
            window=window,
            batch_size=1,
            ring_label_dir=RING_LABELS_DIR,
            choose_indices=test_indices,
        )
        model_params["path"] = model_path
        model_params["device"] = device
        model = build_func(**model_params)
        print(
            f"Testing model {model_path} hyperparameter-finetuned on image {image_index} testing on images {test_indices}"
        )
        precision, recall, _, accuracy, dice_score, cl_dice_score = (
            evaluate_model_with_postprocessing(
                model, testloader, testloader_ring_dir, device, with_wandb=False
            )
        )

        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_dice_score += dice_score
        total_cl_dice_score += cl_dice_score

    total_accuracy /= len(model_paths)
    total_precision /= len(model_paths)
    total_recall /= len(model_paths)
    total_dice_score /= len(model_paths)
    total_cl_dice_score /= len(model_paths)
    print(
        f"Average Precision: {total_precision:.4f}, Average Recall: {total_recall:.4f}, Average Accuracy: {total_accuracy:.4f}, Average Dice Score: {total_dice_score:.4f}, Average CL Dice Score: {total_cl_dice_score:.4f}"
    )
    return (
        total_precision,
        total_recall,
        total_accuracy,
        total_dice_score,
        cl_dice_score,
    )


def build_ensemble_model(path, device):
    """
    Builds and loads an ensemble model from a saved file.

    Args:
        ensemble_path (str): The path to the saved ensemble model file.
        device (str): The device to move the model to (e.g., "cuda:1").

    Returns:
        nn.Module: The loaded ensemble model on the specified device.
    """
    # Load the saved ensemble model directly
    ensemble_model = torch.load(path, map_location=device)

    # Move the ensemble model to the specified device
    ensemble_model = ensemble_model.to(device)

    return ensemble_model


def build_segmentation_model(
    encoder,
    architecture: Literal[
        "Unet",
        "UnetPlusPlus",
        "MAnet",
        "Linknet",
        "FPN",
        "PSPNet",
        "DeepLabV3Plus",
        "PAN",
    ] = "Unet",
    device="cuda",
    in_channels=1,
    classes=1,
    path=None,
):
    if architecture == "Unet":
        model = smp.Unet(
            encoder, encoder_weights=None, in_channels=in_channels, classes=classes
        )
    elif architecture == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder, encoder_weights=None, in_channels=in_channels, classes=classes
        )
    elif architecture == "MAnet":
        model = smp.FPN(
            encoder, encoder_weights=None, in_channels=in_channels, classes=classes
        )
    elif architecture == "Linknet":
        model = smp.Linknet(
            encoder, encoder_weights=None, in_channels=in_channels, classes=classes
        )
    elif architecture == "FPN":
        model = smp.FPN(
            encoder, encoder_weights=None, in_channels=in_channels, classes=classes
        )
    elif architecture == "PSPNet":
        model = smp.PSPNet(
            encoder, encoder_weights=None, in_channels=in_channels, classes=classes
        )
    elif architecture == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder, encoder_weights=None, in_channels=in_channels, classes=classes
        )
    elif architecture == "PAN":
        model = smp.PAN(
            encoder, encoder_weights=None, in_channels=in_channels, classes=classes
        )
    if path is not None:
        model = load_model(model, path, device)
    return model.to(device)


def build_criterion(
    loss: Literal["Dice", "BCE", "Focal", "CrossEntropy", "ClDice"] = "Dice",
    gamma=2,
    iter=3,
    alpha=0.5,
    smooth=1,
):
    if loss == "Dice":
        return smp.losses.DiceLoss(mode="binary")
    elif loss == "BCE":
        return torch.nn.BCEWithLogitsLoss()
    elif loss == "Focal":
        return smp.losses.FocalLoss(mode="binary", gamma=gamma)
    elif loss == "CrossEntropy":
        return torch.nn.CrossEntropyLoss()
    elif loss == "ClDice":

        class ClDiceWrapper(torch.nn.Module):
            def __init__(self, criterion):
                super().__init__()
                self.criterion = criterion

            def forward(self, outputs, labels):
                outputs = torch.sigmoid(outputs)  # Apply sigmoid first
                return self.criterion(labels, outputs)  # Swap labels and outputs

        return ClDiceWrapper(SoftDiceclDiceLoss(iter_=iter, alpha=alpha, smooth=smooth))


def build_optimizer(
    model, learning_rate=0.001, optimizer: Literal["Adam", "SGD"] = "Adam"
):
    if optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    return torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_sweep(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        encoder = config["encoder"]
        device = config["device"]
        architecture = config["architecture"]
        loss = config["loss"]
        learning_rate = config["learning_rate"]
        optimizer = config["optimizer"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]
        channels = config["channels"]
        proportion_augmented_data = config["proportion_augmented_data"]
        classes = config["classes"] if "classes" in config else 1
        iter = config["iter"] if "iter" in config else 3
        alpha = config["alpha"] if "alpha" in config else 0.5
        smooth = config["smooth"] if "smooth" in config else 1
        gamma = config["gamma"] if "gamma" in config else 2

        model = build_segmentation_model(
            encoder,
            architecture=architecture,
            device=device,
            in_channels=channels,
            classes=classes,
        )
        criterion = build_criterion(
            loss, gamma=gamma, iter=iter, alpha=alpha, smooth=smooth
        )
        optimizer = build_optimizer(
            model, learning_rate=learning_rate, optimizer=optimizer
        )
        trainloader, validationloader, testloader = build_FIVES_dataloaders(
            batch_size=batch_size,
            proportion_augmented_data=proportion_augmented_data,
            num_channels=channels,
            width=config["width"],
            height=config["height"],
        )

        train_losses, val_losses = [], []
        highest_dice = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            train_loss = 0.0
            for i, data in enumerate(trainloader):
                inputs, labels = data[0].to(device), data[1].to(device).float()
                outputs = model(inputs)
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_loss += loss.item()

                if (i + 1) % 10 == 0:  # Adjust the condition based on your preference
                    print(
                        f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}"
                    )
                    running_loss = 0.0  # Reset running loss after printing

            # Calculate and print the average loss per epoch
            train_loss = train_loss / len(trainloader)
            train_losses.append(train_loss)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "train/loss": train_loss}, step=epoch + 1)

            # Validation phase
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(validationloader):
                    inputs, labels = data[0].to(device), data[1].to(device).float()
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    if i == 0:
                        log_segmentation_example(model, data, device, epoch)

            val_loss = val_running_loss / len(validationloader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "validation/loss": val_loss}, step=epoch + 1)
            precision, _, _, _, dice_score = evaluate_model(
                model, testloader, device, with_wandb=False
            )
            wandb.log({"epoch": epoch + 1, "precision": precision}, step=epoch + 1)
            wandb.log({"epoch": epoch + 1, "dice_score": dice_score}, step=epoch + 1)

            if dice_score > highest_dice and dice_score > 0.8:
                highest_dice = dice_score
                torch.save(model.state_dict(), f"./models/FIVES_sweep/{run.name}.pth")

        del model
        torch.cuda.empty_cache()


def load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model
