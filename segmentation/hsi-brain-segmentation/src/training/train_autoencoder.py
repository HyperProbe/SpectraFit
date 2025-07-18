import wandb
import torch
import os
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from src.model.FADA.classifier import Classifier
from src.model.FADA.feature_extractor import FeatureExtractor
from src.model.FADA.segmentation_model import SegmentationWithChannelReducerFADA
from src.dataset.dataset import build_hsi_dataloader, build_hsi_testloader
from src.model.dimensionality_reduction.autoencoder import ConvAutoEncoder
from src.model.dimensionality_reduction.gaussian import GaussianAutoEncoder
from src.util.segmentation_util import build_segmentation_model, evaluate_model, load_model


def model_pipeline_autoencoder(
    config,
    project,
    epochs=10,
    device="cuda",
    batch_print=10,
    save_wandb=True,
):
    with wandb.init(project=project, config=config, name=config["model"]):
        config = wandb.config
        trainloader, validationloader = build_dataloaders(config)
        return init_model_and_train(
            trainloader,
            validationloader,
            config,
            epochs,
            device,
            batch_print,
            save_wandb,
        )


def init_model_and_train(
    trainloader,
    validationloader,
    config,
    epochs=10,
    device="cuda",
    batch_print=10,
    save_wandb=True,
):
    model, optimizer, criterion = init_training(config, device)

    train_loss, val_loss = train_and_validate_autoencoder(
        model=model,
        trainloader=trainloader,
        validationloader=validationloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        model_name=config.model,
        device=device,
        batch_print=batch_print,
        save_wandb=save_wandb,
    )
    return model, train_loss, val_loss


def init_training(config, device):
    mu = torch.tensor(config.mu, dtype=torch.float) if "mu" in config else None
    sigma = torch.tensor(config.sigma, dtype=torch.float) if "sigma" in config else None
    num_reduced_channels = config.out_channels if "out_channels" in config else 3
    model = GaussianAutoEncoder(
        num_input_channels=826,
        num_reduced_channels=num_reduced_channels,
        mu=mu,
        sigma=sigma,
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": config.lr_encoder},
            {"params": model.decoder.parameters(), "lr": config.lr_decoder},
        ]
    )

    criterion = nn.MSELoss()
    return model, optimizer, criterion


def train_and_validate_autoencoder(
    model,
    trainloader,
    validationloader,
    criterion,
    optimizer,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
    save_wandb=True,
):
    """
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=10)
        :param model_name: Model file name (default=None)
    Returns
        train_losses, val_losses: List of losses per epoch
    """
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs = data[0].to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loss += loss.item()

            if (i + 1) % batch_print == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / batch_print:.4f}"
                )
                running_loss = 0.0

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
                inputs = data[0].to(device)
                outputs = model(inputs)

                loss = criterion(outputs, inputs)
                val_running_loss += loss.item()

                if i == 0:
                    log_encoded_image(model, epoch, inputs)
            log_gaussians(model, epoch)
            log_dice_score_with_gaussian(model, validationloader, device, epoch)

        val_loss = val_running_loss / len(validationloader)
        val_losses.append(val_loss)

        if model_name:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), f"./models/{model_name}.pth")
                if save_wandb:
                    model_artifact = wandb.Artifact(f"{model_name}", type="model")
                    model_artifact.add_file(f"./models/{model_name}.pth")
                    wandb.log_artifact(model_artifact)

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "validation/loss": val_loss}, step=epoch + 1)

    return train_losses, val_losses


def log_encoded_image(model, epoch, inputs):
    encoded_output = model.encoder(inputs)[0].cpu().numpy().squeeze()
    # Normalize the output image to be in the range [0, 255]
    if encoded_output.shape[0] == 1:
        image = np.stack([encoded_output, encoded_output, encoded_output], axis=-1)
    elif encoded_output.shape[0] == 3:
        image = encoded_output.transpose(1, 2, 0)
    image_min = image.min()
    image_max = image.max()

    image = 255 * (image - image_min) / (image_max - image_min)
    image = image.astype(np.uint8)

    # Log the RGB image to WandB
    wandb.log(
        {"Validation Image": wandb.Image(image, caption=f"Epoch {epoch+1}")},
        step=epoch + 1,
    )


def log_gaussians(autoencoder, epoch):
    mu = autoencoder.encoder.mu.cpu()
    sigma = torch.exp(autoencoder.encoder.log_sigma).cpu()
    colors = ["r", "g", "b"]
    fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 6))
    channel_indices = torch.arange(826).float()

    for i in range(autoencoder.encoder.num_output_channels):
        # Compute normalized Gaussian
        w = torch.exp(-0.5 * ((channel_indices - mu[i]) / sigma[i]) ** 2)
        w = w / w.sum()  # Normalize

        # Plot on the same x-axis as the pixel spectra
        ax_spectrum.plot(
            channel_indices.numpy(),
            w.numpy(),
            label=f"G{i+1} (μ={mu[i].item():.2f}, σ={sigma[i].item():.2f})",
            color=colors[i % len(colors)],
        )
    ax_spectrum.set_title("Learned Gaussians")
    ax_spectrum.set_xlabel("Channel Indices")
    ax_spectrum.set_ylabel("Normalized Intensity")
    ax_spectrum.legend()

    wandb.log({"Gaussians": wandb.Image(fig_spectrum)}, step=epoch + 1)
    plt.close(fig_spectrum)


def log_dice_score_with_gaussian(autoencoder, testloader, device, epoch):
    gcr = autoencoder.encoder
    segmentation_model = build_segmentation_model(
        encoder="timm-regnetx_320",
        architecture="Linknet",
        device=device,
        in_channels=gcr.num_output_channels,
    )

    segmentation_model = load_model(
        segmentation_model, "models/serene-sweep-9.pth", device
    )

    feature_extractor = FeatureExtractor(segmentation_model).to(device)
    classifier = Classifier(segmentation_model).to(device)
    model = SegmentationWithChannelReducerFADA(gcr, feature_extractor, classifier).to(
        device
    )
    precision, _, _, _, dice_score = evaluate_model(
        model, testloader, device, with_wandb=False
    )
    wandb.log({"epoch": epoch + 1, "dice_score": dice_score}, step=epoch + 1)
    wandb.log({"epoch": epoch + 1, "precision": precision}, step=epoch + 1)

    del model
    torch.cuda.empty_cache()


def train_sweep(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        trainloader, validationloader = build_dataloaders(config)
        config["model"] = run.name
        model, _, _ = init_model_and_train(
            trainloader,
            validationloader,
            config,
            epochs=config.epochs if "epochs" in config else 10,
            device=config.device,
            batch_print=5,
            save_wandb=False,
        )
        del model
        if os.path.exists(f"./models/{config.model}.pth"):
            os.remove(f"./models/{config.model}.pth")
            print(f"Removed model {config.model}.pth")
        torch.cuda.empty_cache()


def build_dataloaders(config):
    trainloader = build_hsi_dataloader(
        batch_size=config.batch_size,
        train_split=1,
        val_split=0,
        test_split=0,
        exclude_labeled_data=True,
        augmented=True,
    )[0]

    validationloader = build_hsi_testloader()
    return trainloader, validationloader


def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return recon_loss + kl_div


def train_and_validate_variational_autoencoder(
    model,
    trainloader,
    validationloader,
    optimizer,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
):
    """
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param epochs: Number of epochs (default=10)
        :param model_name: Model file name (default=None)
    Returns
        train_losses, val_losses: List of losses per epoch
    """
    train_losses, val_losses = [], []
    min_val_loss = np.inf

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs = data[0].to(device)
            outputs, mu, logvar = model(inputs)
            optimizer.zero_grad()
            loss = vae_loss(outputs, inputs, mu, logvar)
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
                inputs = data[0].to(device)
                outputs, mu, logvar = model(inputs)

                loss = vae_loss(outputs, inputs, mu, logvar)
                val_running_loss += loss.item()

                if i == 0:
                    encoded_output = (
                        outputs.mean(dim=1, keepdim=True)[0, 0].cpu().numpy()
                    )  # Get the first image and the first channel
                    wandb.log(
                        {
                            "Validation Image": wandb.Image(
                                encoded_output, caption=f"Epoch {epoch+1}"
                            )
                        },
                        step=epoch + 1,
                    )

        if model_name:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), f"./models/{model_name}.pth")
                model_artifact = wandb.Artifact(f"{model_name}", type="model")
                model_artifact.add_file(f"./models/{model_name}.pth")
                wandb.log_artifact(model_artifact)

        val_loss = val_running_loss / len(validationloader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "validation/loss": val_loss}, step=epoch + 1)

    return train_losses, val_losses


def model_pipeline_variational_autoencoder(
    model,
    trainloader,
    validationloader,
    optimizer,
    config,
    project,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
):
    with wandb.init(project=project, config=config):
        config = wandb.config
        model = model.to(device)
        train_loss, val_loss = train_and_validate_variational_autoencoder(
            model,
            trainloader,
            validationloader,
            optimizer,
            epochs,
            model_name,
            device,
            batch_print,
        )
        return model, train_loss, val_loss


def build_conv_channel_reducer(
    num_input_channels=826, num_reduced_channels=3, load_from_path=None, device="cuda"
):
    model = ConvAutoEncoder(num_input_channels, num_reduced_channels).to(device)
    if load_from_path:
        model = load_model(model, load_from_path, device)
    return model.encoder


def build_and_store_gcr(
    mu, sigma, num_reduced_channels=3, device="cuda", save_path="./models/gcr.pth"
):
    mu = torch.tensor(mu, dtype=torch.float)
    sigma = torch.tensor(sigma, dtype=torch.float)
    model = GaussianAutoEncoder(826, num_reduced_channels, mu, sigma).to(device)
    torch.save(model.state_dict(), save_path)
