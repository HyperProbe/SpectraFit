import wandb
import torch.nn.functional as F
from src.util.segmentation_util import (
    log_segmentation_example,
    evaluate_model,
    build_segmentation_model,
    build_criterion,
    build_optimizer,
)
import torch
import torch.nn as nn
import itertools
from src.model.HSI_models.HSI_Net import DomainDiscriminatorFC, ModelWithDomainAdaptation
from segmentation_models_pytorch.encoders import get_encoder


def model_pipeline(
    trainloader_source,
    validationloader_source,
    testloader_source,
    trainloader_target,
    testloader_target,
    config,
    project,
    device="cuda",
    batch_print=10,
    evaluate=True,
    with_overlays=False,
):
    with wandb.init(project=project, config=config, name=config["model"]):
        config = wandb.config
        segmentation_model = build_segmentation_model(
            config.encoder, config.architecture, device, in_channels=config.in_channels
        )
        if "pretrained" in config:
            print(f"Loading pretrained model from {config.pretrained}")
            segmentation_model.load_state_dict(torch.load(config.pretrained))
        domain_discriminator = DomainDiscriminatorFC(
            in_channels_list=get_encoder(
                config.encoder, in_channels=config.in_channels
            ).out_channels,
            hidden_dim=config.hidden_dim,
            num_domains=2,
        ).to(device)
        model = ModelWithDomainAdaptation(
            segmentation_model, config.lambda_param, domain_discriminator
        ).to(device)
        criterion_segmentation = build_criterion(config.loss)
        optimizer = build_optimizer(
            model, learning_rate=config.learning_rate, optimizer=config.optimizer
        )
        train_loss, domain_loss, val_loss_source, val_loss_target = train_and_validate(
            model,
            trainloader_source,
            validationloader_source,
            testloader_source,
            trainloader_target,
            testloader_target,
            criterion_segmentation,
            optimizer,
            epochs=config.epochs,
            model_name=config.model,
            device=device,
            batch_print=batch_print,
            with_overlays=with_overlays,
        )
        if evaluate:
            if config.model:
                model.load_state_dict(torch.load(f"./models/{config.model}.pth"))
            evaluate_model(model, testloader_target, device)
        return model, train_loss, domain_loss, val_loss_source, val_loss_target


def train_and_validate(
    model,
    trainloader_source,
    validationloader_source,
    testloader_source,
    trainloader_target,
    testloader_target,
    criterion_segmentation,
    optimizer,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
    with_overlays=False,
):
    train_losses, domain_losses, val_losses_source, val_losses_target = [], [], [], []
    highest_dice = 0
    wandb.watch(model, criterion_segmentation, log="all", log_freq=10)

    # Loss function for domain classification
    criterion_domain_classifier = nn.CrossEntropyLoss()

    # Create infinite iterators
    source_domain_iter = itertools.cycle(trainloader_source)
    target_domain_iter = itertools.cycle(trainloader_target)

    # Calculate total number of batches per epoch
    num_batches = max(
        len(trainloader_source),
        len(trainloader_target),
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loss = 0.0
        domain_loss = 0.0

        for batch_idx in range(num_batches):
            # Segmentation Training Step
            seg_data = next(source_domain_iter)
            inputs_source, labels_source = (
                seg_data[0].to(device),
                seg_data[1].to(device).float(),
            )
            # Get target domain data for the current window
            target_domain_data = next(target_domain_iter)
            inputs_target = target_domain_data[0].to(device)

            # Forward pass through encoder and decoder
            segmentation_output_source, domain_output_source = model(inputs_source)
            _, domain_output_target = model(inputs_target)

            # Create domain labels: source=0 , target=1
            domain_labels_source = torch.ones(
                inputs_source.size(0), dtype=torch.long
            ).to(device)
            domain_labels_target = torch.zeros(
                inputs_target.size(0), dtype=torch.long
            ).to(device)

            # Compute segmentation loss
            loss_segmentation = criterion_segmentation(
                segmentation_output_source, labels_source
            )
            # Compute domain classification loss
            loss_domain_source = criterion_domain_classifier(
                domain_output_source, domain_labels_source
            )
            loss_domain_target = criterion_domain_classifier(
                domain_output_target, domain_labels_target
            )
            loss_domain = loss_domain_source + loss_domain_target

            # Total loss
            loss_total = loss_segmentation + loss_domain

            # Backward pass and optimization
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # Update running losses
            running_loss += loss_total.item()
            train_loss += loss_total.item()
            domain_loss += loss_domain.item()

            if (batch_idx + 1) % batch_print == 0:
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Total Loss: {running_loss/batch_print:.4f}, Segmentation Loss: {loss_segmentation.item():.4f}, Domain Loss Source: {loss_domain_source.item():.4f}, Domain Loss Target: {loss_domain_target.item():.4f}"
                )
                running_loss = 0.0  # Reset running loss after printing

        # Calculate and print the average loss per epoch
        train_loss = train_loss / num_batches
        domain_loss = domain_loss / num_batches
        train_losses.append(train_loss)
        domain_losses.append(domain_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train/loss": train_loss}, step=epoch + 1)
        wandb.log(
            {"epoch": epoch + 1, "train/domain_loss": domain_loss}, step=epoch + 1
        )

        # Validation phase
        model.eval()
        val_running_loss_target = 0.0
        val_running_loss_source = 0.0
        with torch.no_grad():
            # Validation Phase on target data
            for i, data in enumerate(testloader_target):
                inputs, labels = data[0].to(device), data[1].to(device).float()
                outputs = model(inputs)
                loss = criterion_segmentation(outputs, labels)
                val_running_loss_target += loss.item()
                log_segmentation_example(
                    model, data, device, epoch, title=f"Validation Overlay HSI {i}"
                )

            # validation phase on source data
            for i, data in enumerate(validationloader_source):
                inputs, labels = data[0].to(device), data[1].to(device).float()
                outputs = model(inputs)
                loss = criterion_segmentation(outputs, labels)
                val_running_loss_source += loss.item()
                if with_overlays and i == 0:
                    log_segmentation_example(
                        model, data, device, epoch, title="Validation Overlay FIVES"
                    )
        val_loss_source = val_running_loss_source / len(validationloader_source)
        val_loss_target = val_running_loss_target / len(testloader_target)
        wandb.log(
            {"epoch": epoch + 1, "val/loss_source": val_loss_source}, step=epoch + 1
        )
        wandb.log(
            {"epoch": epoch + 1, "val/loss_target": val_loss_target}, step=epoch + 1
        )

        val_losses_source.append(val_loss_source)
        val_losses_target.append(val_loss_target)

        # Evaluate model performance
        print("Evaluating model performance on source data")
        precision_source, _, _, _, dice_score_source = evaluate_model(
            model, testloader_source, device, with_wandb=False
        )

        print("Evaluating model performance on target data")
        precision_target, _, _, _, dice_score_target = evaluate_model(
            model, testloader_target, device, with_wandb=False
        )
        if model_name:
            if dice_score_target > highest_dice:
                highest_dice = dice_score_target
                torch.save(model.state_dict(), f"./models/{model_name}.pth")
                model_artifact = wandb.Artifact(f"{model_name}", type="model")
                model_artifact.add_file(f"./models/{model_name}.pth")
                wandb.log_artifact(model_artifact)

        print(
            f"Epoch {epoch+1}, Validation Loss Source: {val_loss_source:.4f}, Validation Loss Target: {val_loss_target:.4f}"
        )
        wandb.log(
            {"epoch": epoch + 1, "precision/source": precision_source}, step=epoch + 1
        )
        wandb.log(
            {"epoch": epoch + 1, "dice_score/source": dice_score_source}, step=epoch + 1
        )
        wandb.log(
            {"epoch": epoch + 1, "precision/target": precision_target}, step=epoch + 1
        )
        wandb.log(
            {"epoch": epoch + 1, "dice_score/target": dice_score_target}, step=epoch + 1
        )

    return train_losses, domain_losses, val_losses_source, val_losses_target
