import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from src.model.FADA.segmentation_model import SegmentationModelFADA, build_FADA_segmentation_model
from src.model.FADA.discriminator import PixelDiscriminator
from src.model.FADA.classifier import Classifier
from src.model.FADA.feature_extractor import FeatureExtractor
from src.dataset.dataset import (
    HSIDataset,
    build_FIVES_random_crops_dataloaders,
    build_hsi_dataloader,
)
from src.util.segmentation_util import log_segmentation_example, evaluate_model
from src.util.segmentation_util import build_segmentation_model, build_criterion, build_optimizer
from segmentation_models_pytorch.encoders import get_encoder
from torch.utils.data import DataLoader, Subset


def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float() * F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights * torch.sum(loss, dim=1))


def model_pipeline(
    trainloader_source,
    validationloader_source,
    testloader_source,
    trainloader_target_labeled,
    trainloader_target_unlabeled,
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
        return init_model_and_train(
            trainloader_source,
            validationloader_source,
            testloader_source,
            trainloader_target_labeled,
            trainloader_target_unlabeled,
            testloader_target,
            config,
            device,
            batch_print,
            evaluate,
            with_overlays,
        )


def init_model_and_train(
    trainloader_source,
    validationloader_source,
    testloader_source,
    trainloader_target_labeled,
    trainloader_target_unlabeled,
    testloader_target,
    config,
    device,
    batch_print,
    evaluate,
    with_overlays,
):
    segmentation_model = build_segmentation_model(
        config.encoder, config.architecture, device
    )
    if "pretrained" in config:
        print(f"Loading pretrained model from {config.pretrained}")
        segmentation_model.load_state_dict(torch.load(config.pretrained))
    feature_extractor = FeatureExtractor(segmentation_model).to(device)
    classifier = Classifier(segmentation_model).to(device)
    discriminator = PixelDiscriminator(
        input_nc=get_encoder(config.encoder, config.in_channels).out_channels[-1],
        ndf=config.ndf,
        num_classes=1,
    ).to(device)

    criterion_segmentation = build_criterion(config.seg_loss)
    optimizer_fea = build_optimizer(
        feature_extractor,
        learning_rate=config.learning_rate_fea,
        optimizer=config.optimizer,
    )
    optimizer_cls = build_optimizer(
        classifier,
        learning_rate=config.learning_rate_cls,
        optimizer=config.optimizer,
    )
    optimizer_dis = build_optimizer(
        discriminator,
        learning_rate=config.learning_rate_dis,
        optimizer=config.optimizer,
    )

    train_loss, domain_loss, val_loss_source, val_loss_target = train_and_validate(
        feature_extractor,
        classifier,
        discriminator,
        trainloader_source,
        validationloader_source,
        testloader_source,
        trainloader_target_labeled,
        trainloader_target_unlabeled,
        testloader_target,
        criterion_segmentation,
        optimizer_fea,
        optimizer_cls,
        optimizer_dis,
        epochs=config.epochs,
        model_name=config.model,
        train_indices=config.train_indices,
        device=device,
        batch_print=batch_print,
        with_overlays=with_overlays,
    )
    if evaluate:
        if config.model:
            model = SegmentationModelFADA(feature_extractor, classifier)
            model.load_state_dict(torch.load(f"./models/{config.model}.pth"))

        evaluate_model(model, testloader_target, device)
        print("compare performance with unsupervised Approach")
        unsupervised_model = get_unsupervised_model(device)
        precision_unsupervised, _, _, _, dice_score_unsupervised= evaluate_model(unsupervised_model, testloader_target, device, with_wandb=False)
        wandb.log({"test/precision_unsupervised": precision_unsupervised})
        wandb.log({"test/dice_score_unsupervised": dice_score_unsupervised})
        del unsupervised_model
        
    return model, train_loss, domain_loss, val_loss_source, val_loss_target


def get_unsupervised_model(device):
    return build_FADA_segmentation_model(
        architecture="Linknet",
        encoder="timm-regnetx_320",
        in_channels=1,
        path="models/FADA-Linknet-timm-regnetx_320-window_500-600_pretrained-augmented_target-random_crops_bloodvessel_ratio01-unsupervised.pth",
        device=device,
    )


def train_and_validate(
    feature_extractor,
    classifier,
    discriminator,
    trainloader_source,
    validationloader_source,
    testloader_source,
    trainloader_target_labeled,
    trainloader_target_unlabeled,
    testloader_target,
    criterion_segmentation,
    optimizer_fea,
    optimizer_cls,
    optimizer_D,
    train_indices,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
    with_overlays=False,
):
    train_losses, domain_losses, val_losses_source, val_losses_target = [], [], [], []
    highest_dice = 0

    # Create infinite iterators
    source_domain_iter = itertools.cycle(trainloader_source)
    target_domain_iter_unlabeled = itertools.cycle(trainloader_target_unlabeled)
    target_domain_iter_labeled = itertools.cycle(trainloader_target_labeled)

    # Calculate total number of batches per epoch
    num_batches = max(
        len(trainloader_source),
        len(trainloader_target_unlabeled),
    )

    for epoch in range(epochs):
        feature_extractor.train()
        classifier.train()
        discriminator.train()

        running_loss = 0.0
        train_loss = 0.0
        domain_loss = 0.0

        for batch_idx in range(num_batches):
            optimizer_fea.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_D.zero_grad()
            # Segmentation Training Step
            seg_data = next(source_domain_iter)
            src_input, src_label = (
                seg_data[0].to(device),
                seg_data[1].to(device).float(),
            )
            # Get target domain data for the current window
            target_domain_data_unlabeled = next(target_domain_iter_unlabeled)
            target_domain_data_labeled = next(target_domain_iter_labeled)
            tgt_input_labeled, tgt_label = (
                target_domain_data_labeled[0].to(device),
                target_domain_data_labeled[1].to(device).float(),
            )
            tgt_input_unlabeled = target_domain_data_unlabeled[0].to(device)

            src_size = src_input.shape[-2:]
            tgt_size = tgt_input_unlabeled.shape[-2:]

            # Forward pass through feature_extractor and classifier
            src_features = feature_extractor(src_input)
            src_pred = classifier(src_features)
            temperature = 1.8
            src_pred = src_pred.div(temperature)

            tgt_features_labeled = feature_extractor(tgt_input_labeled)
            tgt_pred_labeled = classifier(tgt_features_labeled)
            tgt_pred_labeled = tgt_pred_labeled.div(temperature)

            combined_pred = torch.cat((src_pred, tgt_pred_labeled), dim=0)
            combined_label = torch.cat((src_label, tgt_label), dim=0)

            loss_seg = criterion_segmentation(combined_pred, combined_label)
            loss_seg.backward()

            # generate soft labels
            src_soft_label = F.softmax(src_pred, dim=1).detach().to(device)
            src_soft_label[src_soft_label > 0.9] = 0.9

            tgt_features = feature_extractor(tgt_input_unlabeled)
            tgt_pred = classifier(tgt_features)
            tgt_pred = tgt_pred.div(temperature)
            tgt_soft_label = F.softmax(tgt_pred, dim=1).detach().to(device)
            tgt_soft_label[tgt_soft_label > 0.9] = 0.9

            tgt_D_pred = discriminator(tgt_features[-1], tgt_size)
            loss_adv_tgt = 0.001 * soft_label_cross_entropy(
                tgt_D_pred,
                torch.cat(
                    (tgt_soft_label, torch.zeros_like(tgt_soft_label).to(device)), dim=1
                ),
            )
            loss_adv_tgt.backward()
            optimizer_fea.step()
            optimizer_cls.step()

            optimizer_D.zero_grad()
            src_D_pred = discriminator(src_features[-1].detach().to(device), src_size)
            loss_D_src = 0.5 * soft_label_cross_entropy(
                src_D_pred,
                torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1),
            )
            loss_D_src.backward()

            tgt_D_pred = discriminator(tgt_features[-1].detach().to(device), tgt_size)
            loss_D_tgt = 0.5 * soft_label_cross_entropy(
                tgt_D_pred,
                torch.cat(
                    (torch.zeros_like(tgt_soft_label).to(device), tgt_soft_label), dim=1
                ),
            )
            loss_D_tgt.backward()

            optimizer_D.step()
            # Update running losses
            running_loss += (
                loss_seg.item()
                + loss_D_src.item()
                + loss_D_tgt.item()
                + loss_adv_tgt.item()
            )
            train_loss += running_loss
            domain_loss += loss_D_src.item() + loss_D_tgt.item() + loss_adv_tgt.item()

            if (batch_idx + 1) % batch_print == 0:
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Total Loss: {running_loss/batch_print:.4f}, Segmentation Loss: {loss_seg.item():.4f}, Domain Loss Source: {loss_D_src.item():.4f}, Domain Loss Target: {loss_D_tgt.item():.4f}, Adversarial Loss Target: {loss_adv_tgt.item():.4f}"
                )
                running_loss = 0.0  # Reset running loss after printing

        train_loss = train_loss / num_batches
        domain_loss = domain_loss / num_batches
        train_losses.append(train_loss)
        domain_losses.append(domain_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train/loss": train_loss}, step=epoch + 1)
        wandb.log(
            {"epoch": epoch + 1, "train/domain_loss": domain_loss}, step=epoch + 1
        )

        # Validation
        model = SegmentationModelFADA(feature_extractor, classifier).to(device)
        model.eval()
        val_running_loss_target = 0.0
        val_running_loss_source = 0.0
        val_indices = list({0, 1, 2, 3, 4} - set(train_indices))
        with torch.no_grad():
            # Validation Phase on target data
            for i, data in enumerate(testloader_target):
                inputs, labels = data[0].to(device), data[1].to(device).float()
                outputs = model(inputs)
                loss = criterion_segmentation(outputs, labels)
                val_running_loss_target += loss.item()
                log_segmentation_example(
                    model,
                    data,
                    device,
                    epoch,
                    title=f"Validation Overlay HSI {val_indices[i]}",
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

        del model

    return train_losses, domain_losses, val_losses_source, val_losses_target


def train_sweep(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        trainloader_source, validationloader_source, testloader_source = (
            build_FIVES_random_crops_dataloaders(
                batch_size=config.batch_size_source,
                num_channels=config.in_channels,
                load_from_path=config.dataset_path,
            )
        )
        window = config.window
        trainloader_target_unlabeled = build_hsi_dataloader(
            batch_size=config.batch_size_target,
            train_split=1,
            val_split=0,
            test_split=0,
            window=window,
            exclude_labeled_data=True,
            augmented=config.augmented,
        )[0]
        
        path = "./data/helicoid_with_labels"
        testset = HSIDataset(path, with_gt=True, window=window)
        testset.crop_dataset()

        trainloader_target_labeled = DataLoader(
            Subset(testset, config.train_indices),
            batch_size=1,
            shuffle=True,
        )
        testloader_target = DataLoader(
            Subset(testset, list({0, 1, 2, 3, 4} - set(config.train_indices))),
            batch_size=1,
            shuffle=False,
        )

        config["model"] = run.name
        model, _, _, _, _ = init_model_and_train(
            trainloader_source=trainloader_source,
            validationloader_source=validationloader_source,
            testloader_source=testloader_source,
            trainloader_target_labeled=trainloader_target_labeled,
            trainloader_target_unlabeled=trainloader_target_unlabeled,
            testloader_target=testloader_target,
            config=config,
            device=config.device,
            batch_print=10,
            evaluate=True,
            with_overlays=True,
        )
        del model
        torch.cuda.empty_cache()
