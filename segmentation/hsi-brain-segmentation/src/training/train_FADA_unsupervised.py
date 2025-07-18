import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from src.model.FADA.segmentation_model import (
    SegmentationModelFADA,
    SegmentationWithChannelReducerFADA,
)
from src.model.FADA.discriminator import PixelDiscriminator
from src.model.FADA.classifier import Classifier
from src.model.FADA.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorWith1x1ConvReducer,
    FeatureExtractorWithCNN,
    BaseFeatureExtractorWithDimReduction,
)
from src.dataset.dataset import (
    build_FIVES_random_crops_dataloaders,
    build_hsi_dataloader,
    build_hsi_testloader,
)
from src.model.dimensionality_reduction.gaussian import build_gaussian_channel_reducer
from src.model.dimensionality_reduction.window_reducer import build_window_reducer
from src.model.dimensionality_reduction.cycle_GAN import GeneratorF
from src.util.segmentation_util import (
    evaluate_model_with_postprocessing,
    load_model,
    log_segmentation_example,
    evaluate_model,
)
from src.util.segmentation_util import build_segmentation_model, build_criterion, build_optimizer
from segmentation_models_pytorch.encoders import get_encoder
from src.training.train_autoencoder import (
    build_conv_channel_reducer,
)
import os


def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float() * F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights * torch.sum(loss, dim=1))


def model_pipeline(
    config,
    project,
    device="cuda",
    batch_print=10,
    evaluate=True,
    with_overlays=False,
    save_wandb=True,
    save_path="models",
):
    with wandb.init(project=project, config=config, name=config["model"]):
        config = wandb.config
        (
            trainloader_source,
            validationloader_source,
            testloader_source,
            trainloader_target,
            testloader_target,
            testloader_target_rings,
        ) = initialize_data_loaders(config)

        return init_model_and_train(
            trainloader_source,
            validationloader_source,
            testloader_source,
            trainloader_target,
            testloader_target,
            testloader_target_rings,
            config,
            device,
            batch_print,
            evaluate,
            with_overlays,
            save_wandb=save_wandb,
            save_path=save_path,
        )


def init_model_and_train(
    trainloader_source,
    validationloader_source,
    testloader_source,
    trainloader_target,
    testloader_target,
    testloader_target_rings,
    config,
    device,
    batch_print,
    evaluate,
    with_overlays,
    save_wandb=True,
    save_path="models",
):
    num_classes = config.classes if "classes" in config else 1
    choose_indices = (
        config.choose_indices if "choose_indices" in config else [0, 1, 2, 3, 4]
    )
    segmentation_model = build_segmentation_model(
        config.encoder,
        config.architecture,
        device,
        in_channels=config.in_channels,
        classes=num_classes,
    )
    if "pretrained" in config:
        print(f"Loading pretrained model from {config.pretrained}")
        segmentation_model = load_model(segmentation_model, config.pretrained, device)

    reducer = None
    if "gaussian" in config:
        print("Using Gaussian Channel Reduction")
        reducer = build_gaussian_channel_reducer(
            num_input_channels=826,
            num_reduced_channels=config.in_channels,
            load_from_path=config.gaussian,
            device=device,
        )
    elif "conv_reducer" in config:
        print("Using Convolutional Channel Reduction")
        reducer = build_conv_channel_reducer(
            num_input_channels=826,
            num_reduced_channels=config.in_channels,
            load_from_path=config.conv_reducer,
            device=device,
        )
    elif "window_reducer" in config:
        print("Using Window Channel Reduction")
        reducer = build_window_reducer(config.window_reducer, device)

    feature_with_conv_reducer = (
        None if "feature_conv_reducer" not in config else config.feature_conv_reducer
    )

    if feature_with_conv_reducer:
        print(
            f"Using Feature Extractor with Convolutional Reducer: {feature_with_conv_reducer}"
        )
        freeze_encoder = (
            False if "freeze_encoder" not in config else config.freeze_encoder
        )
        if feature_with_conv_reducer == "cnn":
            feature_extractor = FeatureExtractorWithCNN(
                segmentation_model,
                hyperspectral_channels=826,
                freeze_encoder=freeze_encoder,
                encoder_in_channels=config.in_channels,
                kernel_size=config.kernel_size if "kernel_size" in config else 3,
            ).to(device)
        else:
            feature_extractor = FeatureExtractorWith1x1ConvReducer(
                segmentation_model,
                hyperspectral_channels=826,
                freeze_encoder=freeze_encoder,
                encoder_in_channels=config.in_channels,
            ).to(device)
    else:
        feature_extractor = FeatureExtractor(segmentation_model).to(device)

    classifier = Classifier(segmentation_model).to(device)
    discriminator = PixelDiscriminator(
        input_nc=get_encoder(config.encoder, config.in_channels).out_channels[-1],
        ndf=config.ndf,
        num_classes=num_classes,
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
    with_contrastive_loss = "contrastive_loss" in config and config.contrastive_loss
    penalize_rings_weight = (
        config.penalize_rings_weight if "penalize_rings_weight" in config else 0.25
    )
    cycle_loss_hyperparams = (
        config.cycle_loss_hyperparams if "cycle_loss_hyperparams" in config else None
    )

    train_loss, domain_loss, val_loss_source, val_loss_target = train_and_validate(
        feature_extractor,
        classifier,
        discriminator,
        trainloader_source,
        validationloader_source,
        testloader_source,
        trainloader_target,
        testloader_target,
        testloader_target_rings,
        criterion_segmentation,
        optimizer_fea,
        optimizer_cls,
        optimizer_dis,
        epochs=config.epochs,
        model_name=config.model,
        device=device,
        batch_print=batch_print,
        with_overlays=with_overlays,
        hsi_reducer=reducer,
        save_wandb=save_wandb,
        cycle_loss_hyperparams=cycle_loss_hyperparams,
        with_contrastive_loss=with_contrastive_loss,
        penalize_rings_weight=penalize_rings_weight,
        choose_indices=choose_indices,
        save_path=save_path,
    )
    if evaluate:
        if config.model:
            model = (
                SegmentationModelFADA(feature_extractor, classifier)
                if not reducer
                else SegmentationWithChannelReducerFADA(
                    reducer, feature_extractor, classifier
                )
            )
            model = load_model(
                model, os.path.join(save_path, f"{config.model}.pth"), device
            )
        evaluate_model(model, testloader_target, device)
        evaluate_model_with_postprocessing(
            model, testloader_target, testloader_target_rings, device
        )
    return model, train_loss, domain_loss, val_loss_source, val_loss_target


def train_and_validate(
    feature_extractor,
    classifier,
    discriminator,
    trainloader_source,
    validationloader_source,
    testloader_source,
    trainloader_target,
    testloader_target,
    testloader_target_rings,
    criterion_segmentation,
    optimizer_fea,
    optimizer_cls,
    optimizer_D,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
    with_overlays=False,
    hsi_reducer=None,
    save_wandb=True,
    with_contrastive_loss=False,
    cycle_loss_hyperparams=None,
    penalize_rings_weight=0.25,
    choose_indices=[0, 1, 2, 3, 4],
    save_path="models",
):
    train_losses, domain_losses, val_losses_source, val_losses_target = [], [], [], []
    highest_dice = 0

    # Create infinite iterators
    source_domain_iter = itertools.cycle(trainloader_source)
    target_domain_iter = itertools.cycle(trainloader_target)

    # Calculate total number of batches per epoch
    num_batches = max(
        len(trainloader_source),
        len(trainloader_target),
    )
    with_cycle_loss = cycle_loss_hyperparams is not None

    if with_cycle_loss:
        print("Initializing Cycle Loss Setup")
        generator_F, cycle_loss_fn, optimizer_F, optimizer_G = init_cycle_loss_setup(
            feature_extractor, device, cycle_loss_hyperparams
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
            target_domain_data = next(target_domain_iter)
            tgt_input = target_domain_data[0].to(device)

            src_size = src_input.shape[-2:]
            tgt_size = tgt_input.shape[-2:]

            # Forward pass through feature_extractor and classifier
            src_features = feature_extractor(src_input)
            src_pred = classifier(src_features)
            temperature = 1.8
            src_pred = src_pred.div(temperature)
            loss_seg = criterion_segmentation(src_pred, src_label)
            loss_seg.backward()

            # generate soft labels
            src_soft_label = F.softmax(src_pred, dim=1).detach().to(device)
            src_soft_label[src_soft_label > 0.9] = 0.9

            tgt_features = feature_extractor(
                hsi_reducer(tgt_input) if hsi_reducer else tgt_input
            )
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
            loss_adv_tgt.backward(retain_graph=True)

            if with_contrastive_loss:
                black_ring_mask = target_domain_data[1].to(device).float()
                tgt_pred_sigmoid = torch.sigmoid(tgt_pred)
                contrastive_loss = torch.mean(tgt_pred_sigmoid * black_ring_mask)
                weighted_contrastive_loss = penalize_rings_weight * contrastive_loss
                running_loss += weighted_contrastive_loss.item()
                weighted_contrastive_loss.backward()

            if with_cycle_loss:
                cycle_loss_backward_step(
                    feature_extractor,
                    cycle_loss_hyperparams,
                    generator_F,
                    cycle_loss_fn,
                    optimizer_F,
                    optimizer_G,
                    src_input,
                    tgt_input,
                )

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
        if hsi_reducer:
            model = SegmentationWithChannelReducerFADA(
                hsi_reducer, feature_extractor, classifier
            ).to(device)
        else:
            model = SegmentationModelFADA(feature_extractor, classifier).to(device)

        model.eval()
        val_running_loss_target = 0.0
        val_running_loss_source = 0.0
        with torch.no_grad():
            # Validation Phase on target data
            for (i, data), (_, ring_data) in zip(
                enumerate(testloader_target), enumerate(testloader_target_rings)
            ):
                inputs, labels = data[0].to(device), data[1].to(device).float()
                outputs = model(inputs)
                loss = criterion_segmentation(outputs, labels)
                val_running_loss_target += loss.item()
                log_segmentation_example(
                    model,
                    data,
                    device,
                    epoch,
                    title=f"Validation Overlay HSI {choose_indices[i]}",
                    channel_reducer=hsi_reducer,
                )
                log_segmentation_example(
                    model,
                    data,
                    device,
                    epoch,
                    title=f"Validation Overlay With Postprocessing {choose_indices[i]}",
                    channel_reducer=hsi_reducer,
                    ring_data=ring_data[1][0].to(device).float(),
                    with_postprocessing=True,
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
        precision_target_postprocessed, _, _, _, dice_score_postprocessed, _ = (
            evaluate_model_with_postprocessing(
                model,
                testloader_target,
                testloader_target_rings,
                device,
                with_wandb=False,
            )
        )
        if model_name:
            if dice_score_postprocessed > highest_dice:
                highest_dice = dice_score_postprocessed
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                model_path = os.path.join(save_path, f"{model_name}.pth")
                torch.save(model.state_dict(), model_path)
                if save_wandb:
                    model_artifact = wandb.Artifact(f"{model_name}", type="model")
                    model_artifact.add_file(model_path)
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
        wandb.log(
            {
                "epoch": epoch + 1,
                "precision/target_postprocessed": precision_target_postprocessed,
            },
            step=epoch + 1,
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "dice_score/target_postprocessed": dice_score_postprocessed,
            },
            step=epoch + 1,
        )

        del model

    return train_losses, domain_losses, val_losses_source, val_losses_target


def cycle_loss_backward_step(
    feature_extractor,
    cycle_loss_hyperparams,
    generator_F,
    cycle_loss_fn,
    optimizer_F,
    optimizer_G,
    src_input,
    tgt_input,
):
    if isinstance(feature_extractor, BaseFeatureExtractorWithDimReduction):
        # F(G(x)) -> x
        fives_like = feature_extractor.forward_transform(tgt_input)
        cycle_loss_F = cycle_loss_hyperparams["lambda_F"] * cycle_loss_fn(
            generator_F(fives_like), tgt_input
        )

        # G(F(y)) -> y
        hsi_from_fives = generator_F(src_input)
        cycle_loss_G = cycle_loss_hyperparams["lambda_G"] * cycle_loss_fn(
            feature_extractor.forward_transform(hsi_from_fives), src_input
        )
        optimizer_G.zero_grad()
        optimizer_F.zero_grad()

        total_cycle_loss = cycle_loss_F + cycle_loss_G
        total_cycle_loss.backward()
        optimizer_F.step()
        optimizer_G.step()


def init_cycle_loss_setup(feature_extractor, device, cycle_loss_hyperparams):
    cycle_loss = nn.L1Loss()
    generator_F = GeneratorF(
        in_channels=feature_extractor.expected_channels,
        out_channels=feature_extractor.hyperspectral_channels,
        num_filters=64,
    ).to(device)
    optimizer_F = torch.optim.Adam(
        generator_F.parameters(), lr=cycle_loss_hyperparams["lr_F"]
    )
    if isinstance(feature_extractor, FeatureExtractorWithCNN):
        optimizer_G = torch.optim.Adam(
            list(feature_extractor.dim_reduction.parameters())
            + list(feature_extractor.cnn_transform.parameters()),
            lr=cycle_loss_hyperparams["lr_G"],
        )
    else:
        optimizer_G = torch.optim.Adam(
            list(feature_extractor.dim_reduction.parameters()),
            lr=cycle_loss_hyperparams["lr_G"],
        )

    return (generator_F, cycle_loss, optimizer_F, optimizer_G)


def black_ring_loss_only(tgt_pred, black_ring_mask):
    """
    tgt_pred: logits of shape [N, 3, H, W]
    black_ring_mask: binary mask of shape [N, H, W]
                     (1 = black ring pixel, 0 = unlabeled/other)
    Returns BCE loss for black ring predictions only.
    """

    # Extract the logits for the black-ring channel (assuming index=2).
    # Shape: [N, H, W]
    black_ring_logits = tgt_pred[:, 2, :, :]
    pred_black_ring = black_ring_logits[black_ring_mask == 1]
    pos_weight = torch.tensor([100]).to(black_ring_logits.device)
    print(pred_black_ring)

    # We can apply binary cross-entropy to these logits
    # versus the ground truth black_ring_mask.
    # 'F.binary_cross_entropy_with_logits' is typically used for single-class segmentation.
    loss = F.binary_cross_entropy_with_logits(
        black_ring_logits, black_ring_mask.float(), pos_weight=pos_weight
    )
    return loss


def train_sweep(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        config["model"] = run.name

        api = wandb.Api()
        sweep = api.sweep(run.sweep_id)
        sweep_name = sweep.name
        save_path = os.path.join("./models", sweep_name)
        os.makedirs(save_path, exist_ok=True)

        (
            trainloader_source,
            validationloader_source,
            testloader_source,
            trainloader_target,
            testloader_target,
            testloader_target_rings,
        ) = initialize_data_loaders(config)
        model, _, _, _, _ = init_model_and_train(
            trainloader_source,
            validationloader_source,
            testloader_source,
            trainloader_target,
            testloader_target,
            testloader_target_rings,
            config,
            device=config.device,
            batch_print=10,
            evaluate=True,
            with_overlays=True,
            save_wandb=False,
            save_path=save_path,
        )
        _, _, _, _, dice_score, _ = evaluate_model_with_postprocessing(
            model,
            testloader_target,
            testloader_target_rings,
            config.device,
            with_wandb=False,
        )

        api = wandb.Api()
        sweep = api.sweep(run.sweep_id)

        best_run = sweep.best_run(order="test/dice_score_postprocessed")
        current_best_dice = best_run.summary.get("best_dice", float("-inf"))

        if dice_score > current_best_dice:
            print(f"New best dice score: {dice_score} found in run {run.name}")
            previous_best_model = best_run.summary.get("best_model_name", None)
            print("Previous best model: ", previous_best_model)
            if previous_best_model:
                previous_best_model_path = os.path.join(
                    save_path, f"{previous_best_model}.pth"
                )
                if os.path.exists(previous_best_model_path):
                    os.remove(previous_best_model_path)
                    print(
                        f"Removed model previous best model {previous_best_model}.pth"
                    )

            run.summary["best_dice"] = dice_score
            run.summary["best_model_name"] = config.model
        else:
            current_model_path = os.path.join(save_path, f"{config.model}.pth")
            if os.path.exists(current_model_path):
                os.remove(current_model_path)
                print(f"Removed non-best model {config.model}.pth")
        torch.cuda.empty_cache()
        del model


def initialize_data_loaders(config):
    rgb = config.rgb if "rgb" in config else False
    rgb_channels = config.rgb_channels if "rgb_channels" in config else (425, 192, 109)
    num_classes = config.classes if "classes" in config else 1
    choose_indices = (
        config.choose_indices if "choose_indices" in config else [0, 1, 2, 3, 4]
    )
    trainloader_source, validationloader_source, testloader_source = (
        build_FIVES_random_crops_dataloaders(
            batch_size=config.batch_size_source,
            num_channels=config.in_channels,
            load_from_path=config.dataset_path,
            classes=num_classes,
        )
    )
    window = config.window if "window" in config else None
    trainloader_target = build_hsi_dataloader(
        batch_size=config.batch_size_target,
        train_split=1,
        val_split=0,
        test_split=0,
        window=window,
        exclude_labeled_data=True,
        augmented=config.augmented,
        rgb=rgb,
        rgb_channels=rgb_channels,
        ring_label_dir="data/helicoid_ring_labels",
        classes=num_classes,
    )[0]

    testloader_target, testloader_target_rings = build_hsi_testloader(
        batch_size=1,
        window=window,
        rgb=rgb,
        rgb_channels=rgb_channels,
        classes=num_classes,
        ring_label_dir="data/helicoid_ring_labels",
        choose_indices=choose_indices,
    )

    return (
        trainloader_source,
        validationloader_source,
        testloader_source,
        trainloader_target,
        testloader_target,
        testloader_target_rings,
    )
