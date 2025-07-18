import torch.nn as nn

from .classifier import Classifier
from .feature_extractor import FeatureExtractor, FeatureExtractorWith1x1ConvReducer, FeatureExtractorWithCNN
from src.model.dimensionality_reduction.gaussian import build_gaussian_channel_reducer
from src.model.dimensionality_reduction.window_reducer import build_window_reducer, build_single_window_reducer
from src.util.segmentation_util import build_segmentation_model, load_model


class SegmentationModelFADA(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(SegmentationModelFADA, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


class SegmentationWithChannelReducerFADA(nn.Module):
    def __init__(self, channel_reducer, feature_extractor, classifier):
        super(SegmentationWithChannelReducerFADA, self).__init__()
        self.channel_reducer = channel_reducer
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        if x.shape[1] > 3:
            x = self.channel_reducer(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


def build_FADA_segmentation_model(
    architecture,
    encoder,
    in_channels,
    path,
    device,
    channel_reducer_path=None,
    windows_in_nm=None,
):
    segmentation_model = build_segmentation_model(
        architecture=architecture,
        encoder=encoder,
        device=device,
        in_channels=in_channels,
    )
    feature_extractor = FeatureExtractor(segmentation_model).to(device)
    classifier = Classifier(segmentation_model).to(device)
    if channel_reducer_path is not None:
        gcr = build_gaussian_channel_reducer(
            num_input_channels=826,
            num_reduced_channels=3,
            load_from_path=channel_reducer_path,
            device=device,
        )
        model = SegmentationWithChannelReducerFADA(
            gcr, feature_extractor, classifier
        ).to(device)
    elif windows_in_nm is not None:
        window_reducer = build_window_reducer(
            windows_in_nm=windows_in_nm, device=device
        )
        model = SegmentationWithChannelReducerFADA(
            window_reducer, feature_extractor, classifier
        ).to(device)
    else:
        model = SegmentationModelFADA(feature_extractor, classifier).to(device)

    model = load_model(model, path, device)
    return model


def build_FADA_segmentation_model_with_single_window_reducer(
    architecture,
    encoder,
    path,
    device,
    window_in_nm,
):
    segmentation_model = build_segmentation_model(
        architecture=architecture,
        encoder=encoder,
        device=device,
        in_channels=1,
    )
    feature_extractor = FeatureExtractor(segmentation_model).to(device)
    classifier = Classifier(segmentation_model).to(device)
    window_reducer = build_single_window_reducer(
        window_in_nm=window_in_nm, device=device
    )
    model = SegmentationModelFADA(feature_extractor, classifier).to(device)
    model = load_model(model, path, device)
    combined_model = nn.Sequential(window_reducer, model).to(device)
    return combined_model

def build_baseline_segmentation_model_with_window_reducer(
    encoder,
    architecture,
    windows_in_nm=[[600, 1000], [500, 600], [400, 500]],
    device="cuda",
    in_channels=1,
    classes=1,
    path=None,
):
    segmentation_model = build_segmentation_model(
        encoder, architecture, device, in_channels, classes, path
    )
    window_reducer = build_window_reducer(windows_in_nm=windows_in_nm, device=device)
    feature_extractor = FeatureExtractor(segmentation_model)
    classifier = Classifier(segmentation_model)
    return SegmentationWithChannelReducerFADA(
        window_reducer, feature_extractor, classifier
    ).to(device)
    
def build_FADA_segmentation_model_with_dimensionality_reduction(
    architecture,
    encoder,
    path,
    device,
    kernel_size=3,
    in_channels=1,
    feature_conv_reducer="cnn",
):
    segmentation_model = build_segmentation_model(
        architecture=architecture,
        encoder=encoder,
        device=device,
        in_channels=in_channels,
    )
    if feature_conv_reducer == "cnn":
        feature_extractor = FeatureExtractorWithCNN(
            segmentation_model,
            hyperspectral_channels=826,
            encoder_in_channels=in_channels,
            kernel_size=kernel_size,
        ).to(device)
    else:
        feature_extractor = FeatureExtractorWith1x1ConvReducer(
            segmentation_model,
            hyperspectral_channels=826,
            encoder_in_channels=in_channels,
        ).to(device)
    classifier = Classifier(segmentation_model).to(device)
    model = SegmentationModelFADA(feature_extractor, classifier).to(device)
    model = load_model(model, path, device)
    return model
