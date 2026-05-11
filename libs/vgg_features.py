import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


VGG19_RELU_LAYERS = {
    "relu1_1": 1,
    "relu1_2": 3,
    "relu2_1": 6,
    "relu2_2": 8,
    "relu3_1": 11,
    "relu3_2": 13,
    "relu3_3": 15,
    "relu3_4": 17,
    "relu4_1": 20,
    "relu4_2": 22,
    "relu4_3": 24,
    "relu4_4": 26,
    "relu5_1": 29,
    "relu5_2": 31,
    "relu5_3": 33,
    "relu5_4": 35,
}


class VGG19FeatureExtractor(nn.Module):
    """VGG19 feature extractor that returns selected intermediate layers."""

    def __init__(self, vgg_features, output_layers):
        super().__init__()
        unknown = sorted(set(output_layers) - set(VGG19_RELU_LAYERS))
        if unknown:
            raise ValueError(f"Unknown VGG19 layer name(s): {unknown}")

        self.output_layers = set(output_layers)
        self.index_to_name = {
            index: name
            for name, index in VGG19_RELU_LAYERS.items()
            if name in self.output_layers
        }
        max_index = max(self.index_to_name)

        layers = []
        for layer in list(vgg_features.children())[: max_index + 1]:
            if isinstance(layer, nn.ReLU):
                layers.append(nn.ReLU(inplace=False))
            else:
                layers.append(layer)
        self.features = nn.Sequential(*layers)

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, x):
        outputs = {}
        for index, layer in enumerate(self.features):
            x = layer(x)
            name = self.index_to_name.get(index)
            if name is not None:
                outputs[name] = x
        return outputs


def build_vgg19(output_layers, device, weights_path=None, pretrained=True):
    """Build a VGG19 feature extractor and optionally load local weights."""
    cache_dir = Path("models/torch_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache_dir))

    vgg = _create_vgg19(pretrained=pretrained and weights_path is None)

    if weights_path is not None:
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        vgg.load_state_dict(state)

    extractor = VGG19FeatureExtractor(vgg.features, output_layers)
    return extractor.to(device)


def _create_vgg19(pretrained):
    try:
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        return models.vgg19(weights=weights)
    except AttributeError:
        return models.vgg19(pretrained=pretrained)
