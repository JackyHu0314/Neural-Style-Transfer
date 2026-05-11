import torch
import torch.nn.functional as F

from libs.image_utils import normalize_batch


def gram_matrix(features):
    """Calculate channel correlation matrix for BCHW features."""
    batch, channels, height, width = features.shape
    flat = features.view(batch, channels, height * width)
    gram = torch.bmm(flat, flat.transpose(1, 2))
    return gram / float(channels * height * width)


class StyleTransferObjective:
    """Content and style loss computed from a fixed VGG19 extractor."""

    def __init__(
        self,
        extractor,
        content_image,
        style_image,
        content_layer,
        style_layers,
        content_weight,
        style_weight,
        style_layer_weights=None,
    ):
        self.extractor = extractor
        self.content_layer = content_layer
        self.style_layers = list(style_layers)
        self.content_weight = content_weight
        self.style_weight = style_weight

        if style_layer_weights is None:
            each = 1.0 / len(self.style_layers)
            self.style_layer_weights = {layer: each for layer in self.style_layers}
        else:
            self.style_layer_weights = style_layer_weights

        with torch.no_grad():
            content_features = self.extractor(normalize_batch(content_image))
            style_features = self.extractor(normalize_batch(style_image))

        self.content_target = content_features[self.content_layer].detach()
        self.style_targets = {
            layer: gram_matrix(style_features[layer]).detach()
            for layer in self.style_layers
        }

    def __call__(self, generated_image):
        generated_features = self.extractor(normalize_batch(generated_image))

        content_loss = F.mse_loss(
            generated_features[self.content_layer],
            self.content_target,
        )

        style_loss = generated_image.new_tensor(0.0)
        for layer in self.style_layers:
            generated_gram = gram_matrix(generated_features[layer])
            layer_loss = F.mse_loss(generated_gram, self.style_targets[layer])
            style_loss = style_loss + self.style_layer_weights[layer] * layer_loss

        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        return total_loss, content_loss.detach(), style_loss.detach()

