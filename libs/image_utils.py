from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path, device, image_size=None, max_size=None):
    """Load an RGB image as a BCHW float tensor in [0, 1]."""
    image = Image.open(path).convert("RGB")

    if image_size is not None:
        image = image.resize((image_size, image_size), Image.LANCZOS)
    elif max_size is not None and max(image.size) > max_size:
        width, height = image.size
        scale = max_size / float(max(width, height))
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    tensor = transforms.ToTensor()(image).unsqueeze(0)
    return tensor.to(device)


def normalize_batch(batch):
    """Normalize a BCHW tensor for ImageNet-pretrained VGG19."""
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


def save_image(tensor, path):
    """Save a BCHW or CHW tensor in [0, 1] as an image file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = tensor.detach().cpu().clone()
    if image.dim() == 4:
        image = image.squeeze(0)
    image = image.clamp(0.0, 1.0)

    transforms.ToPILImage()(image).save(path)


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

