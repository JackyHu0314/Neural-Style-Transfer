from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from libs.dataset import IMAGE_EXTENSIONS


def recursive_image_files(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No training image found under: {root}")
    return sorted(files)


class ContentImageDataset(Dataset):
    """Content image dataset for feed-forward style-transfer training."""

    def __init__(self, root, image_size=256):
        self.files = recursive_image_files(root)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image)

