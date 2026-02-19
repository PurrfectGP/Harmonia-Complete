"""MetaFBP image preprocessing with ImageNet normalization."""
import io
from PIL import Image
import torch
from torchvision import transforms

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

# Inference transform: resize, center crop, normalize
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Training transform (for reference, not used in production)
training_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess a single image for MetaFBP inference.

    Args:
        image_path: Path to the image file.

    Returns:
        Preprocessed tensor of shape (1, 3, 224, 224).
    """
    img = Image.open(image_path).convert("RGB")
    tensor = inference_transform(img)
    return tensor.unsqueeze(0)  # Add batch dimension


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image from bytes for MetaFBP inference."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = inference_transform(img)
    return tensor.unsqueeze(0)


def preprocess_batch(image_paths: list[str]) -> torch.Tensor:
    """Preprocess a batch of images."""
    tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensors.append(inference_transform(img))
    return torch.stack(tensors)
