from abc import ABC, abstractmethod
import torch
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from shared.data.dataset_meta import DatasetMeta as M
from shared.hyperparameters import Hyperparameters


class NoiseFactory:

    @staticmethod
    def SUPPORTED_NOISE_TYPES() -> list:
        h = Hyperparameters.get()
        noises = ["gaussian", "salt_and_pepper", "speckle"]  # rm "poisson"
        if M.is_vision_dataset(h):
            noises = noises + ["affine_transform_without_zoom", "affine_transform", "elastic_distortion"]

        return noises

    @staticmethod
    def _get_noise(noise_type):
        if noise_type == "gaussian":
            return GaussianNoise
        elif noise_type == "salt_and_pepper":
            return SaltAndPepperNoise
        elif noise_type == "poisson":
            return PoissonNoise
        elif noise_type == "speckle":
            return SpeckleNoise
        elif noise_type == "affine_transform_without_zoom":
            return AffineTransformWithoutZoomNoise
        elif noise_type == "affine_transform":
            return AffineTransformNoise
        elif noise_type == "elastic_distortion":
            return ElasticDistortionNoise
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    @staticmethod
    def apply_noise(images, noise_type: str, noise_factor: float):
        """
        In: images: (batch_size, nb_views, channels, height, width)
        """
        assert noise_type in NoiseFactory.SUPPORTED_NOISE_TYPES(), f"Unknown noise type: {noise_type}"

        noise = NoiseFactory._get_noise(noise_type)
        return noise.apply_noise(images, noise_factor)


class AbstractNoise(ABC):
    @staticmethod
    @abstractmethod
    def apply_noise(images, noise_factor):
        pass


class GaussianNoise(AbstractNoise):
    ## WARNING: THE DATA_RANGE IS NOT 100% CORRECT; WHEN IMAGE DOESNT HAVE 1 AS MAX, IT WILL BE CLAMPED TO SMALLER VALUES
    @staticmethod
    def apply_noise(images, noise_factor):
        data_range = images.min(), images.max()
        noisy_images = images + noise_factor * torch.randn_like(images)
        # noisy_images = torch.clamp(noisy_images, data_range[0], data_range[1])
        return noisy_images


class SaltAndPepperNoise(AbstractNoise):
    @staticmethod
    def apply_noise(images, noise_factor):
        data_range = images.min(), images.max()
        noisy_images = images.clone()
        num_salt = int(noise_factor * images.numel() * 0.5)
        num_pepper = int(noise_factor * images.numel() * 0.5)

        # Add salt noise
        coords = [torch.randint(0, i, (num_salt,)) for i in images.shape]
        noisy_images[coords] = data_range[1]

        # Add pepper noise
        coords = [torch.randint(0, i, (num_pepper,)) for i in images.shape]
        noisy_images[coords] = data_range[0]

        return noisy_images


class PoissonNoise(AbstractNoise):
    @staticmethod
    def apply_noise(images, _):
        data_range = images.min(), images.max()
        shift = -data_range[0] if data_range[0] < 0 else 0
        shifted_images = images + shift
        noisy_images = shifted_images + torch.poisson(shifted_images)
        noisy_images = noisy_images - shift
        # noisy_images = torch.clamp(noisy_images, data_range[0], data_range[1])
        return noisy_images


class SpeckleNoise(AbstractNoise):
    @staticmethod
    def apply_noise(images, noise_factor):
        data_range = images.min(), images
        noisy_images = images + noise_factor * images * torch.randn_like(images)
        # noisy_images = torch.clamp(noisy_images, data_range[0], data_range[1])
        return noisy_images


class AffineTransformNoise(AbstractNoise):
    @staticmethod
    def apply_noise(images, noise_factor):
        angle = noise_factor * 30  # Rotate by up to 30 degrees
        translate = (0, 0)
        scale = 1.0 + noise_factor * 1
        shear = noise_factor * 10  # Shear by up to 10 degrees

        noisy_images = []
        for img in images:
            # Pad the image to avoid cropping parts during transformations
            pad_size = int(max(img.size(-2), img.size(-1)) * 0.2)
            padded_img = F.pad(img, [pad_size, pad_size, pad_size, pad_size], fill=int(img.mean().item() * 255))

            # Apply the affine transformation
            transformed_img = F.affine(padded_img, angle, translate, scale, shear)

            # Center crop back to original size
            crop_size = img.size(-2)  # Assume square images
            cropped_img = F.center_crop(transformed_img, crop_size)
            noisy_images.append(cropped_img)

        return torch.stack(noisy_images)


# import torch.nn.functional as F


class AffineTransformWithoutZoomNoise(AbstractNoise):
    @staticmethod
    def apply_noise(images, noise_factor):
        angle = noise_factor * 30  # Rotate by up to 30 degrees
        translate = (0, 0)  # horizontal/vertical shift
        scale = 1.0  # No scaling
        shear = noise_factor * 40  # Shear by up to 40 degrees

        noisy_images = []
        for img in images:
            # Reflect the image symmetrically around its borders
            pad_size = int(max(img.size(-2), img.size(-1)) * 0.4)
            padded_img = F.pad(img, padding=[pad_size, pad_size, pad_size, pad_size], padding_mode='reflect')

            # Apply the affine transformation
            transformed_img = F.affine(padded_img, angle, translate, scale, shear)

            # Center crop back to original size
            crop_size = img.size(-2)  # Assume square images
            cropped_img = F.center_crop(transformed_img, crop_size)
            noisy_images.append(cropped_img)

        return torch.stack(noisy_images)


class ElasticDistortionNoise(AbstractNoise):
    @staticmethod
    def apply_noise(images, noise_factor):
        # images: (batch_size, nb_views, channels, height, width)
        # rm nb_views channel
        images = images.squeeze(1)  # assume nb_views=1

        noisy_images = []
        for img in images:
            # Generate random displacement fields
            displacement = noise_factor * 5  # Displacement scale
            dx = torch.empty(img.shape[1:], device=img.device).uniform_(-displacement, displacement)
            dy = torch.empty(img.shape[1:], device=img.device).uniform_(-displacement, displacement)

            # Create meshgrid
            y, x = torch.meshgrid(torch.arange(img.shape[1]), torch.arange(img.shape[2]))
            x, y = x.to(img.device), y.to(img.device)

            # Apply distortions
            indices_x = torch.clamp(x + dx, 0, img.shape[2] - 1).long()
            indices_y = torch.clamp(y + dy, 0, img.shape[1] - 1).long()

            distorted_img = img[:, indices_y, indices_x]
            noisy_images.append(distorted_img)

        return torch.stack(noisy_images).unsqueeze(1)  # Add back the nb_views dimension


if __name__ == "__main__":
    # Load an image from the local file system
    img_path = "./temp3.png"
    img = Image.open(img_path)

    # Convert the image to a tensor
    img_tensor = F.to_tensor(img).unsqueeze(0).unsqueeze(0)  # Add batch dimension, nb_views_dim

    # Apply affine transform noise
    noise_factor = 1  # Example noise factor
    noisy_img_tensor = ElasticDistortionNoise.apply_noise(img_tensor, noise_factor)
    # noisy_img_tensor = AffineTransformWithoutZoomNoise.apply_noise(img_tensor, noise_factor)
    # noisy_img_tensor = AffineTransformNoise.apply_noise(img_tensor, noise_factor)
    # noisy_img_tensor = SpeckleNoise.apply_noise(img_tensor, noise_factor)

    # Convert tensors back to images
    original_img = F.to_pil_image(img_tensor.squeeze(0).squeeze(0))
    noisy_img = F.to_pil_image(noisy_img_tensor.squeeze(0).squeeze(0))

    # Display the original and noisy images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(noisy_img)
    axes[1].set_title("Noisy Image")
    axes[1].axis("off")

    plt.show()
