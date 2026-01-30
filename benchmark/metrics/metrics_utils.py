"""
Metrics Calculation Utility Module
Provides various image similarity and quality evaluation metrics
"""

import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from typing import Union, Tuple, List
from scipy import linalg
from torchvision import models, transforms
from loguru import logger


class MetricsCalculator:
    """Metrics calculator"""

    def __init__(self, device='cuda'):
        """
        Initialize metrics calculator.

        Args:
            device: Compute device
        """
        self.device = device
        # Initialize LPIPS model (using AlexNet as backbone)
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()

    def cosine_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        Supports token-level features [num_tokens, D] and global features [D].

        Args:
            emb1: First embedding vector, shape [D] or [num_tokens, D]
            emb2: Second embedding vector, shape [D] or [num_tokens, D]

        Returns:
            Cosine similarity [-1, 1]
        """
        # Handle token-level features [num_tokens, D]
        if emb1.ndim == 2 and emb2.ndim == 2:
            # Compute cosine similarity for each token, then average
            # Normalize each token
            emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

            # Compute token-wise cosine similarity matrix [num_tokens1, num_tokens2]
            similarity_matrix = np.matmul(emb1_norm, emb2_norm.T)

            # Take average of max matching (find most similar corresponding token for each token)
            max_sim_per_token1 = np.max(similarity_matrix, axis=1)  # For each token in emb1
            max_sim_per_token2 = np.max(similarity_matrix, axis=0)  # For each token in emb2

            # Average of bidirectional matching
            similarity = (np.mean(max_sim_per_token1) + np.mean(max_sim_per_token2)) / 2.0

        # Handle global features [D]
        elif emb1.ndim == 1 and emb2.ndim == 1:
            # Normalize
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)

            # Compute cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
        else:
            raise ValueError(f"Embedding dimensions don't match: emb1 shape {emb1.shape}, emb2 shape {emb2.shape}")

        return float(similarity)

    def mse(
        self,
        img1: Union[np.ndarray, Image.Image],
        img2: Union[np.ndarray, Image.Image]
    ) -> float:
        """
        Compute Mean Squared Error (MSE).

        Args:
            img1: First image
            img2: Second image

        Returns:
            MSE value [0, inf), lower is better
        """
        # Convert to numpy array
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)

        # Ensure same dimensions
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

        # Normalize to [0, 1]
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        # Compute MSE
        mse_value = np.mean((img1 - img2) ** 2)

        return float(mse_value)

    def ssim(
        self,
        img1: Union[np.ndarray, Image.Image],
        img2: Union[np.ndarray, Image.Image],
        multichannel: bool = True
    ) -> float:
        """
        Compute Structural Similarity Index (SSIM).

        Args:
            img1: First image
            img2: Second image
            multichannel: Whether image is multichannel

        Returns:
            SSIM value [0, 1], higher is better
        """
        # Convert to numpy array
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)

        # Ensure same dimensions
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

        # Normalize to [0, 1]
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        # Compute SSIM
        ssim_value = ssim(
            img1,
            img2,
            channel_axis=2 if multichannel and img1.ndim == 3 else None,
            data_range=1.0
        )

        return float(ssim_value)

    def psnr(
        self,
        img1: Union[np.ndarray, Image.Image],
        img2: Union[np.ndarray, Image.Image]
    ) -> float:
        """
        Compute Peak Signal-to-Noise Ratio (PSNR).

        Args:
            img1: First image
            img2: Second image

        Returns:
            PSNR value [0, inf), higher is better
        """
        # Convert to numpy array
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)

        # Ensure same dimensions
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

        # Normalize to [0, 1]
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        # Compute PSNR
        psnr_value = psnr(img1, img2, data_range=1.0)

        return float(psnr_value)

    def lpips_score(
        self,
        img1: Union[np.ndarray, Image.Image],
        img2: Union[np.ndarray, Image.Image]
    ) -> float:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity).

        Args:
            img1: First image
            img2: Second image

        Returns:
            LPIPS value [0, inf), lower is better
        """
        # Convert to numpy array
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)

        # Ensure same dimensions
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

        # Convert to tensor and normalize to [-1, 1]
        img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

        # Move to device
        img1_tensor = img1_tensor.to(self.device)
        img2_tensor = img2_tensor.to(self.device)

        # Compute LPIPS
        with torch.no_grad():
            lpips_value = self.lpips_model(img1_tensor, img2_tensor)

        return float(lpips_value.item())

    def compute_all_metrics(
        self,
        img1: Union[np.ndarray, Image.Image],
        img2: Union[np.ndarray, Image.Image],
        emb1: np.ndarray = None,
        emb2: np.ndarray = None
    ) -> dict:
        """
        Compute all metrics.

        Args:
            img1: First image
            img2: Second image
            emb1: First image embedding (optional)
            emb2: Second image embedding (optional)

        Returns:
            Dictionary containing all metrics
        """
        metrics = {}

        # If embeddings provided, compute cosine similarity
        if emb1 is not None and emb2 is not None:
            metrics['cosine_similarity'] = self.cosine_similarity(emb1, emb2)

        # Compute image metrics
        try:
            metrics['mse'] = self.mse(img1, img2)
            metrics['ssim'] = self.ssim(img1, img2)
            metrics['psnr'] = self.psnr(img1, img2)
            metrics['lpips'] = self.lpips_score(img1, img2)
        except Exception as e:
            print(f"Error computing metrics: {e}")
            metrics.update({
                'mse': None,
                'ssim': None,
                'psnr': None,
                'lpips': None
            })

        return metrics


def resize_to_match(
    img1: Union[np.ndarray, Image.Image],
    img2: Union[np.ndarray, Image.Image]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize two images to same dimensions.

    Args:
        img1: First image
        img2: Second image

    Returns:
        Two images with matched dimensions (numpy arrays)
    """
    # Convert to PIL Image
    if isinstance(img1, np.ndarray):
        img1 = Image.fromarray(img1)
    if isinstance(img2, np.ndarray):
        img2 = Image.fromarray(img2)

    # Get dimensions
    w1, h1 = img1.size
    w2, h2 = img2.size

    # If dimensions already match, return directly
    if (w1, h1) == (w2, h2):
        return np.array(img1), np.array(img2)

    # Use first image dimensions as target
    target_size = (w1, h1)

    # Resize second image
    img2_resized = img2.resize(target_size, Image.Resampling.LANCZOS)

    return np.array(img1), np.array(img2_resized)


def normalize_metrics(metrics: dict) -> dict:
    """
    Normalize metrics to [0, 1] range, higher is better.

    Args:
        metrics: Original metrics dictionary

    Returns:
        Normalized metrics dictionary
    """
    normalized = {}

    for key, value in metrics.items():
        if value is None:
            normalized[key] = None
            continue

        if key == 'cosine_similarity':
            # [-1, 1] -> [0, 1]
            normalized[key] = (value + 1) / 2
        elif key == 'ssim':
            # [0, 1] -> [0, 1] (already normalized)
            normalized[key] = value
        elif key == 'psnr':
            # [0, inf) -> [0, 1], using sigmoid mapping
            # PSNR typically between 20-40, map 30 to 0.5
            normalized[key] = 1 / (1 + np.exp(-(value - 30) / 5))
        elif key in ['mse', 'lpips']:
            # [0, inf) -> [1, 0], lower is better, use inverse normalized
            # Use inverse sigmoid mapping
            if key == 'mse':
                # MSE typically between 0-0.1
                normalized[key] = 1 / (1 + np.exp((value - 0.05) / 0.01))
            else:  # lpips
                # LPIPS typically between 0-1
                normalized[key] = 1 / (1 + np.exp((value - 0.5) / 0.1))
        else:
            normalized[key] = value

    return normalized


class FIDCalculator:
    """FID (Frechet Inception Distance) Calculator"""

    def __init__(self, device='cuda', dims=2048):
        """
        Initialize FID calculator.

        Args:
            device: Compute device
            dims: Inception feature dimensions (default 2048, using pool3 layer)
        """
        self.device = device
        self.dims = dims

        logger.info("Initializing FID Calculator with InceptionV3...")

        # Load pre-trained InceptionV3 model
        self.inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(299),  # InceptionV3 requires 299x299
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.success("FID Calculator initialized!")

    def extract_features(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Extract Inception features from images.

        Args:
            images: List of images (PIL Image or numpy array)

        Returns:
            Feature matrix (N, dims)
        """
        features = []

        with torch.no_grad():
            for img in images:
                # Convert to PIL Image
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)

                # Ensure RGB mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Preprocess and convert to tensor
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)

                # Extract features
                # InceptionV3 pool3 layer outputs 2048-dim features
                pred = self.inception(img_tensor)
                features.append(pred.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features

    def calculate_fid(
        self,
        real_images: List[Union[Image.Image, np.ndarray]],
        generated_images: List[Union[Image.Image, np.ndarray]]
    ) -> float:
        """
        Calculate FID score.

        Args:
            real_images: List of real images
            generated_images: List of generated images

        Returns:
            FID score (lower is better)
        """
        logger.info(f"Calculating FID for {len(real_images)} real and {len(generated_images)} generated images...")

        # Extract features
        logger.info("Extracting features from real images...")
        real_features = self.extract_features(real_images)

        logger.info("Extracting features from generated images...")
        gen_features = self.extract_features(generated_images)

        # Compute mean and covariance
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)

        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)

        # Compute FID
        fid = self._calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

        logger.success(f"FID calculated: {fid:.4f}")

        return float(fid)

    def _calculate_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps=1e-6
    ) -> float:
        """
        Calculate Frechet distance.

        Args:
            mu1: Mean of first distribution
            sigma1: Covariance matrix of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance matrix of second distribution
            eps: Numerical stability constant

        Returns:
            Frechet distance
        """
        # Compute squared mean difference
        diff = mu1 - mu2
        mean_diff = diff.dot(diff)

        # Compute square root of covariance matrix product
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # Handle numerical instability
        if not np.isfinite(covmean).all():
            logger.warning("FID calculation produced singular matrix. Adding epsilon to diagonal.")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # If complex, take real part
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                logger.warning(f"Imaginary component {m} in FID calculation")
            covmean = covmean.real

        # Compute FID
        fid = mean_diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

        return fid
