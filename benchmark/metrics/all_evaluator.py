"""
All-based Evaluator
DINOv3 feature comparison evaluation based on full images
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from typing import List, Dict, Union
from loguru import logger
from metrics.metrics_utils import MetricsCalculator, resize_to_match


class AllEvaluator:
    """Full image-based evaluator"""

    def __init__(
        self,
        dinov3_path: str,
        device: str = 'cuda',
        image_size: int = 1024,
        preloaded_model: tuple = None
    ):
        """
        Initialize All evaluator.

        Args:
            dinov3_path: DINOv3 model path
            device: Compute device
            image_size: Image processing size
            preloaded_model: Pre-loaded (processor, model) tuple for sharing to avoid repeated loading
        """
        self.image_size = image_size

        if preloaded_model is not None:
            # Use pre-loaded model (avoid repeated loading)
            logger.info("Using pre-loaded DINOv3 model (shared)")
            self.processor, self.model = preloaded_model

            # Get actual device from pre-loaded model
            if hasattr(self.model, 'device'):
                self.device = self.model.device
            else:
                # Get device from model parameters
                self.device = next(self.model.parameters()).device

            logger.info(f"Using model device: {self.device}")
        else:
            # Normal model loading
            self.device = device
            logger.info(f"Loading DINOv3 model from {dinov3_path}...")

            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(dinov3_path)
            self.model = AutoModel.from_pretrained(
                dinov3_path,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True
            ).to(device, dtype=torch.float32)  # Use float32 to avoid NaN issues

            self.model.eval()

            logger.success("DINOv3 model loaded successfully!")

        # Initialize metrics calculator (use same device)
        device_str = str(self.device) if torch.is_tensor(self.device) or isinstance(self.device, torch.device) else self.device
        self.metrics_calculator = MetricsCalculator(device=device_str)

    def extract_features(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """
        Extract DINOv3 token features from images.

        Args:
            images: Single image or list of images

        Returns:
            Feature array [N, num_tokens, D]
        """
        # Ensure it's a list
        if isinstance(images, Image.Image):
            images = [images]

        # Resize images (maintain aspect ratio)
        resized_images = []
        for img in images:
            img = img.convert('RGB')
            width, height = img.size
            max_dim = max(width, height)
            scale = self.image_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_images.append(resized_img)

        # Batch processing
        inputs = self.processor(images=resized_images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device, dtype=torch.float32)  # Use float32

        # Extract token features (use last_hidden_state instead of pooler_output)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            # last_hidden_state: [batch, num_tokens, hidden_dim]
            features = outputs.last_hidden_state.detach().cpu().float().numpy()

        # Check and handle NaN values
        if np.isnan(features).any():
            logger.warning(f"Detected NaN in DINOv2 features! Replacing with zeros.")
            logger.warning(f"NaN count: {np.isnan(features).sum()} / {features.size}")
            # Replace NaN with 0
            features = np.nan_to_num(features, nan=0.0)

        return features

    def evaluate_single(
        self,
        ground_truth: Image.Image,
        generated_image: Image.Image
    ) -> Dict:
        """
        Evaluate a single sample.

        Args:
            ground_truth: Real target image
            generated_image: Generated image

        Returns:
            Dictionary containing all metrics
        """
        try:
            # Extract features
            features = self.extract_features([ground_truth, generated_image])
            gt_feat = features[0]
            gen_feat = features[1]

            # Resize images to same size
            gt_array, gen_array = resize_to_match(ground_truth, generated_image)

            # Compute all metrics
            metrics = self.metrics_calculator.compute_all_metrics(
                gt_array,
                gen_array,
                gt_feat,
                gen_feat
            )

            return metrics

        except Exception as e:
            logger.error(f"Error in all-based evaluation: {e}")
            return self._default_metrics()

    def evaluate_batch(
        self,
        ground_truths: List[Image.Image],
        generated_images: List[Image.Image],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Batch evaluation.

        Args:
            ground_truths: List of real target images
            generated_images: List of generated images
            show_progress: Whether to show progress

        Returns:
            List of evaluation results
        """
        results = []

        iterator = zip(ground_truths, generated_images)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(
                list(iterator),
                desc="All-based Evaluation",
                unit="image"
            )

        for gt, generated in iterator:
            metrics = self.evaluate_single(gt, generated)
            results.append(metrics)

        return results

    def compute_average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """
        Compute average metrics.

        Args:
            metrics_list: List of metrics

        Returns:
            Average metrics dictionary
        """
        if not metrics_list:
            return self._default_metrics()

        metric_names = [
            'cosine_similarity',
            'mse',
            'ssim',
            'psnr',
            'lpips'
        ]

        averages = {}
        for name in metric_names:
            values = [m[name] for m in metrics_list if m.get(name) is not None]
            averages[name] = np.mean(values) if values else None
            averages[f'{name}_std'] = np.std(values) if values else None

        return averages

    def _default_metrics(self) -> Dict:
        """Return default metrics"""
        return {
            'cosine_similarity': None,
            'mse': None,
            'ssim': None,
            'psnr': None,
            'lpips': None
        }
