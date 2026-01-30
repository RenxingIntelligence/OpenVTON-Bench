"""
Pixel-based Evaluator
Direct pixel space comparison between generated and target images (without DINOv2 features)
"""

import numpy as np
from PIL import Image
from typing import List, Dict
from loguru import logger

import sys
from pathlib import Path
benchmark_dir = Path(__file__).parent.parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

from metrics.metrics_utils import MetricsCalculator, FIDCalculator, resize_to_match


class PixelEvaluator:
    """Pixel-based evaluator (without deep features)"""

    def __init__(self, device: str = 'cuda', enable_fid: bool = True):
        """
        Initialize Pixel evaluator.

        Args:
            device: Compute device
            enable_fid: Whether to enable FID calculation (requires batch images)
        """
        self.device = device
        self.enable_fid = enable_fid
        self.metrics_calculator = MetricsCalculator(device=device)

        # Initialize FID calculator (if enabled)
        if self.enable_fid:
            self.fid_calculator = FIDCalculator(device=device)
            logger.info("FID calculation enabled")
        else:
            self.fid_calculator = None
            logger.info("FID calculation disabled")

        logger.success("Pixel evaluator initialized!")

    def evaluate_single(
        self,
        ground_truth: Image.Image,
        generated_image: Image.Image
    ) -> Dict:
        """
        Evaluate a single sample (pure pixel comparison).

        Args:
            ground_truth: Real target image
            generated_image: Generated image

        Returns:
            Dictionary containing pixel-level metrics
        """
        try:
            # Resize images to same size
            gt_array, gen_array = resize_to_match(ground_truth, generated_image)

            # Only compute pixel-level metrics (no features)
            metrics = {
                'mse': self.metrics_calculator.mse(gt_array, gen_array),
                'ssim': self.metrics_calculator.ssim(gt_array, gen_array),
                'psnr': self.metrics_calculator.psnr(gt_array, gen_array),
                'lpips': self.metrics_calculator.lpips_score(gt_array, gen_array)
            }

            return metrics

        except Exception as e:
            logger.error(f"Error in pixel-based evaluation: {e}")
            return self._default_metrics()

    def evaluate_batch(
        self,
        ground_truths: List[Image.Image],
        generated_images: List[Image.Image],
        show_progress: bool = True
    ) -> Dict:
        """
        Batch evaluation.

        Args:
            ground_truths: List of real target images
            generated_images: List of generated images
            show_progress: Whether to show progress

        Returns:
            Dictionary containing per-sample metrics and FID
        """
        # Per-sample evaluation
        per_sample_results = []

        iterator = zip(ground_truths, generated_images)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(
                list(iterator),
                desc="Pixel-based Evaluation",
                unit="image"
            )

        for gt, generated in iterator:
            metrics = self.evaluate_single(gt, generated)
            per_sample_results.append(metrics)

        # Calculate FID (batch metric)
        fid_score = None
        if self.enable_fid and self.fid_calculator is not None:
            try:
                logger.info("Calculating FID score...")
                fid_score = self.fid_calculator.calculate_fid(
                    ground_truths,
                    generated_images
                )
            except Exception as e:
                logger.error(f"Error calculating FID: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Return results
        results = {
            'per_sample_metrics': per_sample_results,
            'fid': fid_score  # FID is batch metric, not per-sample
        }

        return results

    def compute_average_metrics(self, results: Dict) -> Dict:
        """
        Compute average metrics.

        Args:
            results: Results dictionary returned by evaluate_batch

        Returns:
            Average metrics dictionary (including FID)
        """
        per_sample_metrics = results.get('per_sample_metrics', [])

        if not per_sample_metrics:
            return self._default_metrics()

        metric_names = ['mse', 'ssim', 'psnr', 'lpips']

        averages = {}
        for name in metric_names:
            values = [m[name] for m in per_sample_metrics if m.get(name) is not None]
            averages[name] = np.mean(values) if values else None
            averages[f'{name}_std'] = np.std(values) if values else None

        # Add FID score (batch metric)
        averages['fid'] = results.get('fid')

        return averages

    def _default_metrics(self) -> Dict:
        """Return default metrics"""
        return {
            'mse': None,
            'ssim': None,
            'psnr': None,
            'lpips': None,
            'fid': None
        }
