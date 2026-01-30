"""
Garment-based Evaluator
Fine-grained evaluation based on garment regions, including SAM segmentation and multi-scale erosion
"""

import torch
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoImageProcessor
from typing import List, Dict, Optional, Union
from pathlib import Path
from loguru import logger
from utils.sam_segmenter import SAMSegmenter
from metrics.metrics_utils import MetricsCalculator, resize_to_match


class GarmentEvaluator:
    """Garment region-based evaluator"""

    def __init__(
        self,
        dinov3_path: str,
        sam_checkpoint: str,
        device: str = 'cuda',
        image_size: int = 1024,
        erosion_scales: List[int] = [5, 10, 20],
        preloaded_model: tuple = None,
        # SAM3 parameters
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        cluster_label_loader = None,
        default_prompt: str = "clothing",
        # Simplified prompts
        simplified_prompts_path: Optional[str] = None,
        # Debug parameters
        debug_dir: Optional[str] = None,
        save_debug_samples: int = 3,
        # Multi-GPU parameters
        gpu_ids: Optional[List[int]] = None,
        num_workers: int = 4,
    ):
        """
        Initialize the Garment Evaluator

        Args:
            dinov3_path: Path to the DINOv3 model
            sam_checkpoint: Path to the SAM3 model checkpoint
            device: Computing device
            image_size: Image processing size
            erosion_scales: List of erosion scales
            simplified_prompts_path: Path to simplified prompt mapping file
            debug_dir: Debug output directory (if provided, visualization images will be saved)
            save_debug_samples: Save debug images for the first N samples (default 3)
            gpu_ids: List of GPU IDs (for multi-GPU acceleration)
            num_workers: Number of multi-process workers (for accelerating evaluation)
        """
        self.device = device
        self.image_size = image_size
        self.erosion_scales = erosion_scales

        # Multi-GPU configuration
        self.gpu_ids = gpu_ids if gpu_ids else [0]
        self.num_workers = num_workers
        self.use_multi_gpu = len(self.gpu_ids) > 1

        if self.use_multi_gpu:
            logger.info(f"Multi-GPU mode enabled: Using GPUs {self.gpu_ids}")
            logger.info(f"Multi-process workers: {self.num_workers}")
        else:
            logger.info(f"Single GPU mode: Using GPU {self.gpu_ids[0]}")

        # Save initialization parameters (for recreating instances in multi-process)
        self.init_params = {
            'dinov3_path': dinov3_path,
            'sam_checkpoint': sam_checkpoint,
            'image_size': image_size,
            'erosion_scales': erosion_scales,
            'threshold': threshold,
            'mask_threshold': mask_threshold,
            'cluster_label_loader': cluster_label_loader,
            'default_prompt': default_prompt,
            'simplified_prompts_path': simplified_prompts_path,
        }

        # Load simplified prompt mapping
        self.simplified_prompts = {}
        if simplified_prompts_path and Path(simplified_prompts_path).exists():
            try:
                with open(simplified_prompts_path, 'r', encoding='utf-8') as f:
                    self.simplified_prompts = json.load(f)
                logger.info(f"Loaded {len(self.simplified_prompts)} simplified prompts")
            except Exception as e:
                logger.warning(f"Failed to load simplified prompts: {e}")

        # Debug settings
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.save_debug_samples = save_debug_samples
        self.debug_counter = 0  # Record the number of saved samples

        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Debug images will be saved to: {self.debug_dir}")

        # Initialize DINOv3
        if preloaded_model is not None:
            # Use pre-loaded model (to avoid duplicate loading)
            logger.info("Using pre-loaded DINOv3 model (shared)")
            self.processor, self.model = preloaded_model
        else:
            # Normal model loading
            logger.info(f"Loading DINOv3 model from {dinov3_path}...")
            self.processor = AutoImageProcessor.from_pretrained(dinov3_path)
            self.model = AutoModel.from_pretrained(
                dinov3_path,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True
            ).to(device, dtype=torch.float32)
            self.model.eval()
            logger.success("DINOv3 model loaded!")

        # Initialize SAM3 segmenter
        self.sam_segmenter = SAMSegmenter(
            checkpoint_path=sam_checkpoint,
            device=device,
            threshold=threshold,
            mask_threshold=mask_threshold,
            cluster_label_loader=cluster_label_loader,
            default_prompt=default_prompt
        )

        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(device=device)

        logger.success("Garment evaluator initialized!")

    def _get_simplified_prompt(self, source_path: Optional[str]) -> str:
        """
        Get simplified text prompt (reference: test_sam3_segmentation.py)

        Args:
            source_path: Source image path

        Returns:
            Simplified prompt
        """
        if not source_path or not self.sam_segmenter.cluster_label_loader:
            return self.sam_segmenter.default_prompt

        # Get original label from cluster_label_loader
        filename = Path(source_path).name
        original_label = self.sam_segmenter.cluster_label_loader.get_label_for_filename(filename)

        if not original_label:
            return self.sam_segmenter.default_prompt

        # Look up simplified prompt
        simplified = self.simplified_prompts.get(original_label)
        if simplified:
            logger.debug(f"Using simplified prompt '{simplified}' for label '{original_label}'")
            return simplified
        else:
            logger.debug(f"No simplified prompt for '{original_label}', using original")
            return original_label

    def extract_features(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """
        Extract DINOv3 token features from images

        Args:
            images: Single image or list of images

        Returns:
            Feature vector array [N, num_tokens, D]
        """
        # Ensure it is a list
        if isinstance(images, Image.Image):
            images = [images]

        # Resize images
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
        pixel_values = inputs['pixel_values'].to(self.device, dtype=torch.float32)

        # Extract token features (using last_hidden_state instead of pooler_output)
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
        generated_image: Image.Image,
        source_path: Optional[str] = None,
        sample_idx: Optional[int] = None
    ) -> Dict:
        """
        Evaluate a single sample

        Args:
            ground_truth: Ground truth target image
            generated_image: Generated image
            source_path: Source image path (used for SAM3 label lookup)
            sample_idx: Sample index (used for debug image naming)

        Returns:
            Dictionary containing metrics for all scales
        """
        try:
            # 1. Get simplified prompt and segment garment region
            logger.debug("Segmenting garments...")
            # simplified_prompt = "Clothing"
            simplified_prompt = self._get_simplified_prompt(source_path)

            # Directly use simplified prompt for segmentation (instead of segment_garment's internal logic)
            gt_mask = self._segment_with_prompt(ground_truth, simplified_prompt)
            gen_mask = self._segment_with_prompt(generated_image, simplified_prompt)

            if gt_mask is None or gen_mask is None:
                logger.warning("Failed to segment garment, skipping sample (returning None)...")
                return None  # Return None instead of default metrics, allowing caller to filter out

            # 2. Extract garment images
            gt_garment = self.sam_segmenter.apply_mask(ground_truth, gt_mask)
            gen_garment = self.sam_segmenter.apply_mask(generated_image, gen_mask)

            # 3. Multi-scale erosion
            logger.debug(f"Applying multi-scale erosion: {self.erosion_scales}")
            gt_masks_eroded = self.sam_segmenter.multi_scale_erosion(gt_mask, self.erosion_scales)
            gen_masks_eroded = self.sam_segmenter.multi_scale_erosion(gen_mask, self.erosion_scales)

            # 4. Save debug images (first N samples) - only save garment parts
            if self.debug_dir and self.debug_counter < self.save_debug_samples:
                self._save_debug_visualization(
                    ground_truth, generated_image,
                    gt_mask, gen_mask,
                    gt_garment, gen_garment,
                    gt_masks_eroded, gen_masks_eroded,
                    sample_idx if sample_idx is not None else self.debug_counter,
                    simplified_prompt
                )
                self.debug_counter += 1

            # 5. Compute metrics for each scale
            results = {}

            # Original scale (no erosion)
            results['scale_0'] = self._compute_metrics_for_scale(
                ground_truth, generated_image,
                gt_garment, gen_garment,
                scale_name="original"
            )

            # Erosion scales
            for idx, scale in enumerate(self.erosion_scales, 1):
                gt_garment_eroded = self.sam_segmenter.apply_mask(ground_truth, gt_masks_eroded[idx-1])
                gen_garment_eroded = self.sam_segmenter.apply_mask(generated_image, gen_masks_eroded[idx-1])

                results[f'scale_{idx}'] = self._compute_metrics_for_scale(
                    ground_truth, generated_image,
                    gt_garment_eroded, gen_garment_eroded,
                    scale_name=f"{scale}px"
                )

            # 6. Compute average metrics
            results['average'] = self._compute_average_metrics(results)

            return results

        except Exception as e:
            logger.error(f"Error in garment-based evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None  # Return None to allow caller to filter

    def _segment_with_prompt(
        self,
        image: Image.Image,
        prompt: str
    ) -> Optional[np.ndarray]:
        """
        Segment image using specified prompt (reference: test_sam3_segmentation.py)

        Args:
            image: Input image
            prompt: Text prompt

        Returns:
            Segmentation mask
        """
        try:
            img_rgb = image.convert('RGB')
            inputs = self.sam_segmenter.processor(
                images=img_rgb,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.sam_segmenter.model(**inputs)

            # Post-processing
            results = self.sam_segmenter.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.sam_segmenter.threshold,
                mask_threshold=self.sam_segmenter.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            if len(results['masks']) == 0:
                logger.debug(f"No masks found for prompt: '{prompt}'")
                return None

            # Get the mask with highest confidence
            best_idx = torch.argmax(results['scores']).item()
            mask = results['masks'][best_idx].cpu().numpy()
            best_score = results['scores'][best_idx].item()
            mask_uint8 = (mask > 0).astype(np.uint8) * 255

            logger.debug(
                f"Found {len(results['masks'])} masks for '{prompt}', "
                f"selected with score {best_score:.3f}"
            )

            return mask_uint8

        except Exception as e:
            logger.error(f"Error in segmentation with prompt '{prompt}': {e}")
            return None

    def _save_debug_visualization(
        self,
        ground_truth: Image.Image,
        generated_image: Image.Image,
        gt_mask: np.ndarray,
        gen_mask: np.ndarray,
        gt_garment: Image.Image,
        gen_garment: Image.Image,
        gt_masks_eroded: List[np.ndarray],
        gen_masks_eroded: List[np.ndarray],
        sample_idx: int,
        prompt: str
    ):
        """
        Save debug visualization images (only save garment parts)

        Args:
            ground_truth: Ground truth target image
            generated_image: Generated image
            gt_mask: GT original mask
            gen_mask: Generated image's original mask
            gt_garment: GT garment image (after applying mask)
            gen_garment: Generated garment image (after applying mask)
            gt_masks_eroded: List of GT eroded masks
            gen_masks_eroded: List of generated image's eroded masks
            sample_idx: Sample index
            prompt: SAM3 prompt used
        """
        try:
            sample_dir = self.debug_dir / f"sample_{sample_idx:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # 1. Save original segmented garment images (white background)
            gt_garment.save(sample_dir / "01_gt_garment_original.png")
            gen_garment.save(sample_dir / "02_gen_garment_original.png")

            # 2. Save eroded garment images (for each scale)
            for idx, scale in enumerate(self.erosion_scales):
                gt_garment_eroded = self.sam_segmenter.apply_mask(
                    ground_truth, gt_masks_eroded[idx]
                )
                gen_garment_eroded = self.sam_segmenter.apply_mask(
                    generated_image, gen_masks_eroded[idx]
                )

                gt_garment_eroded.save(sample_dir / f"03_gt_garment_erosion_{scale}px.png")
                gen_garment_eroded.save(sample_dir / f"04_gen_garment_erosion_{scale}px.png")

            # 3. Save comparison image (original image + mask overlay + garment extraction)
            self._save_comparison_visualization(
                ground_truth, generated_image,
                gt_mask, gen_mask,
                gt_garment, gen_garment,
                sample_dir,
                prompt
            )

            # 4. Save erosion step comparison (optional)
            self._save_erosion_comparison(
                ground_truth, generated_image,
                gt_masks_eroded, gen_masks_eroded,
                sample_dir
            )

            logger.debug(f"Debug visualization saved for sample {sample_idx} (prompt: '{prompt}')")

        except Exception as e:
            logger.error(f"Error saving debug visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _save_comparison_visualization(
        self,
        gt_image: Image.Image,
        gen_image: Image.Image,
        gt_mask: np.ndarray,
        gen_mask: np.ndarray,
        gt_garment: Image.Image,
        gen_garment: Image.Image,
        sample_dir: Path,
        prompt: str
    ):
        """
        Save comparison visualization: Original -> Mask Overlay -> Extracted Garment

        Args:
            gt_image: GT original image
            gen_image: Generated original image
            gt_mask: GT mask
            gen_mask: Generated mask
            gt_garment: GT garment
            gen_garment: Generated garment
            sample_dir: Save directory
            prompt: SAM3 prompt
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # First row: Ground Truth
        # 1.1 Original image
        axes[0, 0].imshow(gt_image)
        axes[0, 0].set_title(f"GT Original\nPrompt: '{prompt}'", fontsize=10)
        axes[0, 0].axis('off')

        # 1.2 Mask overlay
        gt_overlay = self._create_mask_overlay(gt_image, gt_mask, color='green')
        axes[0, 1].imshow(gt_overlay)
        axes[0, 1].set_title("GT Mask Overlay", fontsize=10)
        axes[0, 1].axis('off')

        # 1.3 Extracted garment
        axes[0, 2].imshow(gt_garment)
        axes[0, 2].set_title("GT Garment (Used for Metrics)", fontsize=10)
        axes[0, 2].axis('off')

        # Second row: Generated
        # 2.1 Original image
        axes[1, 0].imshow(gen_image)
        axes[1, 0].set_title("Generated Original", fontsize=10)
        axes[1, 0].axis('off')

        # 2.2 Mask overlay
        gen_overlay = self._create_mask_overlay(gen_image, gen_mask, color='blue')
        axes[1, 1].imshow(gen_overlay)
        axes[1, 1].set_title("Gen Mask Overlay", fontsize=10)
        axes[1, 1].axis('off')

        # 2.3 Extracted garment
        axes[1, 2].imshow(gen_garment)
        axes[1, 2].set_title("Gen Garment (Used for Metrics)", fontsize=10)
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(sample_dir / "00_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _create_mask_overlay(
        self,
        image: Image.Image,
        mask: np.ndarray,
        color: str = 'red'
    ) -> np.ndarray:
        """
        Create mask overlay visualization

        Args:
            image: Original image
            mask: Binary mask
            color: Overlay color

        Returns:
            Overlaid image array
        """
        img_array = np.array(image.convert('RGB'))
        mask_norm = (mask > 127).astype(np.float32)

        # Color mapping
        color_map = {
            'red': [255, 0, 0],
            'green': [0, 255, 0],
            'blue': [0, 0, 255]
        }
        rgb = color_map.get(color, [255, 0, 0])

        # Create semi-transparent overlay
        overlay = img_array.copy().astype(np.float32)
        tint = np.zeros_like(overlay)
        tint[:, :, :] = rgb

        alpha = 0.3
        overlay = np.where(
            mask_norm[:, :, np.newaxis] > 0,
            overlay * (1 - alpha) + tint * alpha,
            overlay
        ).astype(np.uint8)

        return overlay

    def _save_erosion_comparison(
        self,
        gt_image: Image.Image,
        gen_image: Image.Image,
        gt_masks_eroded: List[np.ndarray],
        gen_masks_eroded: List[np.ndarray],
        sample_dir: Path
    ):
        """
        Save erosion step garment extraction comparison

        Args:
            gt_image: GT image
            gen_image: Generated image
            gt_masks_eroded: List of GT eroded masks
            gen_masks_eroded: List of generated image eroded masks
            sample_dir: Sample directory
        """
        import matplotlib.pyplot as plt

        n_scales = len(self.erosion_scales)
        fig, axes = plt.subplots(2, n_scales, figsize=(5*n_scales, 10))

        if n_scales == 1:
            axes = axes.reshape(2, 1)

        for idx, scale in enumerate(self.erosion_scales):
            # GT garment
            gt_garment = self.sam_segmenter.apply_mask(gt_image, gt_masks_eroded[idx])
            axes[0, idx].imshow(gt_garment)
            axes[0, idx].set_title(f"GT Erosion {scale}px", fontsize=10)
            axes[0, idx].axis('off')

            # Generated garment
            gen_garment = self.sam_segmenter.apply_mask(gen_image, gen_masks_eroded[idx])
            axes[1, idx].imshow(gen_garment)
            axes[1, idx].set_title(f"Gen Erosion {scale}px", fontsize=10)
            axes[1, idx].axis('off')

        plt.tight_layout()
        plt.savefig(sample_dir / "05_erosion_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _compute_metrics_for_scale(
        self,
        gt_full: Image.Image,
        gen_full: Image.Image,
        gt_garment: Image.Image,
        gen_garment: Image.Image,
        scale_name: str
    ) -> Dict:
        """Compute metrics for a single scale"""
        # Extract features
        features = self.extract_features([gt_garment, gen_garment])
        gt_feat = features[0]
        gen_feat = features[1]

        # Resize to the same size
        gt_array, gen_array = resize_to_match(gt_garment, gen_garment)

        # Compute metrics
        metrics = self.metrics_calculator.compute_all_metrics(
            gt_array,
            gen_array,
            gt_feat,
            gen_feat
        )

        metrics['scale'] = scale_name
        return metrics

    def _compute_average_metrics(self, scale_results: Dict) -> Dict:
        """Compute average metrics across all scales (filtering inf and None values)"""
        metric_names = ['cosine_similarity', 'mse', 'ssim', 'psnr', 'lpips']
        averages = {}

        for name in metric_names:
            values = []
            for key, scale_metrics in scale_results.items():
                if key != 'average' and scale_metrics.get(name) is not None:
                    value = scale_metrics[name]
                    # Filter inf, -inf, and NaN values
                    if not np.isinf(value) and not np.isnan(value):
                        values.append(value)

            averages[name] = np.mean(values) if values else None

        averages['scale'] = 'average'
        return averages

    def evaluate_batch(
        self,
        ground_truths: List[Image.Image],
        generated_images: List[Image.Image],
        show_progress: bool = True,
        source_paths: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Batch evaluation (supports multi-GPU multi-process acceleration)

        Args:
            ground_truths: List of ground truth target images
            generated_images: List of generated images
            show_progress: Whether to show progress
            source_paths: List of source image paths (used for SAM3 label lookup)

        Returns:
            List of evaluation results (filtering out None values, i.e., samples with segmentation failure)
        """
        if self.use_multi_gpu and len(ground_truths) > self.num_workers:
            # Use multi-GPU multi-process acceleration
            return self._evaluate_batch_multi_gpu(
                ground_truths, generated_images, show_progress, source_paths
            )
        else:
            # Single GPU serial processing
            return self._evaluate_batch_single_gpu(
                ground_truths, generated_images, show_progress, source_paths
            )

    def _evaluate_batch_single_gpu(
        self,
        ground_truths: List[Image.Image],
        generated_images: List[Image.Image],
        show_progress: bool = True,
        source_paths: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Single GPU batch evaluation (original implementation)

        Args:
            ground_truths: List of ground truth target images
            generated_images: List of generated images
            show_progress: Whether to show progress
            source_paths: List of source image paths

        Returns:
            List of evaluation results
        """
        results = []
        skipped_count = 0

        iterator = enumerate(zip(ground_truths, generated_images))
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(
                list(iterator),
                desc="Garment Evaluation (Single GPU)",
                unit="image"
            )

        for idx, (gt, generated) in iterator:
            source_path = None
            if source_paths and idx < len(source_paths):
                source_path = source_paths[idx]

            # Pass sample_idx for debugging
            metrics = self.evaluate_single(
                gt, generated,
                source_path=source_path,
                sample_idx=idx
            )

            # Only keep successfully evaluated samples (filter None)
            if metrics is not None:
                results.append(metrics)
            else:
                skipped_count += 1

        if skipped_count > 0:
            logger.warning(
                f"Skipped {skipped_count} samples due to segmentation failure. "
                f"Successfully evaluated: {len(results)}/{len(ground_truths)}"
            )

        return results

    def _evaluate_batch_multi_gpu(
        self,
        ground_truths: List[Image.Image],
        generated_images: List[Image.Image],
        show_progress: bool = True,
        source_paths: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Multi-GPU multi-process batch evaluation

        Args:
            ground_truths: List of ground truth target images
            generated_images: List of generated images
            show_progress: Whether to show progress
            source_paths: List of source image paths

        Returns:
            List of evaluation results
        """
        from torch.multiprocessing import Pool, set_start_method
        import torch.multiprocessing as mp

        # Set multi-process start method
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        num_samples = len(ground_truths)
        num_gpus = len(self.gpu_ids)

        logger.info(f"Starting multi-GPU evaluation: {num_samples} samples across {num_gpus} GPUs")

        # Split data into num_workers chunks
        chunk_size = (num_samples + self.num_workers - 1) // self.num_workers
        chunks = []

        for i in range(0, num_samples, chunk_size):
            end_idx = min(i + chunk_size, num_samples)
            chunk_data = {
                'ground_truths': ground_truths[i:end_idx],
                'generated_images': generated_images[i:end_idx],
                'source_paths': source_paths[i:end_idx] if source_paths else None,
                'start_idx': i,
                'gpu_id': self.gpu_ids[len(chunks) % num_gpus],  # Round-robin GPU assignment
                'init_params': self.init_params,
                'simplified_prompts': self.simplified_prompts,
            }
            chunks.append(chunk_data)

        logger.info(f"Created {len(chunks)} chunks for processing")

        # Use process pool for parallel processing
        results_all = []
        skipped_count = 0

        with Pool(processes=self.num_workers) as pool:
            # Import worker function
            from metrics.garment_evaluator_worker import evaluate_chunk_worker

            # Submit all tasks
            async_results = [
                pool.apply_async(evaluate_chunk_worker, (chunk,))
                for chunk in chunks
            ]

            # Collect results (with progress bar)
            if show_progress:
                from tqdm import tqdm
                with tqdm(total=num_samples, desc="Garment Evaluation (Multi-GPU)", unit="image") as pbar:
                    for async_result in async_results:
                        chunk_results, chunk_skipped = async_result.get()
                        results_all.extend(chunk_results)
                        skipped_count += chunk_skipped
                        pbar.update(len(chunk_results) + chunk_skipped)
            else:
                for async_result in async_results:
                    chunk_results, chunk_skipped = async_result.get()
                    results_all.extend(chunk_results)
                    skipped_count += chunk_skipped

        if skipped_count > 0:
            logger.warning(
                f"Skipped {skipped_count} samples due to segmentation failure. "
                f"Successfully evaluated: {len(results_all)}/{num_samples}"
            )

        logger.success(f"Multi-GPU evaluation completed: {len(results_all)} samples processed")

        return results_all

    def compute_summary(self, results_list: List[Dict]) -> Dict:
        """
        Compute summary statistics (filtering inf and None values)

        Args:
            results_list: List of results

        Returns:
            Summary statistics dictionary
        """
        if not results_list:
            return {}

        summary = {}

        # Get all scales
        scales = list(results_list[0].keys())

        for scale in scales:
            metric_names = ['cosine_similarity', 'mse', 'ssim', 'psnr', 'lpips']
            scale_summary = {}

            for name in metric_names:
                values = []
                for result in results_list:
                    if scale in result and result[scale].get(name) is not None:
                        value = result[scale][name]
                        # Filter inf and -inf values
                        if not np.isinf(value) and not np.isnan(value):
                            values.append(value)

                if values:
                    scale_summary[f'{name}_mean'] = np.mean(values)
                    scale_summary[f'{name}_std'] = np.std(values)
                    scale_summary[f'{name}_median'] = np.median(values)
                    scale_summary[f'{name}_valid_count'] = len(values)  # Record valid sample count
                else:
                    scale_summary[f'{name}_mean'] = None
                    scale_summary[f'{name}_std'] = None
                    scale_summary[f'{name}_median'] = None
                    scale_summary[f'{name}_valid_count'] = 0

            summary[scale] = scale_summary

        return summary

    def _default_metrics(self) -> Dict:
        """Return default metrics"""
        default = {
            'cosine_similarity': None,
            'mse': None,
            'ssim': None,
            'psnr': None,
            'lpips': None
        }

        results = {'scale_0': default.copy()}
        for idx in range(1, len(self.erosion_scales) + 1):
            results[f'scale_{idx}'] = default.copy()
        results['average'] = default.copy()

        return results
