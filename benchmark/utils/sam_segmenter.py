"""
SAM3 (Segment Anything Model 3) Segmenter
For segmenting garment regions in images based on text prompts
"""

import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger

try:
    from transformers import Sam3Processor, Sam3Model
    SAM3_AVAILABLE = True
except ImportError:
    logger.error("transformers SAM3 not available. Please install transformers>=4.36.0")
    SAM3_AVAILABLE = False

try:
    from utils.cluster_label_loader import ClusterLabelLoader
except ImportError:
    ClusterLabelLoader = None
    logger.warning("ClusterLabelLoader not available")


class SAMSegmenter:
    """SAM3 Garment Segmenter - Semantic segmentation based on text prompts"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        cluster_label_loader: Optional['ClusterLabelLoader'] = None,
        default_prompt: str = "clothing"
    ):
        """
        Initialize SAM3 segmenter

        Args:
            checkpoint_path: SAM3 model checkpoint path
            device: Compute device ('cuda' or 'cpu')
            threshold: Detection confidence threshold (0.3-0.6, recommended 0.5)
            mask_threshold: Mask binarization threshold (recommended 0.5)
            cluster_label_loader: Cluster label loader (for text prompts)
            default_prompt: Default prompt (used when cluster label is not found)
        """
        if not SAM3_AVAILABLE:
            raise ImportError(
                "SAM3 not available. Please install transformers>=4.36.0\n"
                "pip install transformers>=4.36.0"
            )

        self.device = device
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.cluster_label_loader = cluster_label_loader
        self.default_prompt = default_prompt

        logger.info(f"Loading SAM3 model from {checkpoint_path}...")

        # Load SAM3 model and processor
        self.model = Sam3Model.from_pretrained(
            checkpoint_path,
            local_files_only=True
        ).to(device)
        self.processor = Sam3Processor.from_pretrained(
            checkpoint_path,
            local_files_only=True
        )
        self.model.eval()

        logger.success(
            f"SAM3 model loaded successfully!\n"
            f"  - Threshold: {threshold}\n"
            f"  - Mask Threshold: {mask_threshold}\n"
            f"  - Default Prompt: '{default_prompt}'\n"
            f"  - Cluster Labels: {'Enabled' if cluster_label_loader else 'Disabled'}"
        )

    def _get_text_prompt(self, source_path: Optional[str]) -> str:
        """
        Get text prompt

        Args:
            source_path: Source image path

        Returns:
            Text prompt (cluster label or default prompt)
        """
        if source_path and self.cluster_label_loader:
            filename = Path(source_path).name
            label = self.cluster_label_loader.get_label_for_filename(filename)

            if label:
                logger.debug(f"Using cluster label: '{label}' for {filename}")
                return label
            else:
                logger.debug(f"No cluster label found for {filename}, using default prompt")

        # Fall back to default prompt
        return self.default_prompt

    def segment_garment(
        self,
        image: Image.Image,
        source_path: Optional[str] = None,
        return_largest: bool = True
    ) -> Optional[np.ndarray]:
        """
        Segment garment in image (with fallback mechanism)

        Args:
            image: Input image
            source_path: Source image path (for label lookup)
            return_largest: Whether to return only the highest confidence mask (kept for compatibility)

        Returns:
            Garment mask (binary image), shape [H, W], values 0-255
        """
        # Define fallback prompt list (from specific to generic)
        text_prompt = self._get_text_prompt(source_path)
        fallback_prompts = []

        # If using cluster label, add fallback options
        if text_prompt != self.default_prompt:
            fallback_prompts = [
                text_prompt,           # Original cluster label
                "clothing",            # Generic clothing
                "shirt",               # Shirt
                "top",                 # Top
                self.default_prompt    # Configured default prompt
            ]
        else:
            # If already using default prompt, only try a few generic options
            fallback_prompts = [
                self.default_prompt,
                "clothing",
                "shirt",
                "top"
            ]

        # Remove duplicates while preserving order
        seen = set()
        fallback_prompts = [x for x in fallback_prompts if not (x in seen or seen.add(x))]

        # Try each prompt
        for attempt_idx, prompt in enumerate(fallback_prompts):
            try:
                # 2. Prepare inputs
                img_rgb = image.convert('RGB')
                inputs = self.processor(
                    images=img_rgb,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)

                # 3. Run inference
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # 4. Post-process
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.threshold,
                    mask_threshold=self.mask_threshold,
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]

                # 5. Check if masks were found
                if len(results['masks']) == 0:
                    if attempt_idx < len(fallback_prompts) - 1:
                        logger.debug(f"No masks found for prompt: '{prompt}', trying fallback...")
                        continue
                    else:
                        logger.warning(f"No masks found after trying all prompts: {fallback_prompts}")
                        return None

                # Get mask with highest confidence
                best_idx = torch.argmax(results['scores']).item()
                mask = results['masks'][best_idx].cpu().numpy()
                best_score = results['scores'][best_idx].item()

                # Convert to uint8 [0, 255]
                mask_uint8 = (mask > 0).astype(np.uint8) * 255

                # Log successful prompt
                if attempt_idx > 0:
                    logger.info(
                        f"SAM3 succeeded with fallback prompt '{prompt}' "
                        f"(original: '{fallback_prompts[0]}'), "
                        f"found {len(results['masks'])} masks, score {best_score:.3f}"
                    )
                else:
                    logger.debug(
                        f"SAM3 found {len(results['masks'])} masks for '{prompt}', "
                        f"selected mask with score {best_score:.3f}"
                    )

                return mask_uint8

            except Exception as e:
                if attempt_idx < len(fallback_prompts) - 1:
                    logger.debug(f"Error with prompt '{prompt}': {e}, trying fallback...")
                    continue
                else:
                    logger.error(f"Error in SAM3 segmentation after all attempts: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return None

        return None

    def segment_batch(
        self,
        images: List[Image.Image],
        source_paths: Optional[List[str]] = None,
        return_largest: bool = True,
        show_progress: bool = True
    ) -> List[Optional[np.ndarray]]:
        """
        Batch segmentation

        Args:
            images: List of images
            source_paths: List of source image paths (for label lookup)
            return_largest: Kept for compatibility
            show_progress: Whether to show progress

        Returns:
            List of masks
        """
        results = []

        iterator = enumerate(images)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(
                list(iterator),
                desc="SAM3 Segmentation",
                unit="image"
            )

        for idx, img in iterator:
            source_path = None
            if source_paths and idx < len(source_paths):
                source_path = source_paths[idx]

            mask = self.segment_garment(
                img,
                source_path=source_path,
                return_largest=return_largest
            )
            results.append(mask)

        return results

    def apply_mask(
        self,
        image: Image.Image,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        Apply mask to image to extract foreground

        Args:
            image: Input image
            mask: Binary mask
            background_color: Background color (RGB)

        Returns:
            Image with extracted foreground
        """
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))

        # Normalize mask to [0, 1]
        mask_norm = (mask > 127).astype(np.float32)

        # Create background
        background = np.ones_like(img_array) * background_color

        # Blend foreground and background
        result = img_array * mask_norm[:, :, np.newaxis] + background * (1 - mask_norm[:, :, np.newaxis])

        return Image.fromarray(result.astype(np.uint8))

    def erode_mask(
        self,
        mask: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Erode mask

        Args:
            mask: Input mask
            kernel_size: Erosion kernel size (pixels)

        Returns:
            Eroded mask
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(mask, kernel, iterations=1)
        return eroded

    def multi_scale_erosion(
        self,
        mask: np.ndarray,
        scales: List[int] = [5, 10, 20]
    ) -> List[np.ndarray]:
        """
        Multi-scale erosion

        Args:
            mask: Input mask
            scales: List of erosion scales (pixels)

        Returns:
            List of masks at different scales
        """
        results = []

        for scale in scales:
            eroded = self.erode_mask(mask, kernel_size=scale)
            results.append(eroded)

        return results
