"""
Garment Evaluator Multi-GPU Worker
Multi-process worker function (must be in separate module to avoid circular imports)
"""

import torch
from typing import Dict, List
from loguru import logger
from PIL import Image


def evaluate_chunk_worker(chunk_data: Dict) -> tuple:
    """
    Multi-process worker function: process a data chunk.

    Args:
        chunk_data: Dictionary containing data and configuration

    Returns:
        (results, skipped_count) tuple
    """
    gpu_id = chunk_data['gpu_id']
    ground_truths = chunk_data['ground_truths']
    generated_images = chunk_data['generated_images']
    source_paths = chunk_data['source_paths']
    init_params = chunk_data['init_params']
    simplified_prompts = chunk_data['simplified_prompts']

    # Set GPU for current process
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)

    logger.info(f"Worker started on GPU {gpu_id}: processing {len(ground_truths)} samples")

    # Create evaluator instance in current process
    from metrics.garment_evaluator import GarmentEvaluator

    evaluator = GarmentEvaluator(
        dinov3_path=init_params['dinov3_path'],
        sam_checkpoint=init_params['sam_checkpoint'],
        device=device,
        image_size=init_params['image_size'],
        erosion_scales=init_params['erosion_scales'],
        threshold=init_params['threshold'],
        mask_threshold=init_params['mask_threshold'],
        cluster_label_loader=init_params['cluster_label_loader'],
        default_prompt=init_params['default_prompt'],
        simplified_prompts_path=init_params['simplified_prompts_path'],
        debug_dir=None,  # Disable debug in multi-process mode
        save_debug_samples=0,
        gpu_ids=[gpu_id],  # Single GPU
        num_workers=1,  # Disable nested multi-process
    )

    # Manually set simplified_prompts (avoid repeated loading)
    evaluator.simplified_prompts = simplified_prompts

    # Process samples
    results = []
    skipped_count = 0

    for idx, (gt, generated) in enumerate(zip(ground_truths, generated_images)):
        source_path = None
        if source_paths and idx < len(source_paths):
            source_path = source_paths[idx]

        try:
            metrics = evaluator.evaluate_single(
                gt, generated,
                source_path=source_path,
                sample_idx=None  # Disable debug output
            )

            if metrics is not None:
                results.append(metrics)
            else:
                skipped_count += 1

        except Exception as e:
            logger.error(f"Error evaluating sample {idx} on GPU {gpu_id}: {e}")
            skipped_count += 1

    logger.info(f"Worker on GPU {gpu_id} completed: {len(results)} successful, {skipped_count} skipped")

    return (results, skipped_count)
