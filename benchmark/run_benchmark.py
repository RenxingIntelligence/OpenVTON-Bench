#!/usr/bin/env python3
"""
OpenVTON-Bench - Main Benchmark Script
A comprehensive virtual try-on evaluation benchmark system
"""

import os
import sys
from pathlib import Path

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add benchmark directory to sys.path for relative imports
benchmark_dir = Path(__file__).parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

import json
import yaml
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from loguru import logger
from PIL import Image
import torch

# Import evaluators
from metrics.vlm_evaluator import VLMEvaluator
from metrics.garment_evaluator import GarmentEvaluator
from metrics.all_evaluator import AllEvaluator
from metrics.pixel_evaluator import PixelEvaluator

# Import image processing utilities
from utils.image_processing import load_and_process_image
from utils.cluster_label_loader import ClusterLabelLoader


def load_config(config_path: str) -> Dict:
    """Load configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_test_data(jsonl_path: str) -> List[Dict]:
    """Load test dataset."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} test samples from {jsonl_path}")
    return data


def load_single_sample_images(
    gen_file: Path,
    test_data_map: Dict,
    base_image_dir: Optional[Path]
) -> Optional[Tuple[Image.Image, Image.Image, Image.Image, str, str]]:
    """
    Load all images for a single sample (multi-threaded worker function).

    Args:
        gen_file: Generated image file path
        test_data_map: Test data mapping dictionary
        base_image_dir: Base image directory

    Returns:
        (source_image, ground_truth, generated_image, filename, source_path) or None if failed
    """
    gen_stem = gen_file.stem

    # Find metadata
    if gen_stem not in test_data_map:
        return None

    entry = test_data_map[gen_stem]

    # Build source and target paths
    source_path = Path(entry['source'])
    target_path = Path(entry['target'])

    # If base_image_dir is provided, use it as base path
    if base_image_dir:
        if not source_path.is_absolute():
            source_path = base_image_dir / source_path
        if not target_path.is_absolute():
            target_path = base_image_dir / target_path

    # Check if files exist
    if not source_path.exists() or not target_path.exists():
        return None

    try:
        # Load three images
        source_img = load_and_process_image(str(source_path), target_size=1024)
        target_img = load_and_process_image(str(target_path), target_size=1024)
        gen_img = load_and_process_image(str(gen_file), target_size=1024)

        return (source_img, target_img, gen_img, gen_file.name, str(source_path))
    except Exception as e:
        logger.debug(f"Error loading images for {gen_file.name}: {e}")
        return None


def run_evaluation(
    config: Dict,
    eval_types: List[str] = None,
    models_filter: List[str] = None
):
    """
    Run evaluation.

    Args:
        config: Configuration dictionary
        eval_types: List of evaluation types to run ['vlm', 'garment', 'all', 'pixel']
        models_filter: List of models to evaluate (None means all)
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['data']['output_dir']) / f"{timestamp}_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Load test data (for metadata lookup)
    test_data = load_test_data(config['data']['test_jsonl'])

    # Create source filename to data entry mapping (for fast lookup)
    # Use stem (without extension) for matching, ignoring .jpg/.png differences
    test_data_map = {}
    for entry in test_data:
        source_stem = Path(entry['source']).stem  # Without extension
        test_data_map[source_stem] = entry

    logger.info(f"Loaded {len(test_data_map)} test samples metadata (matched by stem)")

    # Initialize ClusterLabelLoader for SAM3
    cluster_label_loader = None
    sam_config = config.get('models', {}).get('sam', {})
    label_mapping_path = sam_config.get(
        'cluster_label_mapping',
        './data/cluster_label_mapping.json'
    )
    visual_buckets_path = sam_config.get(
        'visual_buckets',
        './data/visual_buckets.json'
    )

    logger.info("Loading cluster label mappings for SAM3...")
    try:
        cluster_label_loader = ClusterLabelLoader(
            label_mapping_path=label_mapping_path,
            visual_buckets_path=visual_buckets_path
        )
        logger.success(
            f"Loaded cluster labels: {cluster_label_loader.num_clusters} clusters, "
            f"{cluster_label_loader.num_filenames} filenames"
        )
    except Exception as e:
        logger.error(f"Failed to load cluster labels: {e}")
        logger.warning("SAM3 will use default prompt only")
        cluster_label_loader = None

    # Get models to evaluate
    all_models = config['data']['generated_dirs']
    if models_filter:
        all_models = [m for m in all_models if m['name'] in models_filter]

    logger.info(f"Evaluating {len(all_models)} models: {[m['name'] for m in all_models]}")

    # Determine evaluation types to run
    if eval_types is None:
        eval_types = []
        if config['evaluation']['enable_vlm']:
            eval_types.append('vlm')
        if config['evaluation']['enable_garment']:
            eval_types.append('garment')
        if config['evaluation']['enable_all']:
            eval_types.append('all')
        if config['evaluation'].get('enable_pixel', False):
            eval_types.append('pixel')

    logger.info(f"Running evaluation types: {eval_types}")

    # Initialize evaluators
    evaluators = {}

    # Get GPU configuration
    gpu_ids_str = config.get('performance', {}).get('gpu_ids', '0')
    gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',') if x.strip()]
    num_gpus = len(gpu_ids)

    logger.info(f"Available GPUs: {gpu_ids} (total: {num_gpus})")

    # Pre-load DINOv3 model (if garment or all evaluation needed)
    shared_dinov3 = None
    if 'garment' in eval_types or 'all' in eval_types:
        logger.info("=" * 80)
        logger.info("Pre-loading shared DINOv3 model (used by garment & all evaluators)...")
        logger.info("=" * 80)

        from transformers import AutoModel, AutoImageProcessor
        import time

        dinov3_path = config['models']['dinov3']['path']
        device = f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() and gpu_ids else 'cpu'

        start_time = time.time()

        processor = AutoImageProcessor.from_pretrained(dinov3_path)
        model = AutoModel.from_pretrained(
            dinov3_path,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True
        ).to(device, dtype=torch.float32)  # Use float32 to avoid NaN issues
        model.eval()

        load_time = time.time() - start_time
        shared_dinov3 = (processor, model)

        logger.success(f"DINOv3 model loaded in {load_time:.2f}s and will be shared!")
        logger.info("=" * 80 + "\n")

    if 'vlm' in eval_types:
        logger.info("Initializing VLM Evaluator...")
        evaluators['vlm'] = VLMEvaluator(
            api_key=config['models']['qwen_vl']['api_key'],
            base_url=config['models']['qwen_vl'].get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
            model_name=config['models']['qwen_vl'].get('model_name', 'qwen-vl-plus'),
            max_tokens=config['models']['qwen_vl'].get('max_tokens', 512),
            max_workers=config['models']['qwen_vl'].get('max_workers', 10)
        )

    if 'garment' in eval_types:
        logger.info("Initializing Garment Evaluator...")
        sam_config = config['models']['sam']

        # Create debug directory (save to current evaluation directory)
        debug_dir = output_dir / 'debug' / 'garment'

        # Simplified prompts mapping file path
        simplified_prompts_path = Path(__file__).parent.parent / "data_process" / "simplified_prompts.json"

        evaluators['garment'] = GarmentEvaluator(
            dinov3_path=config['models']['dinov3']['path'],
            sam_checkpoint=sam_config['checkpoint'],
            erosion_scales=config['evaluation']['erosion_scales'],
            threshold=sam_config.get('threshold', 0.5),
            mask_threshold=sam_config.get('mask_threshold', 0.5),
            cluster_label_loader=cluster_label_loader,
            default_prompt=sam_config.get('default_prompt', 'clothing'),
            preloaded_model=shared_dinov3,
            simplified_prompts_path=str(simplified_prompts_path) if simplified_prompts_path.exists() else None,
            debug_dir=str(debug_dir),  # Save to evaluation directory debug/garment/
            save_debug_samples=3,  # Save first 3 samples
            gpu_ids=gpu_ids,  # Pass GPU IDs for multi-GPU acceleration
            num_workers=config.get('performance', {}).get('num_workers', 4)  # Multi-process worker count
        )

    if 'all' in eval_types:
        logger.info("Initializing All Evaluator...")
        evaluators['all'] = AllEvaluator(
            dinov3_path=config['models']['dinov3']['path'],
            image_size=config['models']['dinov3']['image_size'],
            preloaded_model=shared_dinov3  # Pass pre-loaded model
        )

    if 'pixel' in eval_types:
        logger.info("Initializing Pixel Evaluator...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Read FID configuration
        enable_fid = config.get('evaluation', {}).get('pixel', {}).get('enable_fid', True)
        evaluators['pixel'] = PixelEvaluator(device=device, enable_fid=enable_fid)

    # Evaluate each model
    all_results = {}

    for model_config in all_models:
        model_name = model_config['name']
        model_path = Path(model_config['path'])

        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"{'='*80}\n")

        model_results = {
            'model_name': model_name,
            'model_path': str(model_path),
            'description': model_config.get('description', ''),
            'num_samples': 0,
            'vlm_scores': [],
            'garment_metrics': [],
            'all_metrics': [],
            'pixel_metrics': []
        }

        # New logic: scan all images in generated directory
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            continue

        # Get all generated image files (support jpg, jpeg, png)
        generated_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            generated_files.extend(list(model_path.glob(ext)))

        logger.info(f"Found {len(generated_files)} generated images in {model_path}")

        # Prepare image paths
        source_images = []
        ground_truths = []
        generated_images = []
        valid_filenames = []
        source_paths_list = []  # Track source paths for SAM3

        base_image_dir = Path(config['data']['base_image_dir']) if config['data'].get('base_image_dir') else None

        # Get thread count (read from config, default to 2x CPU cores)
        max_workers = config.get('performance', {}).get('num_workers', os.cpu_count() * 2)
        logger.info(f"Loading images with {max_workers} threads...")

        # Use multi-threading for parallel image loading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(load_single_sample_images, gen_file, test_data_map, base_image_dir): gen_file
                for gen_file in generated_files
            }

            # Collect results (show progress with tqdm)
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Loading images"):
                result = future.result()
                if result is not None:
                    source_img, target_img, gen_img, filename, source_path = result
                    source_images.append(source_img)
                    ground_truths.append(target_img)
                    generated_images.append(gen_img)
                    valid_filenames.append(filename)
                    source_paths_list.append(source_path)

        model_results['num_samples'] = len(valid_filenames)
        logger.info(f"Successfully loaded {len(valid_filenames)} valid samples for {model_name}")

        if len(valid_filenames) == 0:
            logger.warning(f"No valid samples found for {model_name}, skipping...")
            continue

        # VLM evaluation
        if 'vlm' in eval_types:
            logger.info("\n--- Running VLM Evaluation ---")
            vlm_results, vlm_skipped = evaluators['vlm'].evaluate_batch(
                source_images,
                ground_truths,
                generated_images,
                show_progress=True
            )
            model_results['vlm_scores'] = vlm_results
            model_results['vlm_skipped_count'] = vlm_skipped

            # Calculate average scores
            avg_scores = evaluators['vlm'].compute_average_scores(vlm_results)
            model_results['vlm_average'] = avg_scores
            logger.success(f"VLM Average Total Score: {avg_scores.get('total_score', 0):.2f}/5.0")
            logger.info(f"VLM samples evaluated: {len(vlm_results)}, skipped (content filtered): {vlm_skipped}")

        # Garment evaluation
        if 'garment' in eval_types:
            logger.info("\n--- Running Garment-based Evaluation ---")
            garment_results = evaluators['garment'].evaluate_batch(
                ground_truths,
                generated_images,
                show_progress=True,
                source_paths=source_paths_list  # Pass source paths
            )
            model_results['garment_metrics'] = garment_results

            # Calculate summary
            garment_summary = evaluators['garment'].compute_summary(garment_results)
            model_results['garment_summary'] = garment_summary
            logger.success("Garment evaluation completed!")

        # All evaluation
        if 'all' in eval_types:
            logger.info("\n--- Running All-based Evaluation ---")
            all_results_list = evaluators['all'].evaluate_batch(
                ground_truths,
                generated_images,
                show_progress=True
            )
            model_results['all_metrics'] = all_results_list

            # Calculate average
            all_avg = evaluators['all'].compute_average_metrics(all_results_list)
            model_results['all_average'] = all_avg
            logger.success("All-based evaluation completed!")

        # Pixel evaluation
        if 'pixel' in eval_types:
            logger.info("\n--- Running Pixel-based Evaluation ---")
            pixel_results = evaluators['pixel'].evaluate_batch(
                ground_truths,
                generated_images,
                show_progress=True
            )
            # pixel_results is now a dict containing per_sample_metrics and fid
            model_results['pixel_metrics'] = pixel_results.get('per_sample_metrics', [])
            model_results['pixel_fid'] = pixel_results.get('fid')

            # Calculate average (compute_average_metrics now accepts dict)
            pixel_avg = evaluators['pixel'].compute_average_metrics(pixel_results)
            model_results['pixel_average'] = pixel_avg

            # Print FID score separately
            fid_score = pixel_results.get('fid')
            if fid_score is not None:
                logger.success(f"Pixel-based FID Score: {fid_score:.4f}")
            logger.success("Pixel-based evaluation completed!")

        # Save model results
        model_output_dir = output_dir / "per_model" / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        with open(model_output_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)

        logger.success(f"Results saved for {model_name}")

        all_results[model_name] = model_results

    # Generate summary report
    logger.info("\n" + "="*80)
    logger.info("Generating Summary Report")
    logger.info("="*80 + "\n")

    summary = generate_summary(all_results, eval_types)

    # Save summary results
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Generate text report
    generate_text_report(summary, output_dir / "summary.txt")

    # Generate visualizations
    if config['reporting']['visualizations']:
        logger.info("Generating visualizations...")
        from utils.report_generator import ReportGenerator
        reporter = ReportGenerator(output_dir / "visualizations")
        reporter.generate_all_visualizations(all_results, eval_types)

    logger.success(f"\n{'='*80}")
    logger.success(f"Evaluation completed!")
    logger.success(f"Results saved to: {output_dir}")
    logger.success(f"{'='*80}\n")


def generate_summary(all_results: Dict, eval_types: List[str]) -> Dict:
    """Generate summary statistics."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_models': len(all_results),
        'models': list(all_results.keys()),
        'evaluation_types': eval_types,
        'model_rankings': {}
    }

    # VLM ranking
    if 'vlm' in eval_types:
        vlm_scores = {}
        for model_name, results in all_results.items():
            if 'vlm_average' in results:
                vlm_scores[model_name] = results['vlm_average'].get('total_score', 0)

        summary['model_rankings']['vlm'] = sorted(
            vlm_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

    # Garment ranking (based on SSIM)
    if 'garment' in eval_types:
        garment_scores = {}
        for model_name, results in all_results.items():
            if 'garment_summary' in results and 'average' in results['garment_summary']:
                ssim = results['garment_summary']['average'].get('ssim_mean')
                garment_scores[model_name] = ssim if ssim is not None else 0

        summary['model_rankings']['garment_ssim'] = sorted(
            garment_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

    # All ranking (based on SSIM)
    if 'all' in eval_types:
        all_scores = {}
        for model_name, results in all_results.items():
            if 'all_average' in results:
                ssim = results['all_average'].get('ssim')
                all_scores[model_name] = ssim if ssim is not None else 0

        summary['model_rankings']['all_ssim'] = sorted(
            all_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

    # Pixel ranking (based on SSIM)
    if 'pixel' in eval_types:
        pixel_scores = {}
        for model_name, results in all_results.items():
            if 'pixel_average' in results:
                ssim = results['pixel_average'].get('ssim')
                pixel_scores[model_name] = ssim if ssim is not None else 0

        summary['model_rankings']['pixel_ssim'] = sorted(
            pixel_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

    return summary


def generate_text_report(summary: Dict, output_path: Path):
    """Generate text format report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("OpenVTON-Bench Evaluation Summary Report\n")
        f.write("="*80 + "\n\n")

        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Number of Models: {summary['num_models']}\n")
        f.write(f"Models: {', '.join(summary['models'])}\n")
        f.write(f"Evaluation Types: {', '.join(summary['evaluation_types'])}\n\n")

        f.write("="*80 + "\n")
        f.write("Model Rankings\n")
        f.write("="*80 + "\n\n")

        for eval_type, rankings in summary['model_rankings'].items():
            f.write(f"{eval_type.upper()} Rankings:\n")
            f.write("-" * 40 + "\n")
            for rank, (model, score) in enumerate(rankings, 1):
                f.write(f"  {rank}. {model}: {score:.4f}\n")
            f.write("\n")

    logger.info(f"Text report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="OpenVTON-Bench Evaluation")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--eval_type", type=str, nargs='+',
                        choices=['vlm', 'garment', 'all', 'pixel'],
                        help="Evaluation types to run (default: all enabled in config)")
    parser.add_argument("--models", type=str, nargs='+',
                        help="Specific models to evaluate (default: all)")

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Run evaluation
    run_evaluation(
        config,
        eval_types=args.eval_type,
        models_filter=args.models
    )


if __name__ == "__main__":
    main()
