#!/usr/bin/env python3
"""
Correlation Analysis Script

Compares the correlation between:
1. DINOv2-based metrics (from 'all' evaluation)
2. Pixel-based metrics (from 'pixel' evaluation)
3. Human subjective ratings

Determines which metric type better predicts human judgment.
"""

import os
import sys
from pathlib import Path

# Add benchmark directory to sys.path
benchmark_dir = Path(__file__).parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from loguru import logger
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


def load_evaluation_results(result_dir: Path, model_name: str) -> Dict:
    """
    Load evaluation results for a specific model.

    Args:
        result_dir: Evaluation results directory
        model_name: Model name

    Returns:
        Evaluation results dictionary
    """
    result_file = result_dir / "per_model" / model_name / "results.json"

    if not result_file.exists():
        logger.error(f"Results file not found: {result_file}")
        return None

    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    logger.info(f"Loaded evaluation results for {model_name}")
    return results


def load_human_ratings(human_ratings_file: Path) -> pd.DataFrame:
    """
    Load human subjective ratings.

    Expected format (JSON or JSONL):
    [
        {
            "image_id": "sample_001.png",
            "model": "model_a",
            "overall_quality": 4.5,  # Overall quality (1-5)
            "garment_fidelity": 4.2,  # Garment restoration fidelity (1-5)
            "realism": 4.3,  # Realism (1-5)
            ...
        },
        ...
    ]

    Args:
        human_ratings_file: Path to human ratings file

    Returns:
        DataFrame containing human ratings
    """
    if not human_ratings_file.exists():
        logger.error(f"Human ratings file not found: {human_ratings_file}")
        return None

    # Support JSON or JSONL format
    if human_ratings_file.suffix == '.jsonl':
        # JSONL format
        ratings = []
        with open(human_ratings_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    ratings.append(json.loads(line))
        df = pd.DataFrame(ratings)
    else:
        # JSON format
        with open(human_ratings_file, 'r', encoding='utf-8') as f:
            ratings = json.load(f)
        df = pd.DataFrame(ratings)

    logger.info(f"Loaded {len(df)} human ratings")
    return df


def extract_metrics_per_sample(eval_results: Dict, eval_type: str) -> Dict[str, Dict]:
    """
    Extract per-sample metrics from evaluation results.

    Args:
        eval_results: Evaluation results dictionary
        eval_type: Evaluation type ('all' or 'pixel')

    Returns:
        {image_filename: {metric_name: value, ...}, ...}
    """
    metrics_key = f'{eval_type}_metrics'

    if metrics_key not in eval_results or not eval_results[metrics_key]:
        logger.warning(f"No {eval_type} metrics found in evaluation results")
        return {}

    metrics_list = eval_results[metrics_key]

    # Note: evaluation results may not store filenames, need to match by index
    # Assume evaluation results order matches file list order
    sample_metrics = {}
    for idx, metrics in enumerate(metrics_list):
        # Use index as key, need to match with filename later
        sample_metrics[idx] = metrics

    return sample_metrics


def compute_correlation(
    metric_values: List[float],
    human_ratings: List[float],
    metric_name: str
) -> Tuple[float, float, float, float]:
    """
    Compute correlation between metrics and human ratings.

    Args:
        metric_values: List of metric values
        human_ratings: List of human ratings
        metric_name: Metric name

    Returns:
        (pearson_r, pearson_p, spearman_r, spearman_p)
    """
    # Filter out None values
    valid_pairs = [(m, h) for m, h in zip(metric_values, human_ratings)
                   if m is not None and not np.isnan(m) and h is not None and not np.isnan(h)]

    if len(valid_pairs) < 3:
        logger.warning(f"Not enough valid pairs for {metric_name}: {len(valid_pairs)}")
        return (None, None, None, None)

    metric_vals, human_vals = zip(*valid_pairs)

    try:
        pearson_r, pearson_p = pearsonr(metric_vals, human_vals)
        spearman_r, spearman_p = spearmanr(metric_vals, human_vals)

        return (pearson_r, pearson_p, spearman_r, spearman_p)
    except Exception as e:
        logger.error(f"Error computing correlation for {metric_name}: {e}")
        return (None, None, None, None)


def analyze_correlations(
    result_dir: Path,
    human_ratings_file: Path,
    model_name: str,
    output_dir: Path
):
    """
    Main analysis function: compute and compare DINOv2-based and Pixel-based metric correlations.

    Args:
        result_dir: Evaluation results directory
        human_ratings_file: Human ratings file
        model_name: Model name
        output_dir: Output directory
    """
    logger.info("=" * 80)
    logger.info("Starting Correlation Analysis")
    logger.info("=" * 80)

    # 1. Load evaluation results
    eval_results = load_evaluation_results(result_dir, model_name)
    if eval_results is None:
        return

    # 2. Load human ratings
    human_ratings_df = load_human_ratings(human_ratings_file)
    if human_ratings_df is None:
        return

    # Filter ratings for this model
    if 'model' in human_ratings_df.columns:
        human_ratings_df = human_ratings_df[human_ratings_df['model'] == model_name]

    logger.info(f"Found {len(human_ratings_df)} human ratings for model {model_name}")

    if len(human_ratings_df) == 0:
        logger.error(f"No human ratings found for model {model_name}")
        return

    # 3. Extract metrics
    all_metrics = extract_metrics_per_sample(eval_results, 'all')
    pixel_metrics = extract_metrics_per_sample(eval_results, 'pixel')

    if not all_metrics and not pixel_metrics:
        logger.error("No metrics found in evaluation results")
        return

    # 4. Define metrics and human rating dimensions to analyze
    metric_names = ['cosine_similarity', 'mse', 'ssim', 'psnr', 'lpips']

    # Human rating dimensions (adjust based on actual data)
    human_dimensions = []
    for col in human_ratings_df.columns:
        if col not in ['image_id', 'model', 'annotator_id']:
            human_dimensions.append(col)

    if not human_dimensions:
        logger.error("No human rating dimensions found in the data")
        return

    logger.info(f"Analyzing metrics: {metric_names}")
    logger.info(f"Against human dimensions: {human_dimensions}")

    # 5. Compute correlations
    correlation_results = {
        'dinov2': {},  # DINOv2-based (all)
        'pixel': {}    # Pixel-based
    }

    for human_dim in human_dimensions:
        correlation_results['dinov2'][human_dim] = {}
        correlation_results['pixel'][human_dim] = {}

        logger.info(f"\n--- Analyzing correlation with: {human_dim} ---")

        for metric_name in metric_names:
            # DINOv2-based metrics
            if all_metrics:
                dinov2_values = []
                human_values = []

                for idx, row in human_ratings_df.iterrows():
                    # Need to match image_id with evaluation results
                    # Assume human_ratings has image_id field
                    if idx < len(all_metrics):
                        metric_val = all_metrics.get(idx, {}).get(metric_name)
                        human_val = row.get(human_dim)

                        if metric_val is not None and human_val is not None:
                            dinov2_values.append(metric_val)
                            human_values.append(human_val)

                if dinov2_values:
                    pearson_r, pearson_p, spearman_r, spearman_p = compute_correlation(
                        dinov2_values, human_values, f"DINOv2-{metric_name}"
                    )

                    correlation_results['dinov2'][human_dim][metric_name] = {
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'n_samples': len(dinov2_values)
                    }

                    if pearson_r is not None:
                        logger.info(f"  DINOv2-{metric_name}: Pearson={pearson_r:.4f} (p={pearson_p:.4f}), "
                                  f"Spearman={spearman_r:.4f} (p={spearman_p:.4f})")

            # Pixel-based metrics
            if pixel_metrics and metric_name != 'cosine_similarity':  # pixel doesn't have cosine_similarity
                pixel_values = []
                human_values = []

                for idx, row in human_ratings_df.iterrows():
                    if idx < len(pixel_metrics):
                        metric_val = pixel_metrics.get(idx, {}).get(metric_name)
                        human_val = row.get(human_dim)

                        if metric_val is not None and human_val is not None:
                            pixel_values.append(metric_val)
                            human_values.append(human_val)

                if pixel_values:
                    pearson_r, pearson_p, spearman_r, spearman_p = compute_correlation(
                        pixel_values, human_values, f"Pixel-{metric_name}"
                    )

                    correlation_results['pixel'][human_dim][metric_name] = {
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'n_samples': len(pixel_values)
                    }

                    if pearson_r is not None:
                        logger.info(f"  Pixel-{metric_name}: Pearson={pearson_r:.4f} (p={pearson_p:.4f}), "
                                  f"Spearman={spearman_r:.4f} (p={spearman_p:.4f})")

    # 6. Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{model_name}_correlation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(correlation_results, f, indent=2, ensure_ascii=False)

    logger.success(f"Correlation results saved to {results_file}")

    # 7. Generate visualizations
    generate_visualizations(correlation_results, human_dimensions, output_dir, model_name)

    # 8. Generate comparison report
    generate_comparison_report(correlation_results, human_dimensions, output_dir, model_name)


def generate_visualizations(
    correlation_results: Dict,
    human_dimensions: List[str],
    output_dir: Path,
    model_name: str
):
    """
    Generate correlation visualization charts.

    Args:
        correlation_results: Correlation results
        human_dimensions: List of human rating dimensions
        output_dir: Output directory
        model_name: Model name
    """
    logger.info("\nGenerating visualizations...")

    # Generate comparison chart for each human dimension
    for human_dim in human_dimensions:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Correlation with Human "{human_dim}" - {model_name}', fontsize=14, fontweight='bold')

        # DINOv2-based correlations
        dinov2_corrs = correlation_results['dinov2'].get(human_dim, {})
        dinov2_metrics = []
        dinov2_pearson = []
        dinov2_spearman = []

        for metric, values in dinov2_corrs.items():
            if values.get('pearson_r') is not None:
                dinov2_metrics.append(f"DINOv2\n{metric}")
                dinov2_pearson.append(abs(values['pearson_r']))
                dinov2_spearman.append(abs(values['spearman_r']))

        # Pixel-based correlations
        pixel_corrs = correlation_results['pixel'].get(human_dim, {})
        pixel_metrics = []
        pixel_pearson = []
        pixel_spearman = []

        for metric, values in pixel_corrs.items():
            if values.get('pearson_r') is not None:
                pixel_metrics.append(f"Pixel\n{metric}")
                pixel_pearson.append(abs(values['pearson_r']))
                pixel_spearman.append(abs(values['spearman_r']))

        # Plot Pearson correlations
        all_metrics = dinov2_metrics + pixel_metrics
        all_pearson = dinov2_pearson + pixel_pearson
        colors = ['#3498db'] * len(dinov2_metrics) + ['#e74c3c'] * len(pixel_metrics)

        axes[0].bar(range(len(all_metrics)), all_pearson, color=colors)
        axes[0].set_xticks(range(len(all_metrics)))
        axes[0].set_xticklabels(all_metrics, rotation=45, ha='right', fontsize=9)
        axes[0].set_ylabel('|Pearson Correlation|')
        axes[0].set_title('Pearson Correlation (Absolute Values)')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)

        # Plot Spearman correlations
        all_spearman = dinov2_spearman + pixel_spearman

        axes[1].bar(range(len(all_metrics)), all_spearman, color=colors)
        axes[1].set_xticks(range(len(all_metrics)))
        axes[1].set_xticklabels(all_metrics, rotation=45, ha='right', fontsize=9)
        axes[1].set_ylabel('|Spearman Correlation|')
        axes[1].set_title('Spearman Correlation (Absolute Values)')
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='DINOv2-based'),
            Patch(facecolor='#e74c3c', label='Pixel-based')
        ]
        axes[1].legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        output_file = output_dir / f"{model_name}_{human_dim}_correlation.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved visualization: {output_file}")


def generate_comparison_report(
    correlation_results: Dict,
    human_dimensions: List[str],
    output_dir: Path,
    model_name: str
):
    """
    Generate comparison report summarizing which metric type is better.

    Args:
        correlation_results: Correlation results
        human_dimensions: List of human rating dimensions
        output_dir: Output directory
        model_name: Model name
    """
    logger.info("\nGenerating comparison report...")

    report_file = output_dir / f"{model_name}_comparison_report.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Correlation Analysis Report - {model_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Comparison: DINOv2-based Metrics vs Pixel-based Metrics\n")
        f.write("Against Human Subjective Ratings\n\n")

        for human_dim in human_dimensions:
            f.write("-" * 80 + "\n")
            f.write(f"Human Dimension: {human_dim}\n")
            f.write("-" * 80 + "\n\n")

            dinov2_corrs = correlation_results['dinov2'].get(human_dim, {})
            pixel_corrs = correlation_results['pixel'].get(human_dim, {})

            # Find best correlation
            best_dinov2_metric = None
            best_dinov2_corr = 0

            for metric, values in dinov2_corrs.items():
                if values.get('pearson_r') is not None:
                    abs_corr = abs(values['pearson_r'])
                    if abs_corr > best_dinov2_corr:
                        best_dinov2_corr = abs_corr
                        best_dinov2_metric = metric

            best_pixel_metric = None
            best_pixel_corr = 0

            for metric, values in pixel_corrs.items():
                if values.get('pearson_r') is not None:
                    abs_corr = abs(values['pearson_r'])
                    if abs_corr > best_pixel_corr:
                        best_pixel_corr = abs_corr
                        best_pixel_metric = metric

            f.write(f"Best DINOv2-based Metric:\n")
            if best_dinov2_metric:
                f.write(f"  {best_dinov2_metric}: |r| = {best_dinov2_corr:.4f}\n")
            else:
                f.write(f"  N/A\n")

            f.write(f"\nBest Pixel-based Metric:\n")
            if best_pixel_metric:
                f.write(f"  {best_pixel_metric}: |r| = {best_pixel_corr:.4f}\n")
            else:
                f.write(f"  N/A\n")

            f.write(f"\n")

            # Determine which is better
            if best_dinov2_corr > 0 and best_pixel_corr > 0:
                if best_dinov2_corr > best_pixel_corr:
                    diff = best_dinov2_corr - best_pixel_corr
                    f.write(f"Winner: DINOv2-based (+{diff:.4f})\n")
                    f.write(f"DINOv2-based metrics show {diff:.2%} stronger correlation with {human_dim}\n")
                elif best_pixel_corr > best_dinov2_corr:
                    diff = best_pixel_corr - best_dinov2_corr
                    f.write(f"Winner: Pixel-based (+{diff:.4f})\n")
                    f.write(f"Pixel-based metrics show {diff:.2%} stronger correlation with {human_dim}\n")
                else:
                    f.write(f"Tie: Both approaches show similar correlation\n")

            f.write("\n\n")

        # Summary
        f.write("=" * 80 + "\n")
        f.write("Overall Summary\n")
        f.write("=" * 80 + "\n\n")

        dinov2_wins = 0
        pixel_wins = 0

        for human_dim in human_dimensions:
            dinov2_corrs = correlation_results['dinov2'].get(human_dim, {})
            pixel_corrs = correlation_results['pixel'].get(human_dim, {})

            best_dinov2 = max([abs(v.get('pearson_r', 0)) for v in dinov2_corrs.values()
                              if v.get('pearson_r') is not None] or [0])
            best_pixel = max([abs(v.get('pearson_r', 0)) for v in pixel_corrs.values()
                            if v.get('pearson_r') is not None] or [0])

            if best_dinov2 > best_pixel:
                dinov2_wins += 1
            elif best_pixel > best_dinov2:
                pixel_wins += 1

        f.write(f"DINOv2-based metrics win: {dinov2_wins}/{len(human_dimensions)} dimensions\n")
        f.write(f"Pixel-based metrics win: {pixel_wins}/{len(human_dimensions)} dimensions\n\n")

        if dinov2_wins > pixel_wins:
            f.write("Conclusion: DINOv2-based metrics show overall stronger correlation with human ratings.\n")
        elif pixel_wins > dinov2_wins:
            f.write("Conclusion: Pixel-based metrics show overall stronger correlation with human ratings.\n")
        else:
            f.write("Conclusion: Both approaches show comparable correlation with human ratings.\n")

    logger.success(f"Comparison report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze correlation between metrics and human ratings")

    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to evaluation results directory (contains per_model/)")
    parser.add_argument("--human_ratings", type=str, required=True,
                        help="Path to human ratings file (JSON or JSONL)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name to analyze")
    parser.add_argument("--output_dir", type=str, default="./correlation_analysis",
                        help="Output directory for analysis results")

    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    human_ratings_file = Path(args.human_ratings)
    output_dir = Path(args.output_dir)

    analyze_correlations(
        result_dir=result_dir,
        human_ratings_file=human_ratings_file,
        model_name=args.model,
        output_dir=output_dir
    )

    logger.success("\n" + "=" * 80)
    logger.success("Correlation analysis completed!")
    logger.success(f"Results saved to: {output_dir}")
    logger.success("=" * 80 + "\n")


if __name__ == "__main__":
    main()
