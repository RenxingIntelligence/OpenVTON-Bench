"""
Report Generator
Generates visualization charts and HTML reports for evaluation results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from loguru import logger

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


class ReportGenerator:
    """Report Generator"""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_visualizations(
        self,
        all_results: Dict,
        eval_types: List[str]
    ):
        """
        Generate all visualization charts

        Args:
            all_results: Results from all models
            eval_types: List of evaluation types
        """
        if 'vlm' in eval_types:
            self.plot_vlm_radar(all_results)
            self.plot_vlm_heatmap(all_results)

        if 'garment' in eval_types:
            self.plot_garment_metrics(all_results)

        if 'all' in eval_types:
            self.plot_all_metrics(all_results)

        # Comprehensive comparison
        self.plot_model_comparison(all_results, eval_types)

        logger.success(f"All visualizations saved to {self.output_dir}")

    def plot_vlm_radar(self, all_results: Dict):
        """Plot VLM score radar chart"""
        fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 6))

        if len(all_results) == 1:
            axes = [axes]

        dimensions = [
            'background_consistency',
            'person_consistency',
            'garment_texture',
            'garment_shape',
            'overall_realism'
        ]

        dim_labels = [
            'Background',
            'Person',
            'Texture',
            'Shape',
            'Realism'
        ]

        for idx, (model_name, results) in enumerate(all_results.items()):
            if 'vlm_average' not in results:
                continue

            avg_scores = results['vlm_average']
            values = [avg_scores.get(dim, 3) for dim in dimensions]

            # Close radar chart
            values += values[:1]

            angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
            angles += angles[:1]

            ax = axes[idx]
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dim_labels)
            ax.set_ylim(0, 5)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_title(f'{model_name}\nTotal: {avg_scores.get("total_score", 0):.2f}/5.0',
                         fontsize=14, fontweight='bold')
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / "vlm_radar.png", bbox_inches='tight')
        plt.close()

        logger.info("VLM radar plot saved")

    def plot_vlm_heatmap(self, all_results: Dict):
        """Plot VLM score heatmap"""
        models = list(all_results.keys())
        dimensions = [
            'Background',
            'Person',
            'Texture',
            'Shape',
            'Realism'
        ]

        dim_keys = [
            'background_consistency',
            'person_consistency',
            'garment_texture',
            'garment_shape',
            'overall_realism'
        ]

        # Prepare data
        data = []
        for model in models:
            if 'vlm_average' in all_results[model]:
                avg = all_results[model]['vlm_average']
                row = [avg.get(key, 3) for key in dim_keys]
                data.append(row)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.8)))
        sns.heatmap(
            data,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            vmin=1,
            vmax=5,
            xticklabels=dimensions,
            yticklabels=models,
            cbar_kws={'label': 'Score'},
            ax=ax
        )
        ax.set_title('VLM Evaluation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "vlm_heatmap.png", bbox_inches='tight')
        plt.close()

        logger.info("VLM heatmap saved")

    def plot_garment_metrics(self, all_results: Dict):
        """Plot Garment metrics comparison"""
        models = list(all_results.keys())
        metrics = ['ssim', 'psnr', 'lpips', 'cosine_similarity']
        metric_labels = ['SSIM', 'PSNR', 'LPIPS', 'Cosine Sim']

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]

            # Prepare data
            model_names = []
            average_values = []

            for model in models:
                if 'garment_summary' in all_results[model]:
                    if 'average' in all_results[model]['garment_summary']:
                        val = all_results[model]['garment_summary']['average'].get(f'{metric}_mean')
                        if val is not None:
                            model_names.append(model)
                            average_values.append(val)

            # Plot bar chart
            if model_names:
                x = np.arange(len(model_names))
                ax.bar(x, average_values, alpha=0.7, color='skyblue')
                ax.set_xticks(x)
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                ax.set_ylabel(label)
                ax.set_title(f'Garment-based: {label}', fontweight='bold')
                ax.grid(axis='y', alpha=0.3)

                # Add value labels
                for i, v in enumerate(average_values):
                    ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / "garment_metrics.png", bbox_inches='tight')
        plt.close()

        logger.info("Garment metrics plot saved")

    def plot_all_metrics(self, all_results: Dict):
        """Plot All-based metrics comparison"""
        models = list(all_results.keys())
        metrics = ['ssim', 'psnr', 'lpips', 'cosine_similarity']
        metric_labels = ['SSIM', 'PSNR', 'LPIPS', 'Cosine Sim']

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]

            # Prepare data
            model_names = []
            values = []

            for model in models:
                if 'all_average' in all_results[model]:
                    val = all_results[model]['all_average'].get(metric)
                    if val is not None:
                        model_names.append(model)
                        values.append(val)

            # Plot bar chart
            if model_names:
                x = np.arange(len(model_names))
                ax.bar(x, values, alpha=0.7, color='lightcoral')
                ax.set_xticks(x)
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                ax.set_ylabel(label)
                ax.set_title(f'All-based: {label}', fontweight='bold')
                ax.grid(axis='y', alpha=0.3)

                # Add value labels
                for i, v in enumerate(values):
                    ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / "all_metrics.png", bbox_inches='tight')
        plt.close()

        logger.info("All-based metrics plot saved")

    def plot_model_comparison(self, all_results: Dict, eval_types: List[str]):
        """Plot comprehensive model comparison"""
        models = list(all_results.keys())

        # Prepare data
        data = []
        labels = []

        if 'vlm' in eval_types:
            vlm_scores = []
            for model in models:
                if 'vlm_average' in all_results[model]:
                    vlm_scores.append(all_results[model]['vlm_average'].get('total_score', 0) / 5.0)  # Normalize to [0,1]
                else:
                    vlm_scores.append(0)
            data.append(vlm_scores)
            labels.append('VLM Score')

        if 'garment' in eval_types:
            garment_ssim = []
            for model in models:
                if 'garment_summary' in all_results[model]:
                    if 'average' in all_results[model]['garment_summary']:
                        val = all_results[model]['garment_summary']['average'].get('ssim_mean')
                        garment_ssim.append(val if val is not None else 0)
                    else:
                        garment_ssim.append(0)
                else:
                    garment_ssim.append(0)
            data.append(garment_ssim)
            labels.append('Garment SSIM')

        if 'all' in eval_types:
            all_ssim = []
            for model in models:
                if 'all_average' in all_results[model]:
                    val = all_results[model]['all_average'].get('ssim')
                    all_ssim.append(val if val is not None else 0)
                else:
                    all_ssim.append(0)
            data.append(all_ssim)
            labels.append('All SSIM')

        # Plot grouped bar chart
        if data:
            x = np.arange(len(models))
            width = 0.25
            fig, ax = plt.subplots(figsize=(12, 6))

            for i, (scores, label) in enumerate(zip(data, labels)):
                offset = width * (i - len(data)//2)
                ax.bar(x + offset, scores, width, label=label, alpha=0.8)

            ax.set_xlabel('Models', fontweight='bold')
            ax.set_ylabel('Score (Normalized)', fontweight='bold')
            ax.set_title('Model Comparison Across All Metrics', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / "model_comparison.png", bbox_inches='tight')
            plt.close()

            logger.info("Model comparison plot saved")
