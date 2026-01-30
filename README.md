# OpenVTON-Bench

A comprehensive benchmark for evaluating Virtual Try-On (VTON) models with multi-dimensional metrics.

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#evaluation-metrics">Metrics</a> •
  <a href="#citation">Citation</a>
</p>

## Overview

OpenVTON-Bench is a systematic evaluation framework for virtual try-on generation models. It provides four types of evaluation approaches:

| Evaluation Type | Description | Metrics |
|-----------------|-------------|---------|
| **VLM-based** | Multi-dimensional quality assessment using Vision-Language Models | Background Consistency, Person Consistency, Texture Fidelity, Shape Preservation, Overall Realism |
| **Garment-based** | Fine-grained garment region evaluation with SAM3 segmentation | Cosine Similarity, SSIM, PSNR, LPIPS (multi-scale) |
| **All-based** | Full image comparison using DINOv3 features | Cosine Similarity, SSIM, PSNR, LPIPS |
| **Pixel-based** | Direct pixel-level comparison | MSE, SSIM, PSNR, LPIPS, FID |

## Installation

### Prerequisites

- Python 3.12+
- CUDA 12.4+
- 4+ GPUs recommended for large-scale evaluation

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/RenxingIntelligence/OpenVTON-Bench.git
cd OpenVTON-Bench

# Create conda environment
conda env create -f env.yaml
conda activate bench_new

# Or install via pip
pip install -r requirements.txt
```

### Download Models

Download the required model checkpoints:

```bash
# DINOv3 model (for feature extraction)
# Place in: ./models/dinov3-vith16plus/

# SAM3 model (for garment segmentation)
# Place in: ./models/sam3/
```

## Quick Start

### 1. Prepare Configuration

Copy and modify the configuration file:

```bash
cp benchmark/config.yaml benchmark/config.local.yaml
```

Edit `config.local.yaml` to set your paths:

```yaml
data:
  test_jsonl: "./data/test_samples.jsonl"
  generated_dirs:
    - name: "your_model"
      path: "./generated_images/your_model"
      description: "Your Model Description"
  output_dir: "./results"

models:
  dinov3:
    path: "./models/dinov3-vith16plus"
  sam:
    checkpoint: "./models/sam3/"
  qwen_vl:
    api_key: "${QWEN_VL_API_KEY}"  # Set via environment variable
```

### 2. Set Environment Variables

```bash
# For VLM evaluation (optional)
export QWEN_VL_API_KEY="your-api-key"
```

### 3. Run Evaluation

```bash
# Run full evaluation
bash run_benchmark.sh --config benchmark/config.local.yaml

# Run specific evaluation type
bash run_benchmark.sh --eval-type pixel
bash run_benchmark.sh --eval-type garment
bash run_benchmark.sh --eval-type vlm

# Evaluate specific models
bash run_benchmark.sh --models "model_a model_b"
```

Or use Python directly:

```bash
python benchmark/run_benchmark.py \
    --config benchmark/config.yaml \
    --eval_type pixel garment \
    --models your_model
```

## Data Format

### Test Data (JSONL)

```json
{"source": "garment_001.jpg", "target": "person_wearing_001.jpg"}
{"source": "garment_002.jpg", "target": "person_wearing_002.jpg"}
```

### Generated Images

Place your model's generated images in the configured directory. File names should match the source image names in the test JSONL.

## Evaluation Metrics

### VLM-based Evaluation (1-5 scale)

| Metric | Description |
|--------|-------------|
| Background Consistency | Preservation of background elements |
| Person Consistency | Identity and body structure preservation |
| Texture Fidelity | Garment texture detail restoration |
| Shape Preservation | Garment shape and style accuracy |
| Overall Realism | Overall image naturalness |

### Pixel/Feature-based Metrics

| Metric | Range | Better |
|--------|-------|--------|
| SSIM | [0, 1] | Higher |
| PSNR | [0, ∞) | Higher |
| LPIPS | [0, ∞) | Lower |
| FID | [0, ∞) | Lower |
| Cosine Similarity | [-1, 1] | Higher |
| MSE | [0, ∞) | Lower |

## Output Structure

```
results/
└── YYYYMMDD_HHMMSS_evaluation/
    ├── summary.json              # Overall summary
    ├── summary.txt               # Text report
    ├── per_model/
    │   └── model_name/
    │       └── results.json      # Detailed per-model results
    ├── visualizations/           # Generated charts
    │   ├── vlm_radar.png
    │   ├── vlm_heatmap.png
    │   ├── garment_metrics.png
    │   └── model_comparison.png
    └── debug/                    # Debug visualizations (optional)
        └── garment/
            └── sample_000/
```

## Correlation Analysis

Analyze the correlation between automated metrics and human ratings:

```bash
python benchmark/analyze_correlation.py \
    --result_dir ./results/YYYYMMDD_HHMMSS_evaluation \
    --human_ratings ./data/human_ratings.json \
    --model your_model \
    --output_dir ./correlation_analysis
```

## Project Structure

```
OpenVTON-Bench/
├── benchmark/
│   ├── run_benchmark.py          # Main evaluation script
│   ├── analyze_correlation.py    # Correlation analysis tool
│   ├── config.yaml               # Configuration template
│   ├── metrics/
│   │   ├── vlm_evaluator.py      # VLM-based evaluator
│   │   ├── garment_evaluator.py  # Garment-based evaluator
│   │   ├── all_evaluator.py      # All-based evaluator
│   │   ├── pixel_evaluator.py    # Pixel-based evaluator
│   │   └── metrics_utils.py      # Metric calculation utilities
│   └── utils/
│       ├── sam_segmenter.py      # SAM3 segmentation
│       ├── image_processing.py   # Image utilities
│       └── report_generator.py   # Report generation
├── run_benchmark.sh              # Shell script entry point
├── env.yaml                      # Conda environment
└── README.md
```

## Hardware Requirements

| Evaluation Type | GPU Memory | Recommended GPUs |
|-----------------|------------|------------------|
| Pixel-based | ~4GB | 1x GPU |
| All-based | ~8GB | 1x GPU |
| Garment-based | ~16GB | 2-4x GPUs |
| VLM-based | API-based | N/A |

## Citation

If you find this benchmark useful for your research, please cite our paper:

```bibtex
@article{openvton-bench2024,
  title={OpenVTON-Bench: A Comprehensive Benchmark for Virtual Try-On Evaluation},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

**Paper**: [Coming Soon](#)

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

- [DINOv2/v3](https://github.com/facebookresearch/dinov2) for visual feature extraction
- [SAM](https://github.com/facebookresearch/segment-anything) for garment segmentation
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) for perceptual similarity

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com).
