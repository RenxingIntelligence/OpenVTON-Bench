<div align="center">

# 👕 OpenVTON-Bench
**A Large-Scale High-Resolution Benchmark for Controllable Virtual Try-On Evaluation**

🎉 **Accepted as a Poster at NeurIPS 2026, Evaluations and Datasets Track.**

<p align="center">
  <a href="https://openreview.net/forum?id=hkbyKxmccF">
    <img src="https://img.shields.io/badge/NeurIPS%202026-ED%20Track%20%C2%B7%20Poster-6844A3?style=flat-square" alt="NeurIPS 2026 ED Track Poster">
  </a>
  <a href="https://arxiv.org/abs/2601.22725">
    <img src="https://img.shields.io/badge/arXiv-2601.22725-b31b1b.svg?style=flat-square&logo=arxiv" alt="arXiv">
  </a>
  <a href="https://huggingface.co/datasets/RenxingIntelligence/OpenVTON">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-orange?style=flat-square&logo=huggingface" alt="Dataset">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://developer.nvidia.com/cuda-downloads">
    <img src="https://img.shields.io/badge/CUDA-12.4%2B-green.svg?style=flat-square&logo=nvidia" alt="CUDA">
  </a>
  <a href="https://opensource.org/license/mit">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="Code license">
  </a>
</p>

*A comprehensive multi-modal evaluation protocol using VLM-based scores and multi-scale representation metrics.*

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#data-splits-and-sampling-clusters">Splits & Clusters</a> •
  <a href="#baseline-configurations">Baselines</a> •
  <a href="#-evaluation-protocols">Evaluation</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-citation">Citation</a>
</p>
</div>

---

## 🔍 Overview

**OpenVTON-Bench** is a **large-scale, high-resolution benchmark** designed for the **systematic evaluation of controllable virtual try-on (VTON) models**.

Unlike existing datasets and evaluation protocols that struggle with texture details and semantic consistency, OpenVTON-Bench provides:

- 🖼️ **99,925 Image Pairs**: Resolutions up to **1536×1536** for evaluating fine-grained texture generation.
- 🏷️ **Visual Cluster Sampling**: Approximately balanced sampling across **20 DINOv3 K-Means clusters**, with **14 caption-derived labels** provided separately as metadata.
- 📐 **Multi-Level Automated Evaluation**: Comprehensively covering:
  - Pixel fidelity
  - Garment consistency
  - Semantic realism

This benchmark enables **fair, reproducible, and scalable comparison** across modern diffusion-based and transformer-based try-on systems.

---

## 🧬 Data Construction Pipeline

The dataset construction combines image collection, annotation, and visual sampling:

1. 🌐 **Image Collection**: Integrating open-source VTON data and web images, followed by resolution and quality filtering.
2. ✍️ **Hybrid Annotation**: Combining human verification with Vision-Language Model (VLM) dense captioning.
3. ⚖️ **Cluster-Stratified Sampling**: Applying **K-Means (K=20)** to **DINOv3 garment embeddings**, then sampling approximately evenly across the visual clusters, subject to source availability.

---

## 📦 Dataset

### 🔗 HuggingFace Access
The full dataset is publicly available on Hugging Face under **CC BY-NC 4.0**:

> [!IMPORTANT]
> 👉 **[RenxingIntelligence/OpenVTON Dataset](https://huggingface.co/datasets/RenxingIntelligence/OpenVTON/)**

### 📊 Statistics

| Property | Value | Description |
| :--- | :--- | :--- |
| **Image Pairs** | `99,925` | Garment and person pairs |
| **Train / Validation / Test** | `89,927 / 4,989 / 5,009` | Released split sizes |
| **Resolution** | Up to `1536×1536` | Critical for fine-grained texture assessment |
| **Visual Sampling Clusters** | `20` | K-Means over DINOv3 garment embeddings |
| **Caption-Derived Labels** | `14` | Keyword-matched metadata, distinct from sampling clusters |
| **Annotation** | Hybrid | VLM dense captioning + human verified |

---

## Data splits and sampling clusters

OpenVTON-Bench contains 99,925 image pairs, split into 89,927 training, 4,989 validation, and 5,009 test pairs. Final sampling uses 20 visual clusters (C0-C19) obtained by applying K-Means to DINOv3 garment embeddings.

The `category` metadata field instead contains 14 coarse labels extracted by deterministic keyword matching over Gemini captions, with first-match precedence and the merges `tshirt -> top`, `knitwear -> sweater`, and `suit -> jacket`. These labels describe caption content; they are not a manually verified taxonomy and do not define the sampling strata. One label can span several visual clusters.

Nineteen clusters contain 5,012-5,088 pairs each. C13 contains all 3,340 available pairs from its source cluster. Sampling is approximately balanced across visual clusters, subject to source availability. The coefficient of variation is 7.612% across all 20 clusters and 0.331% across the other 19 clusters. Counts below describe the complete dataset rather than a dataset-viewer preview.

<details>
<summary>Complete label-by-cluster counts (14 labels × 20 clusters)</summary>

The cross-tabulation is split into two panels for readability. `Total` in the second panel is the sum over all 20 clusters.

| Label | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blazer | 17 | 1 | 5 | 82 | 1271 | 28 | 8 | 8 | 4 | 3 |
| cardigan | 15 | 0 | 24 | 13 | 18 | 3 | 159 | 2 | 13 | 16 |
| coat | 0 | 0 | 24 | 111 | 1115 | 0 | 118 | 1 | 2 | 10 |
| dress | 97 | 0 | 130 | 33 | 104 | 62 | 29 | 4 | 3065 | 2832 |
| hoodie | 10 | 1 | 400 | 903 | 2 | 14 | 655 | 3 | 0 | 0 |
| jacket | 101 | 3 | 134 | 412 | 1981 | 84 | 3575 | 25 | 27 | 12 |
| jeans | 563 | 234 | 36 | 29 | 21 | 1140 | 23 | 1008 | 70 | 58 |
| pants | 3923 | 29 | 33 | 18 | 3 | 3480 | 6 | 30 | 38 | 42 |
| shirt | 81 | 23 | 1755 | 2898 | 125 | 25 | 109 | 22 | 46 | 59 |
| shorts | 60 | 4761 | 15 | 8 | 3 | 131 | 7 | 3691 | 40 | 16 |
| skirt | 88 | 30 | 35 | 0 | 4 | 4 | 10 | 200 | 671 | 889 |
| sweater | 63 | 3 | 1163 | 450 | 25 | 97 | 56 | 6 | 34 | 15 |
| top | 52 | 2 | 1285 | 105 | 104 | 16 | 52 | 5 | 1065 | 1132 |
| vest | 17 | 0 | 49 | 26 | 311 | 4 | 280 | 7 | 13 | 4 |

| Label | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 | Total |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blazer | 4 | 3 | 6 | 1 | 1904 | 13 | 5 | 16 | 75 | 0 | 3454 |
| cardigan | 3858 | 9 | 39 | 334 | 33 | 18 | 50 | 94 | 8 | 7 | 4713 |
| coat | 8 | 9 | 40 | 37 | 2105 | 5 | 14 | 3 | 3 | 753 | 4358 |
| dress | 142 | 364 | 144 | 25 | 190 | 257 | 632 | 2680 | 57 | 2 | 10849 |
| hoodie | 6 | 3 | 1084 | 584 | 1 | 4 | 161 | 4 | 4 | 77 | 3916 |
| jacket | 168 | 157 | 333 | 1818 | 508 | 55 | 35 | 13 | 131 | 2823 | 12395 |
| jeans | 21 | 53 | 83 | 27 | 4 | 813 | 91 | 71 | 351 | 1 | 4697 |
| pants | 13 | 21 | 47 | 12 | 8 | 43 | 49 | 57 | 4142 | 3 | 11997 |
| shirt | 220 | 3556 | 1978 | 80 | 28 | 65 | 555 | 125 | 55 | 73 | 11878 |
| shorts | 0 | 14 | 86 | 1 | 1 | 40 | 18 | 15 | 52 | 0 | 8959 |
| skirt | 32 | 22 | 75 | 13 | 34 | 3674 | 110 | 481 | 49 | 7 | 6428 |
| sweater | 92 | 6 | 609 | 58 | 27 | 43 | 1993 | 59 | 71 | 223 | 5093 |
| top | 418 | 869 | 461 | 121 | 125 | 36 | 1347 | 1455 | 66 | 8 | 8724 |
| vest | 105 | 1 | 103 | 229 | 119 | 21 | 28 | 14 | 23 | 1110 | 2464 |

For example, `pants` spans C18 (4,142), C0 (3,923), and C5 (3,480), while `shorts` spans C1 (4,761) and C7 (3,691). The complete label frequencies are 12.01% for `pants` and 8.97% for `shorts`. Approximately balanced sampling clusters do not imply uniform frequencies for the caption-derived labels.

</details>

<details>
<summary>Cluster counts before sampling and in each released split</summary>

`Source cluster` is the number of items in the saved clustering pool assigned to that cluster before final sampling. It is distinct from the earlier raw collection and resolution-filtered candidate pool. `Total` is the number of released pairs in the cluster.

| Cluster | Source cluster | Train | Validation | Test | Total |
|:---|---:|---:|---:|---:|---:|
| C0 | 5625 | 4578 | 254 | 255 | 5087 |
| C1 | 6972 | 4578 | 254 | 255 | 5087 |
| C2 | 17824 | 4579 | 254 | 255 | 5088 |
| C3 | 21346 | 4579 | 254 | 255 | 5088 |
| C4 | 12334 | 4578 | 254 | 255 | 5087 |
| C5 | 13568 | 4579 | 254 | 255 | 5088 |
| C6 | 6735 | 4578 | 254 | 255 | 5087 |
| C7 | 5012 | 4510 | 250 | 252 | 5012 |
| C8 | 12641 | 4579 | 254 | 255 | 5088 |
| C9 | 15741 | 4579 | 254 | 255 | 5088 |
| C10 | 8493 | 4578 | 254 | 255 | 5087 |
| C11 | 9182 | 4578 | 254 | 255 | 5087 |
| C12 | 13222 | 4579 | 254 | 255 | 5088 |
| C13 | 3340 | 3006 | 167 | 167 | 3340 |
| C14 | 5250 | 4578 | 254 | 255 | 5087 |
| C15 | 7986 | 4578 | 254 | 255 | 5087 |
| C16 | 14037 | 4579 | 254 | 255 | 5088 |
| C17 | 8707 | 4578 | 254 | 255 | 5087 |
| C18 | 6611 | 4578 | 254 | 255 | 5087 |
| C19 | 5374 | 4578 | 254 | 255 | 5087 |

The released split totals are **89,927 / 4,989 / 5,009**, summing to **99,925** pairs.

</details>

### Garment taxonomy and coverage across datasets

The datasets below organize their data differently. Explicit garment labels, inherited attributes, and visual sampling clusters should be distinguished when comparing coverage.

| Dataset | Published garment taxonomy or coverage |
|:---|:---|
| VITON / VITON-HD | Tops/upper-body only; no released fine garment taxonomy |
| DressCode | Upper-body: tops, T-shirts, shirts, sweatshirts, sweaters; lower-body: skirts, trousers, shorts, leggings; dresses |
| SHHQ | Unpaired full-human data; upper/lower texture and length attributes rather than paired VTON labels |
| StreetTryOn | Unpaired; inherits DeepFashion2's 13 garment types |
| LH-400K / WVTON | Unpaired image-text training set / 440 VTON test pairs; no named garment taxonomy published |
| VTBench | Evaluator subsets; cross-category set uses top, lower, dresses, skirts |
| VTONQA | T-shirt, shirt, sweater, shorts, trousers, maxiskirt, skirt, dress |
| OpenVTON-Bench | 14 caption-derived coarse labels and 20 DINOv3 visual sampling clusters |

Dataset sources are cited in the paper's dataset comparison and bibliography. The OpenVTON-Bench labels and cluster counts above refer to the complete released manifest.

## Baseline configurations

The following settings describe the experiments reported in the paper. Public-checkpoint baselines use official released weights. `PC` means a parameter is provider-controlled or unexposed. `No fixed seed` means a fixed seed was not supplied.

<details>
<summary>Model identifiers, modes, and image settings</summary>

| Method | Checkpoint or mode | Image setting |
|:---|:---|:---|
| [OOTD](https://github.com/levihsu/OOTDiffusion) | Official OOTD-DC `checkpoint-36000` | 768×1,024 |
| [EasyControl](https://github.com/Xiaojiu-z/EasyControl) | FLUX.1-dev + official subject/inpainting weights; BF16 | 1,024²; condition 512 |
| [UNO](https://github.com/bytedance/UNO) | FLUX.1-dev + official UNO LoRA (rank 512) | 768×1,024; two references, long side 320 |
| [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | Official weights; image-editing mode; BF16 | Garment and masked-person conditioning |
| [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) | Official weights; native multi-reference editing; BF16 | Two 1,024² references |
| NanobananaPro | Fixed-output evaluation snapshot; backend revision unexposed | 1,024² padded two-image input |
| Qwen-Editor | `qwen-image-edit-plus`; `n=1`; `prompt_extend=true`; no watermark | 1,024² input/output |
| HuiWa / YingHui | Fixed-output evaluation snapshots | Two-image input protocol |
| [DCI-VTON](https://github.com/bcmi/DCI-VTON-Virtual-Try-On) | Official `viton512_v2.ckpt` / `viton512_v2.yaml`; PLMS; eta 0; `n_samples=1` | 512² |
| [PromptDresser](https://github.com/rlawjdghek/PromptDresser) | Official VITON-HD UNet; FP16; intermediate-cloth-mask start ratio 0.5 | 1,024×768 |
| [StableVITON](https://github.com/rlawjdghek/StableVITON) | Official `VITONHD_PBE_pose.ckpt`; PLMS; eta 0 | 512×384 |

</details>

<details>
<summary>Sampling steps, guidance, and seeds</summary>

| Method | Steps | Guidance | Seed |
|:---|---:|:---|:---|
| OOTD | 20 | 2.0 | 200 |
| EasyControl | 25 | 3.5 | 42 |
| UNO | 25 | 4.0 | 3407 |
| FLUX.1-Kontext-dev | 30 | 2.5 | No fixed seed |
| FLUX.2-dev | 50 | 4.0 | No fixed seed |
| NanobananaPro | PC | PC | PC |
| Qwen-Editor | PC | PC | PC |
| HuiWa / YingHui | PC | PC | PC |
| DCI-VTON | 100 | 1.0 | 23 |
| PromptDresser | 30 | Base 2.0; image 4.5; text 7.5 | No fixed seed |
| StableVITON | 50 | Effective scale 1.0 | No fixed seed |

PromptDresser uses separate base, image, and text guidance scales.

</details>

### Evaluation preprocessing

Before computing the main metrics, source, ground-truth, and generated images undergo identical aspect-ratio-preserving LANCZOS resizing and centered white padding to a common 1,024² canvas. This evaluation preprocessing is separate from the models' native image settings above.

The separate CIEDE2000 color diagnostic resizes images to 256² and uses the same valid sample IDs across methods. Target garment masks are eroded by five pixels, and the non-edited region excludes a five-pixel dilation of the garment mask. The paper's color consistency appendix provides the complete protocol and results.

The two-image inference wrappers used a fixed try-on prompt, up to three attempts with a 2-second interval, exact-filename resume, explicit failure logs, and no failure imputation. For API-based systems, the reported results refer to fixed generated outputs. The evaluation code revision used for the reported results is [`dd3d4a7`](https://github.com/RenxingIntelligence/OpenVTON-Bench/tree/dd3d4a706ebdf455f50eb3559dcda0becd1de6af).

---

## 📊 Evaluation Protocols

OpenVTON-Bench introduces a **hybrid evaluation paradigm** featuring four complementary protocols. This multi-view design captures **both perceptual and structural quality** of generated try-on results:

| Type | Description | Key Metrics |
| :--- | :--- | :--- |
| 🧠 **VLM-based** | Semantic realism via Vision-Language Models (VLM-as-a-Judge) | Background, Identity, Texture, Shape, Overall |
| ✂️ **Garment-based** | Region-level evaluation via [SAM3](https://github.com/facebookresearch/sam3) | SSIM, PSNR, LPIPS, Cosine |
| 🖼️ **All-based** | Full-image feature comparison using [DINOv3](https://github.com/facebookresearch/dinov3) | SSIM, LPIPS, Cosine |
| 👾 **Pixel-based** | Raw pixel structural comparison | MSE, PSNR, SSIM, LPIPS, FID |

### 🧮 Evaluation Metrics Guide

| Metric | Goal | Description |
| :--- | :---: | :--- |
| **SSIM** | **↑** Higher | Structural Similarity Index |
| **PSNR** | **↑** Higher | Peak Signal-to-Noise Ratio |
| **Cosine Sim** | **↑** Higher | Feature-level cosine similarity |
| **LPIPS** | **↓** Lower | Learned Perceptual Image Patch Similarity |
| **FID** | **↓** Lower | Fréchet Inception Distance |
| **MSE** | **↓** Lower | Mean Squared Error |

> [!NOTE]
> **VLM evaluation** utilizes a **1–5 scoring scale** across five semantic dimensions (Background, Identity, Texture, Shape, and Overall realism).

---

## 🛠 Installation

### Requirements

- Python **3.12+**
- CUDA **12.4+**
- *Recommended: **4× GPUs** for large-scale evaluation*

### Setup Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RenxingIntelligence/OpenVTON-Bench.git
   cd OpenVTON-Bench
   ```

2. **Create and activate the environment:**
   Using `conda` (Recommended):
   ```bash
   conda env create -f env.yaml
   conda activate bench_new
   ```
   *Or using `pip`:*
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start

### 1. Download Required Models

Before running the benchmark, prepare the necessary backbone weights for feature extraction and segmentation in the `models/` directory:

```text
models/
 ├── dinov3-vith16plus/    # Feature extraction
 └── sam3/                 # Garment segmentation
```

### 2. Configure Paths

Copy the template configuration file:
```bash
cp benchmark/config.yaml benchmark/config.local.yaml
```

Update the configuration (`benchmark/config.local.yaml`) with your generated image directories and model paths:
```yaml
data:
  test_jsonl: "./data/test_samples.jsonl"
  generated_dirs:
    - name: "your_model"
      path: "./generated_images/your_model"

models:
  dinov3:
    path: "./models/dinov3-vith16plus"
```

### 3. Run Benchmark

Run the full benchmark suite:
```bash
bash run_benchmark.sh --config benchmark/config.local.yaml
```

**Or run specific evaluation types individually:**
```bash
bash run_benchmark.sh --eval-type pixel
bash run_benchmark.sh --eval-type garment
bash run_benchmark.sh --eval-type vlm
```

> [!WARNING]
> **Dataset Format Requirement:**
> The generated images must maintain identical filenames to the source/target images as specified in the JSONL:
> `{"source": "00001.jpg", "target": "00001.jpg"}`

---

## 📁 Output Structure

The benchmark automatically generates a rich suite of analytics, outputted to the `results/` directory:

```text
results/
 └── YYYYMMDD_HHMMSS/
     ├── summary.json          # Aggregate metric scores
     ├── per_model/            # Detailed model-specific data
     └── visualizations/       # Radar plots, comparison charts, per-sample diagnostics
```

## 📈 Correlation Analysis

To measure the agreement between our automated multi-modal metrics and human subjective evaluation:

```bash
python benchmark/analyze_correlation.py \
  --result_dir results/... \
  --human_ratings data/human.json
```

---

## 🧱 Project Structure

<details>
<summary><b>Click to expand</b></summary>

```text
benchmark/
 ├── metrics/                  # Implementation of all evaluation metrics
 ├── utils/                    # Helper scripts and visualizers
 ├── run_benchmark.py          # Main execution entrypoint
 └── analyze_correlation.py    # Statistical correlation tools
```
</details>

---

## 🧾 Citation

If you find this benchmark useful in your research, please consider citing:

```bibtex
@misc{li2026openvtonbenchlargescalehighresolutionbenchmark,
      title={OpenVTON-Bench: A Large-Scale High-Resolution Benchmark for Controllable Virtual Try-On Evaluation}, 
      author={Jin Li and Tao Chen and Shuai Jiang and Weijie Wang and Jingwen Luo and Chenhui Wu},
      year={2026},
      eprint={2601.22725},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.22725}, 
}
```

---

## 📜 License

The code is licensed under the [MIT License](https://opensource.org/license/mit).

The [OpenVTON dataset](https://huggingface.co/datasets/RenxingIntelligence/OpenVTON) is released under [Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). Original asset terms and third-party rights continue to apply.
