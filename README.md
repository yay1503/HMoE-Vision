# HMoE-Vision

**Hierarchical Mixture-of-Experts for Multi-Domain Image Classification**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## Project Overview

HMoE-Vision implements a **Hierarchical Mixture-of-Experts (HMoE)** architecture that replaces monolithic multi-domain classifiers with a tree of lightweight binary routers cascading into domain-specialized expert networks. Instead of forcing a single CNN to learn disjoint visual distributions simultaneously, the system decomposes the problem into a hierarchy of binary routing decisions followed by expert inference — a first-principles approach to conditional computation in vision.

**The core thesis**: a tree of small, specialized models can outperform a single large generalist model on heterogeneous multi-domain classification while using fewer active parameters per inference pass.

The pipeline operates across **four visually distinct benchmark domains** — MNIST, FashionMNIST, SVHN, and CIFAR-10 — unified into a single training and evaluation stream. A hierarchical router tree determines the data domain at inference time, then dispatches each sample to the appropriate domain expert for final class prediction.

---

## Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.12 |
| **Deep Learning** | PyTorch 2.5.1 (CUDA 12.1) |
| **Vision** | TorchVision 0.20.1 |
| **Compute** | NVIDIA GPU (tested on RTX 4060 Laptop, 8 GB VRAM) |
| **Data** | `torchvision.datasets` (MNIST, FashionMNIST, SVHN, CIFAR-10) |
| **Environment** | Jupyter Notebook / IPython |
| **Visualization** | Matplotlib |

---

## System Architecture

### High-Level Routing Tree

The architecture implements a **binary decision tree** of CNN-based routers that partition the input space hierarchically before dispatching to specialized expert networks:

```
                        ┌─────────────┐
                        │  Router 0   │
                        │ (Grayscale  │
                        │  vs. Color) │
                        └──────┬──────┘
                   ┌───────────┴───────────┐
                   ▼                       ▼
            ┌─────────────┐         ┌─────────────┐
            │  Router 1   │         │  Router 2   │
            │ (MNIST vs.  │         │ (SVHN vs.   │
            │  FashionM.) │         │  CIFAR-10)  │
            └──────┬──────┘         └──────┬──────┘
              ┌────┴────┐            ┌─────┴─────┐
              ▼         ▼            ▼           ▼
        ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
        │ Expert  │ │ Expert  │ │ Expert  │ │ Expert  │
        │  MNIST  │ │ FashionM│ │  SVHN   │ │ CIFAR-10│
        └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

### Component Breakdown

#### 1. `Router` — Lightweight Binary Classifier (CNN)
A 4-block convolutional network (`3→16→32→64→128` channels) with Global Average Pooling and a 2-way linear head. Each router makes a single binary routing decision. Channel widths are intentionally narrow — routers are designed to be fast gatekeepers, not feature extractors.

#### 2. `experts` — Domain-Specialized Classifiers (CNN)
A 4-block convolutional network (`3→32→64→128→256` channels) with GAP and a 10-class linear head. Each expert is trained exclusively on its target domain, enabling specialization without interference from heterogeneous data distributions.

#### 3. `baseLineModel` — Monolithic Baseline (CNN)
A wider 4-block CNN (`3→64→128→256→512` channels) with dual output heads — a **domain head** (4 classes) and a **class head** (10 classes) — trained jointly on the unified dataset. Serves as the performance baseline that the HMoE system is compared against.

#### 4. `HMoE_Config` — Inference Orchestrator
A `dataclass` that composes all 3 routers + 4 experts into a single inference configuration with unified `.eval()` and `.to(device)` methods.

#### 5. `HMoE_pipeline` — Hierarchical Inference Engine
The core inference function that processes batches through the routing tree. For each batch:
- **Tier 1**: `Router 0` splits samples into grayscale-origin vs. color-origin subsets.
- **Tier 2**: `Router 1` and `Router 2` further split their respective subsets into individual domains.
- **Leaf**: Each sample reaches exactly one domain expert for final 10-class prediction.

Routing decisions use `argmax` on router logits — hard routing with no gating softmax.

### Custom Data Infrastructure

#### `DataWrapper` — Domain-Tagged Dataset
Wraps standard `torchvision` datasets to emit `(image, label, domain_id)` triples, enabling domain-aware training and evaluation on a `ConcatDataset` of all four domains.

#### `BinaryRouterDataset` — Binary Label Remapping
Re-labels samples with binary labels (`0` or `1`) for router-specific training — e.g., MNIST+FashionMNIST → `0`, SVHN+CIFAR-10 → `1` for Router 0.

### Data Augmentation Pipeline

| Domain | Training Augmentations |
|---|---|
| **SVHN** | `RandomRotation(15°)`, `RandomAffine(translate=0.1)`, `ColorJitter(0.2)` |
| **FashionMNIST** | `Resize(32×32)`, `RandomHorizontalFlip(0.5)`, `RandomAffine(10°, 0.1)`, `Grayscale→3ch` |
| **CIFAR-10** | `RandomCrop(32, pad=4)`, `RandomHorizontalFlip(0.5)`, `ColorJitter(0.2)` |
| **MNIST** | `ToTensor()` only (no augmentation) |

All domains are normalized to `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`. Grayscale datasets are expanded to 3-channel for architectural uniformity.

### Directory Structure

```
HMoE-Vision/
├── main.ipynb          # Full pipeline: data loading, model definitions,
│                       #   training loops, and HMoE inference engine
├── requirements.txt    # Pinned Python dependencies (PyTorch + CUDA 12.1)
├── models/             # Directory for saved model checkpoints (empty)
├── data/               # Auto-downloaded dataset cache (gitignored)
│   ├── MNIST/
│   ├── FashionMNIST/
│   ├── SVHN/
│   └── CIFAR10/
└── .gitignore
```

---

## Features

- **Hierarchical Routing** — Binary decision tree replaces flat multi-class domain classification, reducing per-router complexity to a 2-way decision.
- **Conditional Computation** — Only one router path and one expert are activated per sample at inference, minimizing active parameter count.
- **Domain-Specialized Experts** — Each expert trains on a single dataset distribution, eliminating negative transfer and gradient interference.
- **Multi-Domain Unified Evaluation** — All four domains are concatenated into a single DataLoader for apples-to-apples accuracy reporting across 156k+ test images.
- **Dual-Metric Reporting** — The pipeline separately tracks domain routing accuracy and final class prediction accuracy.
- **Modular Configuration** — `HMoE_Config` dataclass cleanly composes arbitrary router/expert combinations for experimentation.
- **Baseline Comparison** — Includes a monolithic dual-head CNN baseline for direct A/B evaluation.
- **Label Smoothing** — `CrossEntropyLoss(label_smoothing=0.1)` regularizes both domain and class predictions.

---

## Installation & Setup

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA 12.1 compatible drivers
- ~2 GB disk space for dataset downloads

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yay1503/HMoE-Vision.git
cd HMoE-Vision

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install PyTorch with CUDA 12.1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Launch the notebook
jupyter notebook main.ipynb
```

> **Note**: Datasets (MNIST, FashionMNIST, SVHN, CIFAR-10) are automatically downloaded to `data/` on first execution. Total download size is ~600 MB.

---

## Usage

### Running the Full Pipeline

Open `main.ipynb` and execute cells sequentially. The notebook is structured as:

1. **Cells 1–9**: Device setup, data augmentation transforms, dataset loading.
2. **Cells 10–18**: `DataWrapper` / `BinaryRouterDataset` creation, unified DataLoader construction.
3. **Cells 19–28**: Baseline model definition, training loop, and evaluation.
4. **Cells 29–36**: Expert and Router model definitions, router dataset creation.
5. **Cells 37–end**: `HMoE_Config` assembly, `HMoE_pipeline` inference, and accuracy reporting.

### Training the Baseline Model

```python
train_baseline(
    model=model_base,
    loss_fn_domain=loss_fn_domain,
    optimizer=optimizer,
    data_loader=unified_train_loader,
    device=device,
    loss_fn_class=loss_fn_class,
    epochs=30
)
```

### Running Hierarchical MoE Inference

```python
config = HMoE_Config()

HMoE_pipeline(
    config=config,
    dataloader=unified_test_loader,
    device=device
)
# Output:
#   Hierarchical MoE - Final Domain Accuracy: XX.XX%
#   Hierarchical MoE - Final Class Accuracy: XX.XX%
```

### Extending to New Domains

To add a fifth domain (e.g., STL-10):
1. Create a new `experts()` instance and train on STL-10.
2. Modify the routing tree — extend Router 0 or add a Tier 3 router.
3. Add a `BinaryRouterDataset` entry with the appropriate binary label.
4. Update `HMoE_Config` and `HMoE_pipeline` with the new routing path.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Hard routing (argmax)** | Eliminates soft-gating overhead; each sample activates exactly one expert. |
| **Binary routers** | Simpler decision boundary per node; tree depth scales logarithmically with domain count. |
| **Separate router training** | Routers learn domain discrimination independently of class semantics. |
| **Narrow router channels** | Routers need coarse features (texture, color distribution), not fine-grained class features. |
| **No shared backbone** | Experts operate independently — no gradient crosstalk between domains. |

---

*Built from scratch with PyTorch — no pretrained models, no external MoE libraries.*
