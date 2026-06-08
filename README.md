# Dysformer

Dysformer is a PyTorch implementation of **Dysformer: A Spatial-Spectral Dual-Stream Dynamic Hyperbolic Hypergraph Transformer**.

Dysformer is designed for graph representation learning on complex structured data. It combines hyperbolic geometry, dynamic hypergraph modeling, and spatial-spectral feature fusion to capture hierarchical structures, high-order relationships, and multi-scale graph patterns.

## Overview

Most graph neural networks are built on Euclidean space and static graph topology, which may limit their ability to model hierarchical and high-order relationships. Dysformer addresses this problem through:

- **Hyperbolic representation learning** for hierarchical and tree-like graph structures.
- **Dynamic hypergraph construction** for adaptive high-order relationship modeling.
- **Spatial-spectral dual-stream fusion** for combining global structural information and local spectral details.
- **Trainable curvature regulation** for adapting the geometry across network layers.
![Dysformer Framework](https://github.com/HaoWuLab-Bioinformatics/Dysformer/blob/main/framework.png)
## Repository Structure

```text
Dysformer/
├── data/
│   └── zijian/
│       └── jiaolvyiyu/
│           ├── 2020-2023年体质+问卷数据汇总处理后_等级划分(带缺失).xlsx
│           └── 2020-2023年数据汇总_等级划分_最终.xlsx
│
├── medium/
│   ├── manifolds/              # Hyperbolic manifold operations and layers
│   ├── data_utils.py           # Data preprocessing utilities
│   ├── dataset.py              # Dataset loading and split utilities
│   ├── gnns.py                 # Graph neural network modules
│   ├── Dysformer_fusion.py     # Core Dysformer model
│   ├── logger.py               # Logging utilities
│   ├── main_new2.py            # Training / evaluation entry point
│   ├── parse.py                # Command-line arguments
│   └── tools_hyp.py            # Hyperbolic utility functions
│
├── large/
│   ├── manifolds/              # Hyperbolic manifold operations and layers
│   ├── data_utils.py           # Data preprocessing utilities
│   ├── dataset.py              # Dataset loading utilities
│   ├── eval.py                 # Evaluation utilities
│   ├── gnns.py                 # Graph neural network modules
│   ├── Dysformer_fusion.py     # Dysformer  model variants
│   ├── load_data.py            # Dataset loaders
│   ├── logger.py               # Logging utilities
│   ├── main.py                 # Training / evaluation entry point
│   └── parse.py                # Command-line arguments
│
├── __init__.py
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/HaoWuLab-Bioinformatics/Dysformer.git
cd Dysformer
```

### 2. Create a virtual environment

Using Conda:

```bash
conda create -n dysformer python=3.9
conda activate dysformer
```

### 3. Install dependencies

Install PyTorch according to your CUDA version from the official PyTorch website, then install the remaining packages:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric geoopt numpy scipy pandas scikit-learn openpyxl ogb tqdm
```

The main dependencies include:

```text
Python
PyTorch
PyTorch Geometric
Geoopt
NumPy
SciPy
Pandas
Scikit-learn
OpenPyXL
OGB
```

## Data Preparation

The repository contains adolescent health data under:

```text
data/zijian/jiaolvyiyu/
```

## Usage

### Medium-scale experiments

```bash
python yunxing_medium.py
```


### Large-scale experiments

```
python yunxing_large.py
```

## Citation

If you use Dysformer in your research, please cite:

```bibtex
@article{mei2026dysformer,
  title   = {Dysformer: A Spatial-Spectral Dual-Stream Dynamic Hyperbolic Hypergraph Transformer},
  author  = {Mei, Zhangyu and Wu, Shuang and Yi, Xiangren and Wu, Hao},
  year    = {2026},
  doi     = {10.21203/rs.3.rs-8513633/v1}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.