<!-- # STUltra -->

<div align="center">


# **STUltra: Mapping disease traits onto  spatiotemporal domain landscapes through scalable multi-sample integration of spatial transcriptomics**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/ZZhangsm/STUltra)
<!-- [![Downloads](https://img.shields.io/badge/Downloads-pypi-orange.svg)](https://pypi.org/project/STUltra/) -->
<!-- [![DOI](https://img.shields.io/badge/DOI-Coming%20Soon-yellow.svg)](https://github.com/ZZhangsm/STUltra) -->

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Step-by-step Installation](#step-by-step-installation)
- [Tutorials](#-tutorials)
- [Support](#-support)

---

## 🎯 Overview

![STUltra_Overview](https://github.com/ZZhangsm/STUltra/blob/main/overview.jpg)

High-resolution spatial transcriptomics (ST) generates large-scale datasets across multiple tissue sections, developmental stages, and disease conditions, making integration challenging due to biological heterogeneity and batch effects.

**STUltra** is a scalable hypergraph framework for multi-sample ST integration, spatial domain detection, and GWAS-based trait mapping. By combining interval-sampled hypergraphs, robust graph representation learning, and contrastive learning, STUltra learns batch-corrected spatial embeddings that enable accurate domain identification and disease trait mapping across integrated tissue landscapes.

### ✨ Key Features

- 🚀 **Million-scale Scalability**: Efficiently processes datasets containing over **1,000,000** spatial locations
- 🔄 **Cross-platform Integration**: Supports Visium HD, Xenium, Stereo-seq, and other high-resolution spatial transcriptomics platforms
- 🧩 **Robust Batch Correction**: Integrates samples across spatial, temporal, and biological conditions while reducing batch effects
- 🧬 **Disease Trait Mapping**: Integrates GWAS summary statistics across ultra-large spatiotemporal tissues

---

## 💻 Installation

### Prerequisites

- Python 3.8+
- Conda (recommended)
- R (for mclust algorithm)

### Step-by-step Installation

#### 1. Clone the repository
First clone the repository. 

```
git clone https://github.com/ZZhangsm/STUltra.git
cd STUltra-main
```


#### 2. Create a conda environment


```
# create an environment called STUltra
conda create -n env_STUltra python=3.8

# activate your environment
conda activate env_STUltra
```



#### 3. Install required packages
Install all the required packages. 

For Linux
```
# Install Python packages

pip install -r requirements.txt
```
#### 4. Install additional dependencies

**mclust algorithm:**
- Python: Install `rpy2` package - see [rpy2 documentation](https://pypi.org/project/rpy2/)
- R: Install `mclust` package - see [mclust documentation](https://cran.r-project.org/web/packages/mclust/index.html)

**torch-geometric:**
- Follow the installation steps in [PyTorch Geometric documentation](https://github.com/pyg-team/pytorch_geometric#installation)

#### 5. Install STUltra

```
python setup.py build
python setup.py install
```
## 📚 Tutorials

Step-by-step tutorials are included in the `Tutorials` folder to demonstrate how to use STUltra:

- **Tutorial 1**: Integrating 8 mouse embryo slices sampled at the time stages of E9.5-E16.5 (Stereo-seq)
- **Tutorial 2**: Integrating two Xenium breast cancer samples.
- **Tutorial 3**: Integrating two Visium HD colorectal cancer samples from the 10x Genomics official dataset (total cell count > 1M)

---

## 💬 Support

If you have any questions, please feel free to contact us:

📧 **Email**: [sm.zhang@smail.nju.edu.cn](mailto:sm.zhang@smail.nju.edu.cn)

## License

STUltra is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the developers of [STAligner](https://github.com/zhoux85/STAligner) and [STAGATE](https://github.com/RucDongLab/STAGATE) for making their code publicly available. We acknowledge their contributions, which have been valuable to the development of STUltra.

---

