# STUltra

<div align="center">

![STUltra_Overview](https://raw.githubusercontent.com/ZZhangsm/STUltra/main/overview.png)

# **STUltra: scalable and accurate integration for subcellular-level spatial omics data**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/ZZhangsm/STUltra)
[![Downloads](https://img.shields.io/badge/Downloads-pypi-orange.svg)](https://pypi.org/project/STUltra/)
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

Subcellular-level spatial transcriptomics data contain unprecedented contexts to uncover finer cellular clusters and their interactions. However, integrative analysis at subcellular resolution faces many challenges due to its ultra-large volume, ultra-high sparsity, and severe susceptibility to technical conditions and batch effects.

We introduce **STUltra**, a scalable and accurate framework for integrating subcellular-level spatial omics data across spatial, temporal, and biomedical dimensions. Built on contrastive learning, STUltra combines a robust graph autoencoder with an interval sampling step to enhance batch-effect correction, enable clear characterization of shared and condition-specific tissue structures, and provide seamless extension to super-resolution platforms such as **Visium HD**, **Xenium**, and **Stereo-seq**.

### ✨ Key Features

- 🔬 **Subcellular Resolution**: Handles ultra-large volume and ultra-high sparsity data
- 🚀 **Scalable**: Processes datasets containing over 1,000,000 cells efficiently
- 🎯 **Accurate**: Outperforms existing tools in both accuracy and efficiency
- 🔄 **Multi-platform Support**: Compatible with Visium HD, Xenium, Stereo-seq and other subcellular platforms 
- 📊 **Batch Correction**: Robust batch-effect correction across spatial, temporal, and biomedical dimensions

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
#create an environment called STUltra
conda create -n env_STUltra python=3.8

#activate your environment
conda activate env_STUltra
```



#### 3. Install required packages
Install all the required packages. 

For Linux
```
pip install -r requirements.txt
```


# Install Python packages
pip install -r requirements.txt#### 4. Install additional dependencies

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
- **Tutorial 2**: Integrating two Xenium breast cancer samples

---

## 💬 Support

If you have any questions, please feel free to contact us:

📧 **Email**: [sm.zhang@smail.nju.edu.cn](mailto:sm.zhang@smail.nju.edu.cn)

---

<!-- 
## 📖 Citation
-->