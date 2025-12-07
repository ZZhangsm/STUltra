# STUltra

![STUltra_Overview](https://github.com/ZZhangsm/STUltra/blob/main/overview.png)


## Overview

We introduce STUltra, a scalable and accurate framework for integrating subcellular-level spatial omics data across spatial, temporal, and biomedical dimensions.



First clone the repository. 

```
git clone https://github.com/ZZhangsm/STUltra.git
cd STUltra-main
```

It's recommended to create a separate conda environment for running STUltra:

```
#create an environment called STUltra
conda create -n env_STUltra python=3.8

#activate your environment
conda activate env_STUltra
```

Install all the required packages. 

For Linux
```
pip install -r requirements.txt
```


The use of the mclust algorithm requires the rpy2 package (Python) and the mclust package (R). See https://pypi.org/project/rpy2/ and https://cran.r-project.org/web/packages/mclust/index.html for detail.

The torch-geometric library is also required, please see the installation steps in https://github.com/pyg-team/pytorch_geometric#installation

Install STUltra.

```
python setup.py build
python setup.py install
```



<!-- ## Tutorials

Three step-by-step tutorials are included in the `Tutorial` folder and https://staligner.readthedocs.io/en/latest/ to show how to use STUltra. 


- Tutorial 1: Integrating 8 mouse embryo slices sampled at the time stages of E9.5-E16.5 (Stereo-seq)
 -->


## Support

If you have any questions, please feel free to contact us [sm.zhang@smail.nju.edu.cn](mailto:sm.zhang@smail.nju.edu.cn). 


<!-- 
## Citation
Zhou, X., Dong, K. & Zhang, S. Integrating spatial transcriptomics data across different conditions, technologies and developmental stages. Nat Comput Sci 3, 894–906 (2023). https://doi.org/10.1038/s43588-023-00528-w
 -->
