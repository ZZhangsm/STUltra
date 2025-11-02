# STIntg

<!-- ![STIntg_Overview](https://github.com/zhoux85/STAligner/assets/31464727/1358f6b0-75ed-4bdd-9d0b-257788dff73a) -->


## Overview

STIntg is designed for alignment and integration of spatially resolved transcriptomics data.




First clone the repository. 

```
git clone https://github.com/ZZhangsm/STIntg.git
cd STIntg-main
```

It's recommended to create a separate conda environment for running STIntg:

```
#create an environment called STIntg
conda create -n env_STIntg python=3.8

#activate your environment
conda activate env_STIntg
```

Install all the required packages. 

For Linux
```
pip install -r requirements.txt
```


The use of the mclust algorithm requires the rpy2 package (Python) and the mclust package (R). See https://pypi.org/project/rpy2/ and https://cran.r-project.org/web/packages/mclust/index.html for detail.

The torch-geometric library is also required, please see the installation steps in https://github.com/pyg-team/pytorch_geometric#installation

Install STIntg.

```
python setup.py build
python setup.py install
```



<!-- ## Tutorials

Three step-by-step tutorials are included in the `Tutorial` folder and https://staligner.readthedocs.io/en/latest/ to show how to use STIntg. 


- Tutorial 1: Integrating 8 mouse embryo slices sampled at the time stages of E9.5-E16.5 (Stereo-seq)
 -->


## Support

If you have any questions, please feel free to contact us [sm.zhang@smail.nju.edu.cn](mailto:sm.zhang@smail.nju.edu.cn). 


<!-- 
## Citation
Zhou, X., Dong, K. & Zhang, S. Integrating spatial transcriptomics data across different conditions, technologies and developmental stages. Nat Comput Sci 3, 894–906 (2023). https://doi.org/10.1038/s43588-023-00528-w
 -->
