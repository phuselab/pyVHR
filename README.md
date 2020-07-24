![pyVHR](https://raw.githubusercontent.com/phuselab/pyVHR/master/img/pyVHR-logo.png)

---

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyvhr)
![PyPI](https://img.shields.io/pypi/v/pyvhr)
![GitHub last commit](https://img.shields.io/github/last-commit/phuselab/pyVHR)
[![GitHub license](https://img.shields.io/github/license/phuselab/pyVHR)](https://github.com/phuselab/pyVHR/blob/master/LICENSE)

## Description

**Package pyVHR** (short for Python framework for Virtual Heart Rate) is a comprehensive framework for studying methods of pulse rate estimation relying on remote photoplethysmography (rPPG). The methodological rationale behind the framework is that in order to study, develop and compare new rPPG methods in a principled and reproducible way, the following conditions should be met: *i)* a structured pipeline to monitor rPPG algorithms' input, output, and main control parameters; *ii)* the availability and the use of multiple datasets; *iii)* a sound statistical assessment of methods' performance.
pyVHR allows to easily handle rPPGmethods  and  data,  while  simplifying  the  statistical  assessment. Its main features lie in the following.
- **Analysis-oriented**. It  constitutes  a  platform  for  experiment design, involving an arbitrary number of methods applied to multiple video datasets. It provides a systemic end-to-end  pipeline,  allowing  to  assess  different  rPPG  algorithms, by easily setting parameters and meta-parameters.
- **Openness**. It comprises both method and dataset factory, so to easily extend the pool of elements to be evaluatedwith newly developed rPPG methods and any kind of videodatasets.
- **Robust assessment**. The outcomes are arranged intostructured data ready for in-depth analyses. Performance comparison is carried out based on robust nonparametric statistical tests.

Eight well-known rPPG methods, namely  *ICA*,  *PCA*, *GREEN,CHROM*, *POS*, *SSR*, *LGI*, *PBV*, are evaluated through extensive experiments across five public video datasets,  i.e. *PURE*, *LGI*, *USBC*, *MAHNOB* and *COHFACE*, and subsequent nonparametric statistical analysis.  

![pyVHR](https://raw.githubusercontent.com/phuselab/pyVHR/master/img/frameworkVHR.png)

## Installation

Install this library directly into an activated virtual environment:

```text
$ pip install pyvhr
```

or download from source and install via:

```text
$ python setup.py install
```

## Usage

The `notebooks` folder contains three different Jupyter notebooks:

**Basic usage**  
Shows the basic steps to process video for heart rate estimation trough remote PPG methods.  
[Notebook](#here)

**Extend the framework**  
This notebook shows how to extend the framework with additional datasets and methods.  
[Notebook](#here)

**Statistical analysis**  
Includes statistical analysis and the results presented in the paper (see [Reference](#reference)) applying all the available methods on six different datasets.  
[Notebook](#here)


## Methods

The framework contains the implementation of the most common methods for remote-PPG measurement, and are located in the `methods` folder.  
Currently implemented methods with reference publications are:

**Green**
> Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445.

**CHROM**
> Benezeth, Y., Li, P., Macwan, R., Nakamura, K., Gomez, R., & Yang, F. (2018, March). Remote heart rate variability for emotional state monitoring. In 2018 IEEE EMBS International Conference on Biomedical & Health Informatics (BHI) (pp. 153-156). IEEE.

**ICA**
> Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774.

**LGI**
> Pilz, C. S., Zaunseder, S., Krajewski, J., & Blazek, V. (2018). Local group invariance for heart rate estimation from face videos in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 1254-1262).

**PBV**
> De Haan, G., & Van Leest, A. (2014). Improved motion robustness of remote-PPG by using the blood volume pulse signature. Physiological measurement, 35(9), 1913.

**PCA**
> Lewandowska, M., Rumiński, J., Kocejko, T., & Nowak, J. (2011, September). Measuring pulse rate with a webcam—a non-contact method for evaluating cardiac activity. In 2011 federated conference on computer science and information systems (FedCSIS) (pp. 405-410). IEEE.

**POS**
> Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.

**SSR**
> Wang, W., Stuijk, S., & De Haan, G. (2015). A novel algorithm for remote photoplethysmography: Spatial subspace rotation. IEEE transactions on biomedical engineering, 63(9), 1974-1984.

## Datasets

Interfaces for six different datasets are provided in the `datasets` folder. Once the datasets are obtained, the respective files must be edited to match the correct path.  
Currently supported datasets are:

**COHFACE**
> https://www.idiap.ch/dataset/cohface

**LGI-PPGI**
> https://github.com/partofthestars/LGI-PPGI-DB

**MAHNOB-HCI**
> https://mahnob-db.eu/hci-tagging/

**PURE**
> https://www.tu-ilmenau.de/en/neurob/data-sets-code/pulse/

**UBFC1**
> https://sites.google.com/view/ybenezeth/ubfcrppg

**UBFC2**
> https://sites.google.com/view/ybenezeth/ubfcrppg


## Reference

If you use this code, please cite the paper:

```
@reference{
}
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details
