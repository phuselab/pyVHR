<img src="https://raw.githubusercontent.com/phuselab/pyVHR/master/img/pyVHR-logo.png" alt="pyVHR logo" width="300"/>

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyvhr)
[![PyPI](https://img.shields.io/pypi/v/pyvhr)](https://pypi.org/project/pyVHR/)
![GitHub last commit](https://img.shields.io/github/last-commit/phuselab/pyVHR)
[![GitHub license](https://img.shields.io/github/license/phuselab/pyVHR)](https://github.com/phuselab/pyVHR/blob/master/LICENSE)


**Package pyVHR** (short for Python framework for Virtual Heart Rate) is a comprehensive framework for studying methods of pulse rate estimation relying on video, also known as remote photoplethysmography (rPPG).

## Description

The methodological rationale behind the framework is that in order to study, develop and compare new rPPG methods in a principled and reproducible way, the following conditions should be met: *i)* a structured pipeline to monitor rPPG algorithms' input, output, and main control parameters; *ii)* the availability and the use of multiple datasets; *iii)* a sound statistical assessment of methods' performance.
pyVHR allows to easily handle rPPGmethods  and  data,  while  simplifying  the  statistical  assessment. Its main features lie in the following:
- **Analysis-oriented**. It  constitutes  a  platform  for  experiment design, involving an arbitrary number of methods applied to multiple video datasets. It provides a systemic end-to-end  pipeline,  allowing  to  assess  different  rPPG  algorithms, by easily setting parameters and meta-parameters.
- **Openness**. It comprises both method and dataset factory, so to easily extend the pool of elements to be evaluatedwith newly developed rPPG methods and any kind of videodatasets.
- **Robust assessment**. The outcomes are arranged intostructured data ready for in-depth analyses. Performance comparison is carried out based on robust nonparametric statistical tests.

Nine classical rPPG methods, namely  *ICA*,  *PCA*, *GREEN*, *CHROM*, *POS*, *SSR*, *LGI*, *PBV*, *OMIT*, as well as the recent DL-based model *MTTS-CAN* are evaluated through extensive experiments across five public video datasets,  i.e. *PURE*, *LGI*, *UBFC*, *MAHNOB* and *COHFACE*, and subsequent nonparametric statistical analysis.  

![pipeline](https://user-images.githubusercontent.com/642555/152432564-12014442-d455-4462-9b1e-3082a3fdd5bf.png)

## Getting started

### Dependencies

The quickest way to get started is to install the [miniconda](http://conda.pydata.org/miniconda.html) distribution, a lightweight minimal installation of Anaconda Python.

Once installed, create a new `conda` environment and automatically fetch all the dependencies based on your architecture (with or without GPU), using one of the following commands:

**CPU-only version**
```bash
conda env create --file https://raw.githubusercontent.com/phuselab/pyVHR/pyVHR_CPU/pyVHR_CPU_env.yml
```

**CPU+GPU version**  
This yml environment is for cudatoolkit=10.2 and python=3.8.
```bash
conda env create --file https://raw.githubusercontent.com/phuselab/pyVHR/master/pyVHR_env.yml
```

### Installation

Enter the newly created conda environment and install the latest stable release build of pyVHR with:

**CPU-only version**
```bash
conda activate pyvhr
(pyvhr) pip install pyvhr-cpu
```

**CPU+GPU version**  
```bash
conda activate pyvhr
(pyvhr) pip install pyvhr
```

## Basic usage and Notebooks
Run the following code to obtain BPM estimates over time for a single video:

```python
from pyVHR.analysis.pipeline import Pipeline
import matplotlib.pyplot as plt

pipe = Pipeline()
time, BPM, uncertainty = pipe.run_on_video('/path/to/video', roi_approach="patches", roi_method="faceparsing")

plt.figure()
plt.plot(time, BPM)
plt.fill_between(time, BPM-uncertainty, BPM+uncertainty, alpha=0.2)
plt.show()
```
The full documentation of `run_on_video` method, with all the possible parameters, can be found here: [https://phuselab.github.io/pyVHR/](https://phuselab.github.io/pyVHR/pyVHR.analysis.html?highlight=run_on_video#pyVHR.analysis.pipeline.Pipeline.run_on_video)

The `notebooks` folder contains useful Jupyter notebooks.

## Documentation

The full documentation of the pyVHR framework is available at [https://phuselab.github.io/pyVHR/](https://phuselab.github.io/pyVHR/).

## GUI
In the folder `realtime` you can find an example of a simple GUI created using the pyVHR package.
You can launch it by going into the path `pyVHR/realtime/` and using the command 

```bash
python GUI.py
```

If you want to use a specific rPPG method and pre-post filterings, you must set them in the last lines of `GUI.py`. 

Below is a video showing the use of the GUI.

https://user-images.githubusercontent.com/34277835/136981161-8799051a-ca0d-45c6-b4dd-e146457c7bdd.mp4


## Developing

The latest unstable development build of pyVHR is available on GitHub, and can be obtained downloading from source and installing via:
```bash
git clone git@github.com:phuselab/pyVHR.git
cd pyVHR/
python setup.py install
```

The `main` branch refers to the full pyVHR framework (requires GPU), while the `pyVHR_CPU` branch is dedicated to the CPU-only architectures.

### Custom installation
If you want to create your environment from scratch you should follow these steps:
- Install PyTorch ([here](https://pytorch.org/))
- Install Numba ([here](https://numba.pydata.org/numba-doc/latest/user/installing.html))
- Install Cupy (for GPU only) with the correct CUDA version ([here](https://docs.cupy.dev/en/stable/install.html#installing-cupy))
- Install CuSignal (for GPU only) using conda and remove from the command 'cudatoolkit=x.y' ([here](https://github.com/rapidsai/cusignal))
- Install Kaleido ([here](https://pypi.org/project/kaleido/))
- Install PyTables ([here](https://anaconda.org/anaconda/pytables))
- Install pyVHR as shown above.

## Methods

The framework contains the implementation of many common methods for remote-PPG measurement. Currently implemented methods with reference publications are:

**Green** / *Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445.*

**CHROM** / *De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.*

**ICA** / *Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774.*

**LGI** / *Pilz, C. S., Zaunseder, S., Krajewski, J., & Blazek, V. (2018). Local group invariance for heart rate estimation from face videos in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 1254-1262).*

**PBV** / *De Haan, G., & Van Leest, A. (2014). Improved motion robustness of remote-PPG by using the blood volume pulse signature. Physiological measurement, 35(9), 1913.*

**PCA** / *Lewandowska, M., Rumiński, J., Kocejko, T., & Nowak, J. (2011, September). Measuring pulse rate with a webcam—a non-contact method for evaluating cardiac activity. In 2011 federated conference on computer science and information systems (FedCSIS) (pp. 405-410). IEEE.*

**POS** / *Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.*

**SSR** / *Wang, W., Stuijk, S., & De Haan, G. (2015). A novel algorithm for remote photoplethysmography: Spatial subspace rotation. IEEE transactions on biomedical engineering, 63(9), 1974-1984.*

**OMIT** / *Álvarez Casado, C., Bordallo López, M. (2022). Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces. arXiv (eprint 2202.04101).*

**MTTS-CAN** / *Liu, X., Fromm, J., Patel, S., & McDuff, D. (2020). Multi-task temporal shift attention networks for on-device contactless vitals measurement. Advances in Neural Information Processing Systems, 33, 19400-19411.*

## Datasets

Interfaces for five different datasets are provided in the `datasets` folder. Once the datasets are obtained, the respective files must be edited to match the correct path.  
Currently supported datasets are:

**COHFACE** / *https://www.idiap.ch/dataset/cohface*

**LGI-PPGI** / *https://github.com/partofthestars/LGI-PPGI-DB*

**MAHNOB-HCI** / *https://mahnob-db.eu/hci-tagging/*

**PURE** / *https://www.tu-ilmenau.de/neurob/data-sets-code/pulse-rate-detection-dataset-pure*

**UBFC1** / *https://sites.google.com/view/ybenezeth/ubfcrppg*

**UBFC2** / *https://sites.google.com/view/ybenezeth/ubfcrppg*

## Reference

If you use this code, please cite the paper:

```
@article{Boccignone2020,
  doi = {10.1109/access.2020.3040936},
  url = {https://doi.org/10.1109/access.2020.3040936},
  year = {2020},
  publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
  pages = {1--1},
  author = {Giuseppe Boccignone and Donatello Conte and Vittorio Cuculo and Alessandro D’Amelio and Giuliano Grossi and Raffaella Lanzarotti},
  title = {An Open Framework for Remote-{PPG} Methods and their Assessment},
  journal = {{IEEE} Access}
}
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details
