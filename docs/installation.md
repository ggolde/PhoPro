# Installation
Follow the [base package](#base-package) installation below. Using the package manager [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) is highly recommened.

## Base package

1. Create a [conda](https://www.geeksforgeeks.org/python/what-is-conda/) or [virtual enviroment](https://www.w3schools.com/python/python_virtualenv.asp) and activate it (optional, but recommended).
```bash
conda env create --name photometry python=3.11
conda activate photometry
```

```bash
python -m venv ./photometry_env
source ./photometry_env/Scripts/activate # for MacOS / Linux
photometry_env\Scripts\activate # for Windows
```

2. Install from PyPi or GitHub. Installing from GitHub will install the latest version.
```bash
pip install PhoPro
```
```bash
pip install git+https://github.com/ggolde/PhoPro.git
```

You can update the package with by including the ``--force-reinstall`` flag in the pip install command.

## FMM support
Follow the next step after the basic installation to add support for the advanced analysis method [Functional Linear Mixed Modeling](https://github.com/gloewing/photometry_FLMM/). It is highly recommended that you install PhoPro in a conda enviroment if you want FMM support. 

The setup can be a pain but the FMM analysis framework can be very insightful. For more information about FMM for photometry data see the [original paper](https://doi.org/10.7554/eLife.95802.2) by Dr. Loewinger.

1. Install all R packages avaliable from conda:
```
conda activate photometry
conda install -c conda-forge r-base=4.4.3 r-essentials
conda install -c conda-forge r-lme4=1.1_35.5 r-mass=7.3_64 r-mvtnorm=1.2_6 r-gridextra=2.3 r-rfast=2.1.0 r-matrix=1.7_2 r-arrangements r-here r-devtools r-rcurl
```

2. Then run the [install_FLMM.R](https://github.com/ggolde/PhoPro/blob/main/envs/install_FLMM.R) script with your conda enviroment active.
```
Rscript envs/install_FLMM.R
```

* If you get an error like ``Error: C17 standard requested but CC17 is not defined`` for any of your packages try creating the file ``~/.R/Makevars`` and add the following to it depending on your platform:
```
CC17 = clang                # for MacOS
CC17 = gcc                  # for Linux and Windows
C17FLAGS = -O2 -std=gnu17   # for all
```

3. Install the fastFMM Python wrapper with:
```
pip install fast-fmm-rpy2
```

4. With your conda enviroment active verify ``rpy2`` is working properly with:
```
python -m rpy2.situation
```

* If you get a and error similar to ``Error importing in API mode: ImportError("dlopen(..._rinterface_cffi_api.abi3.so, 0x0002): Library not loaded: .../lib/libRblas.dylib``, you will have to set the enviroment variable R_HOME within your conda enviroment then unintall and reinstall ``rpy2``.
```
conda activate photometry
conda env config vars set R_HOME="$(R RHOME)"
conda deactivate
conda activate photometry
```
```
pip uninstall rpy2 -y
pip uninstall rpy2-rinterface -y
pip uninstall rpy2-robjects -y
conda install -c conda-forge rpy2
```