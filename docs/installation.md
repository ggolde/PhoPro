# Installation
Follow the Base Package installation below. Using the package manager [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) is highly recommened.

## Base package
For a basic install there are 3 main options:

1. Download and unzip the package repository and run the following command from the project's root folder:
```
pip install .
```

2. (Requires git) Create a new venv or conda enviroment and run:
```
pip install git+https://github.com/ggolde/pyFiberPhotometry.git
```

3. (Requires conda) Download and unzip the package repository. In the terminal navigate to the project's root folder and run:
```
conda env create --name photometry --file=envs/basic_env.yml
```

Then activate your enviroment and install pyFiberPhotometry through pip:
```
conda activate photometry
pip install .
```

## FLMM support (requires conda)
The following tutorial is for adding support for the advanced analysis method [Functional Linear Mixed Modeling](https://github.com/gloewing/photometry_FLMM/) after installing the basic enviroment. The setup can be a pain but the FMM analysis framework can be very insightful. For more information about FMM for photometry data see the [original paper](https://doi.org/10.7554/eLife.95802.2) by Dr. Loewinger.

1. Install all R packages avaliable from conda:
```
conda activate photometry
conda install -c r-base=4.4.3 r-essentials
conda install -c conda-forge r-lme4=1.1_35.5 r-mass=7.3_64 r-mvtnorm=1.2_6 r-gridextra=2.3 r-rfast=2.1.0 r-matrix=1.7_2 r-arrangements r-here r-devtools
```

2. Then run the install_FLMM.R script with your conda enviroment active.
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