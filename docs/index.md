# pyFiberPhotometry

*A Python package for extendable and flexible processing and analysis of behavior-coupled fiber photometry experiments.*

## Overview
This package provides extensive functionaly for processing, handling, analyzing, and bulk processing fiber photometry datasets while remaining highly extendable to specific use cases. The high level APIs only require basic programming experience. It is built around 4 main modules:

* **[PhotometryLoader](api/PhotometryLoader.md)** - loaders for various photometry data formats.

* **[PhotometryExperiment](api/PhotometryExperiment.md)** - a class for processing and windowing photometry experiments with many avaliable preprocessing methods.

* **[PhotometryData](api/PhotometryData.md)** - a class holding trial-wise signals and metadata with advanced filtering, averaging, windowing, and analysis functionality.

* **[PhotometryPipeline](api/PhotometryPipeline.md)** - a class for easy, highly-customizable bulk processing of photometry data.


With another module for simulating photometry data:

* **[SimulatedPhotometryGenerator](api/SimulatedPhotometryGenerator.md)** - a class for simulating complex photometry traces including realistic photobleaching, movement artifacts, and custom event dynamics.

## Installation
You can install directly from GitHub using the command:
```
pip install git+https://github.com/ggolde/pyFiberPhotometry.git
```

See the [Installation](installation.md) page for more details.

## Citations

If you use this package in your work, and want to support the package, use the citation below (a proper publication is hopefully on the horizon):

* Golde G. *pyFiberPhotometry: Python toolkit for simulating, processing, handling, and analyzing fiber photometry data*.
GitHub repository. Version 0.1.0. Available at: https://github.com/ggolde/pyFiberPhotometry.

Additionally, if you used the FMM module, please review the [recommended reference(s)](https://github.com/gloewing/photometry_FLMM/blob/main/README.md#references) for the ``fast-fmm-rpy2`` package.