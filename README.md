# **pyFiberPhotometry**

*A Python package for extendable and flexible processing and analysis of behavior-coupled fiber photometry experiments.*

## Overview
This package provides extensive functionaly for processing, handling, analyzing, and bulk processing fiber photometry datasets while remaining highly extendable to specific use cases. The high level APIs only require basic programming experience. It is built around 4 main modules:

* **[PhotometryLoader](https://ggolde.github.io/pyFiberPhotometry/api/PhotometryLoader/)** - loaders for various photometry data formats.

* **[PhotometryExperiment](https://ggolde.github.io/pyFiberPhotometry/api/PhotometryExperiment/)** - a class for processing and windowing photometry experiments with many avaliable preprocessing methods.

* **[PhotometryData](https://ggolde.github.io/pyFiberPhotometry/api/PhotometryData/)** - a class holding trial-wise signals and metadata with advanced filtering, averaging, windowing, and analysis functionality.

* **[PhotometryPipeline](https://ggolde.github.io/pyFiberPhotometry/api/PhotometryPipeline/)** - a class for easy, highly-customizable bulk processing of photometry data.


With another module for simulating photometry data:

* **[SimulatedPhotometryGenerator](https://ggolde.github.io/pyFiberPhotometry/api/SimulatedPhotometryGenerator/)** - a class for simulating complex photometry traces including realistic photobleaching, movement artifacts, and custom event dynamics.

---

## [Documentation](https://ggolde.github.io/pyFiberPhotometry/)

Documentation for this package can be found [here](https://ggolde.github.io/pyFiberPhotometry/). It is built with mkdocs-material and hosted on GitHub pages.

---

## Installation
You can install directly from GitHub using the command:
```
pip install git+https://github.com/ggolde/pyFiberPhotometry.git
```

See the [Installation](https://ggolde.github.io/pyFiberPhotometry/installation/) page of the docs for more details.

---

## Example Usage

```python
import numpy as np
from pyFiberPhotometry import PhotometryExperiment, SimulatedPhotometryGenerator

# create simulated data
sim = SimulatedPhotometryGenerator(
    T_sec=1000,
    fs=30,
    n_events=50,
    event_dur_sec=2,
    seed=43546
)
sim.add_event(
    relative_to='event',
    time_range=(1, 2),
    overall_prob=0.7,
    choices=['lever1', 'lever2'],
    choice_probs=[0.7, 0.3],
)
exp = sim.to_PhotometryExperiment()

# process signal
exp.preprocess_signal(
    cutoff_frequency=2,
    order=4,
    correction_method='dF',
    model='IRLS',
    c=1.4,
)
# extract trials
exp.extract_trial_data(
    align_to='event',
    center_on=['lever1', 'lever2'],
    trial_bounds=(0, 3),
    baseline_bounds=(-3, 0),
    event_tolerences={'lever1':(1, 2), 'lever2':(1, 2)},
    trial_normalization='zscore',
    check_overlap=False,
)
# get trial data in PhotometryData format
trials = exp.trial_data

# average data
avg = trials.collapse(
    group_on=None,
    method=np.nanmean,
    metrics={'std': np.std},
    data_cols=['event'],
    count_col='n_trials'
)

# plot average with errorbars
avg.plot_all(err_layer='std')

# save data
trials.write_h5ad('example.h5ad')
```

---

## Planned Features
This package is in ongoing development, please suggest any features you would like to see implemented on the [issues page](https://github.com/ggolde/pyFiberPhotometry/issues). Below are a few planned features:
* [ ] Implement import support for more formats.
* [ ] More methods for artifact detection and correction.
* [ ] Easier APIs for the cluster based permutation test and functional mixed modeling.
* [ ] Phase out "variants" modules.
* [x] Pipeline class for easy bulk processing.

## License

MIT License.

## Authors
This package is developed and used internally at the [Bizon-Setlow Lab](https://burke.neuroscience.ufl.edu/profile/bizon-jennifer/) at the University of Florida.

**Griffin Golde**: University of Florida, contact: **[ggolde@ufl.edu](mailto:ggolde@ufl.edu)**

## Citations

If you use this package in your work, and want to support the package, use the citation below (a proper publication is hopefully on the horizon):

* Golde G. *pyFiberPhotometry: Python toolkit for simulating, processing, handling, and analyzing fiber photometry data*.
GitHub repository. Version 0.2.0. Available at: https://github.com/ggolde/pyFiberPhotometry.

Additionally, if you used the FMM module, please review the [recommended reference(s)](https://github.com/gloewing/photometry_FLMM/blob/main/README.md#references) for the ``fast-fmm-rpy2`` package.