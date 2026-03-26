# **pyFiberPhotometry**

**Python package for processing fiber photometry data**

`pyFiberPhotometry` provides a framework for processing, analyzing, and simulating behavior-coupled fiber photometry datasets.
 Currently it only natively supports importing data from Tucker-Davis Technologies (TDT) acquisition systems.
 Built around two core classes that are subclassed for specific pipeline implementations. \
 - PhotometryData holds trial-wise data in an AnnData format. \
 - PhotometryExperiment extracts and processes signals, yeilding PhotometryData.

 Used internally in the **[Bizon-Setlow lab](https://neuroscience.ufl.edu/profile/bizon-jennifer/)** at the University of Florida.

---

## **Installation**

Install from GitHub:

```bash
pip install git+https://github.com/ggolde/pyFiberPhotometry.git
```

---

## **Quick Start**

### **Load and preprocess a TDT fiber photometry session**

```python
import numpy as np
from pyFiberPhotometry import PhotometryExperiment, SimulatedPhotometryGenerator

# read data from TDT folder
exp = PhotometryExperiment(
    data_folder="path/to/TDT/block",
    box="A",
)

# or create simulated data
sim = SimulatedPhotometryGenerator(
    T_sec=1000,
    fs=30,
    n_events=50,
    seed=43546
)
exp = sim.to_PhotometryExperiment()

# process signal
exp.preprocess_signal(
    cutoff_frequency=2,
    order=4,
    method='dF/F',
    model='IRLS',
    c=1.4,
)
# extract trials
exp.extract_trial_data(
    align_to='event',
    center_on=['event'],
    trial_bounds=(0, 3),
    baseline_bounds=(-3, 0),
    event_tolerences={},
    normalization='zscore',
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

# save data
trials.write_h5ad('example.h5ad')
```

### **Plot a trial**

```python
trials.plot_line(0)
```

---

## **Planned Features**
This package is still in development, please suggest any features you would like to see implemented!
- Improve import support for formats other than TDT.
- Full tutorials and docs.
- Better support for events in simulated data.

## **License**

MIT License.

---

## **Author**

**Griffin Golde**: University of Florida,
contact: **[ggolde@ufl.edu](mailto:ggolde@ufl.edu)**