# **pyFiberPhotometry**

**Python package for processing fiber photometry data**

`pyFiberPhotometry` provides a framework for loading, processing, analyzing, and simulating behavior-coupled fiber photometry datasets.
 Currently it only natively supports importing data from Tucker-Davis Technologies (TDT) acquisition systems.
 Built around core classes that are subclassed for specific pipeline implementations.
 - PhotometryLoader extracts relevant data from input format.
 - PhotometryExperiment processes and slices signals with many options for processing.
 - PhotometryData holds trial-wise data in an AnnData format. 

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
from pyFiberPhotometry import PhotometryExperiment, SimulatedPhotometryGenerator, TDTLoader

# read data from TDT folder
exp = PhotometryExperiment.load_TDT(
    data_folder="path/to/TDT/block",
    box="A",
    event_labels=['event']
)

# or create simulated data
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
    iso_correction_method='dF',
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

## **License**

MIT License.

---

## **Author**

**Griffin Golde**: University of Florida,
contact: **[ggolde@ufl.edu](mailto:ggolde@ufl.edu)**