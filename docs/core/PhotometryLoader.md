# PhotometryLoader

**A module for loading data into the ``pyFiberPhotometry`` enviroment.**

Currently supports only TDT and CSV formats natively. For other data types, either extend the ``PhotometryLoader`` class or request the feature on [GitHub](https://github.com/ggolde/pyFiberPhotometry/issues).

---

## Example Usage
```python

# initialize the loader
loader = CSVLoader(
    csv= 'data/experiments/example_experiment.csv',
    time_col= 'time',
    signal_col= 'raw_signal',
    isosbestic_col= 'raw_isosbestic',
    event_cols= ['trial_cue', 'lever1', 'lever2', 'shock'],
    annotation_file='data/experiments/example_annotation.json',
    annotation_handler='json',
)

# extract the data and load it into the PhotometryExperiment class
exp = loader.load()
```

---

::: pyFiberPhotometry.core.PhotometryLoader