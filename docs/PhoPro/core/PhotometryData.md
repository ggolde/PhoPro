# PhotometryData

**This class is used to handle, manipulate, and analyze large trial-wise photometry datasets. It possesses functionality for filtering, plotting, averaging, and analyzing trials.**

---

## Example Usage

**Reading and writing**
```python
trials = PhotometryData.read_h5ad(fpath)
trials.write_h5ad(fpath)
```

**Filtering, averaging, and plotting**
```python
filtered = trials.filter_rows(
    trials.obs['trial_label'] != 'NoResponse',
)

avg = filtered.collapse(
    group_on=['trial_label'],
    metrics={'std' : np.std},
    data_cols=['event', 'AUC'],
    count_col='n_trials',
)

avg.plot_trials(
    label_with=['trial_label', 'n_trials'],
    err_layer='std',
)
```

**Recentering**
```python
recentered = trials.window(
    centers = trials.obs['event'],
    bounds = (-2, 5),
    event_cols=['event', 'lever1', 'lever2', 'loud_noise']
)
```

**Area under curve**
```python
trials.area_under_curve(
    centers=0,
    bounds=(0, 6.5),
)
```

**Export to flat formats**
```python
trials.trials_to_long_df(downsample=10)
trials.trials_to_wide_df(signal_prefix='X', downsample=20)
```

---

::: PhoPro.core.PhotometryData