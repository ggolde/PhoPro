# PhotometryExperiment

**This class is used to process and slice photometry experiments, with support for both dual and single channel experiments.**

The two main APIs:

* ``preprocess_signal()`` applies a low-pass filter, fits a reference trace, and applies correction methods, with optionally functionality to detect and correct movement artifacts.

* ``extract_trial_data()`` uses the events timestamps in the ``.events`` attribute to slice, align, and pass down trial-relative event timesteps to a 2D ``PhotometryData`` object.

---

## Example Usage

**Dual channel preprocessing**
```python
exp.preprocess_signal(
    # lowpass butterworth params
    cutoff_frequency=3,
    order=4,

    # correction method and isosbestic fit params
    correction_method='dF/F',
    fit_using='IRLS',
    maxiter=1000,
    c=2,
)
```

**Single channel preprocessing**
```python
# import artifact handlers
from pyFiberPhotometry.analysis.artifact import ODS_Detector, Spline_Corrector

# instantiate artifact detector and corrector
detector = ODS_Detector(
    score_threshold=5,
    jump_score_threshold=10,
    expand_sec=(0.5, 2),
    buffer_sec=1.5,
    n_chunks=20,
)

corrector = Spline_Corrector(
    anchor_sec=(0.2, 0.2),
    correct_spikes=True,
    correct_jumps=True,
)

exp.preprocess_signal(
    cutoff_frequency=3,
    order=4,
    correction_method='dB/B',
    fit_using='IRLS',
    maxiter=1000,
    c=2,

    # pass in artifact handlers
    artifact_detector=detector,
    artifact_corrector=corrector,
)
```

**Trial extraction**
```python
exp.extract_trial_data(
    # what event should we consider the "start" of a trial
    align_to='event',

    # what events do we want to center on
    center_on=['lever1', 'lever2'],

    # how long in seconds should our trials be relative to "center_on"
    trial_bounds=(-10, 10),

    # expected range of our events relative to "align_to"
    event_tolerences={
        'lever1':(2, 10),
        'lever2':(2, 10),
        'loud_noise':(2, 12),
    },

    # which trial-wise normalization should be preformed
    trial_normalization='none',

    # if multiple of the same event are within tolerences which should be picked
    event_conflict_logic='first',

    # should an error be thrown if multiple "center_on" event are present in the same trial
    check_overlap=True,
)
```

---

::: pyFiberPhotometry.core.PhotometryExperiment