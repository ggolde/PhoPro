# PhotometryPipeline

**A highly flexible class for bulk processing photometry data. Supports custom operations and subclasses of PhotometryData and PhotometryExperiment**

---

## Example Usage

This example is for an risky decision making task in rats stored in the TDT format with multiple experiments per TDT folder.

### Setup
```python
from PhoPro import PhotometryExperiment, PhotometryData, PhotometryPipeline, TDTLoader

# --- loader params ---
shared_loader_kwargs = dict(
    event_labels = ['Lrg', 'Sml', 'Hsl', 'Zap'],
    signal_label = '_465',
    isosbestic_label = '_405',
    downsample = 10,
    annotation_file = 'annotations.json',
    annotation_handler = 'json',
)
loader_kwargs = [
    dict(box='A', **shared_loader_kwargs),
    dict(box='B', **shared_loader_kwargs)
]

# --- preprocess params ---
preprocess_kwargs = dict(
    cutoff_frequency = 3.0,
    order = 4,
    correction_method = 'dF/F',
    signal_normalization = 'none',
    fit_using = 'OLS',
    maxiter = 1000,
    c = 3,
    artifact_detector = None,
    artifact_corrector = None,
)

# --- trial extraction params ---
trial_extraction_kwargs = dict(
        align_to = 'Hsl',
        center_on = ['Lrg', 'Sml'],
        trial_bounds = (-23.0, 5.0),
        baseline_bounds = (-5, -1),
        event_tolerences = {'Lrg' : (5, 18), 'Sml' : (5, 18), 'Zap': (4.5, 18.5)},
        trial_normalization = 'zero',
        check_overlap = False,
        time_error_threshold = 0.01,
        event_conflict_logic = 'first',
)

# --- uid builder ---
def RDT_id_builder(exp: PhotometryExperiment) -> str:
    id = (
        f"{exp.metadata.get('rat', 'UnknownRat')}_"
        f"{exp.metadata.get('current', 'UnknownCurrent')}uA_"
        f"Box{exp.metadata.get('box', 'UnknownBox')}_"
        f"{exp.metadata.get('stripped_date', 'UnknownDate')}_"
        f"{exp.metadata.get('source', 'UnknownSource').split('-')[-1]}"
    )
    return id

# --- post loading operation ---
# removes last dummy trial
def RDT_post_load(exp: PhotometryExperiment) -> None:
    if 'Hsl' in exp.events:
        exp.events['Hsl'] = exp.events['Hsl'][:-1]
```

### Run
```python
pipeline = PhotometryPipeline(
    data_directory='database/NAc_Young_RDT_Photometry',
    target_type='folder',
    loader_cls=TDTLoader,
    experiment_cls=PhotometryExperiment,
    data_cls=PhotometryData,
    recursive=False,
    pattern='Emely*'
)

result = pipeline.run(
    output_dir='/pipeline_RDT',
    loader_kwargs=loader_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    trial_extraction_kwargs=trial_extraction_kwargs,
    log_file='pipeline.log',
    passdown_metadata=['rat', 'current', 'box', 'source'],
    id_builder=RDT_id_builder,
    post_load_operation=RDT_post_load,
)
```

---

::: PhoPro.core.PhotometryPipeline