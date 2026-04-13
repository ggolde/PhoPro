
# Core

## PhotometryLoader
A collection of classes to extract raw signals and event timestamps from various formats. It is seperated from `PhotometryExpriment` for easier support of custom loading functions.

Every native loader class has takes in file arguements in its constructor. Extracts data and packages it into a dictionary through `.extract_data()` and returns a PhotometryExperiment with `.load()`

### TDTLoader

`TDTLoader(...)` \
Args:
- data_folder (str): Path to the TDT block folder.
- box (str): TDT identifier used in stream and epoc labels.
- event_labels (list[str]): Event labels to extract from epocs; extracts only onset event timestamps by default.
- signal_label (str): Base label for the signal channel (without "box" identifier).
- isosbestic_label (str): Base label for the isosbestic channel (without "box" identifier).
- downsample (int): downsampling factor for the raw streams (mean pooling).

`.load()` \
Returns:
- `PhotometryExperiment`: containing th edownsampled data extracted from the TDT folder.

Example:
```
loader = TDTLoader(
    data_folder="data/photometry_run041026/",
    box="A",
    event_labels=["lever1", "lever2"],
    signal_label="_500",
    isosbestic_label="_450",
    downsample=10,
)

exp = loader.load()
```

## PhotometryExperiment
Handles signal preprocessing, isosbestic correction, artifact detection, and trial windowing.

### Main API

`PhotometryExperiment(...)` \
Attributes:
- id (str): unique identifier.
- metadata (dict[str, Any]): experiment metadata.
- events (dict[str, np.ndarray]): event timestamps keyed by event labels.
- raw_signal (np.ndarray): 1D array of experimental signal timepoints.
- raw_isosbestic (np.ndarray | None): 1D array of isosbestic signal timepoint or None if the experiment is single channel.
- frequency (float | None): frequency of 

metadata: dict[str, Any]
events: dict[str, np.ndarray]
raw_signal: np.ndarray
raw_isosbestic: np.ndarray
frequency: float
time: np.ndarray
trial_data: PhotometryData
Args:
- raw_signal (np.ndarray):
- raw_isosbestic (np.ndarray):
- time (np.ndarray):
- frequency (float):
- events (dict):
- metadata (dict):

`.preprocess_signal()`