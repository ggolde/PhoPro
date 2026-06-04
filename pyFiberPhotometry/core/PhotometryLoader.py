from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import json
import yaml
import tdt
import os

from .PhotometryExperiment import PhotometryExperiment
from ..utils.ops import downsample_1d

AnnotationHandler = Callable[[str, "PhotometryLoader"], dict[str, Any]]

class PhotometryLoader(ABC):
    """Abstract base class for photometry data loaders."""

    def load(self, exp_cls: type[PhotometryExperiment] = PhotometryExperiment) -> PhotometryExperiment:
        """Load data and return a `PhotometryExperiment` instance.

        Args:
            exp_cls (type[PhotometryExperiment]): Subclass of 
            PhotometryExperiment to use.

        Returns:
            PhotometryExperiment: Loaded experiment object.
        """
        data = self.extract_data()
        return exp_cls(**data)
    
    @abstractmethod
    def extract_data(self) -> dict[str, Any]:
        pass

    def read_annotation(
            self, 
            file: str | None, 
            handler: Literal['json', 'yaml'] | AnnotationHandler, 
            parent_key: str | None = None
            ) -> dict:
        """Load experiment metadata from annotation file.
        
        Args:
            file (str | None): path to annotation file.
            handler (Literal['json', 'yaml'] | AnnotationHandler): a string
                specifying how to read the annotation file or a custom
                function that takes in the file path and loader object and
                returns a dictionary.
        
        Returns:
            Dict: containing the read in metadata.
        """
        # check input
        if file is None:
            return {}
        elif not os.path.exists(file):
            raise ValueError(f'Annotation file {file} does not exsit.')
            
        # use handler to load annotations
        match handler:
            case func if callable(func):
                annots = handler(file, self)
            case 'json':
                with open(file, 'r') as f:
                    annots = json.load(f)
            case 'yaml':
                with open(file, 'r') as f:
                    annots = yaml.load(f, Loader=yaml.SafeLoader)
            case _:
                raise ValueError(f'Annotation handler {handler} not recognized!')
            
        # use parent key if specified
        if parent_key is not None:
            if parent_key not in annots:
                raise ValueError(f'{parent_key} is not a key in the full annotation file.')
            
            annots = annots[parent_key]
            
        # validate handler output
        if not isinstance(annots, dict):
            raise ValueError(f'Loaded annotations are not a dictionary. It is type {type(annots)}')
        
        return annots

class TDTLoader(PhotometryLoader):
    """Extract photometry data from TDT folder format."""

    def __init__(
            self,
            data_folder: str,
            box: str,
            event_labels: list[str],
            signal_label: str,
            isosbestic_label: str,
            downsample: int | None = None,
            annotation_file: str | None = None,
            annotation_handler: Literal['json', 'yaml'] | AnnotationHandler = 'json',
            ):
        """Initialize a TDT photometry loader.

        Args:
            data_folder (str): Path to the TDT block folder.
            box (str): TDT box identifier used in stream and epoc labels.
            event_labels (list[str]): Event labels to extract from epocs.
            signal_label (str): Base label for the signal channel.
            isosbestic_label (str): Base label for the isosbestic channel.
            downsample (int | None, optional): Downsampling factor for the raw
                streams (mean pooling). If None, no downsampling is performed.
                Defaults to ``None``.
            annotation_file (str, optional): JSON file within the TDT folder
                that contains experiment metadata. For TDT, the annotations
                should be a nested dictionary with the box prefix being the
                first key.
            annotation_handler (Literal['json', 'yaml'] | AnnotationHandler):
                a string specifying how to read the annotation file or a
                custom function that takes in the file path and loader object
                and returns a dictionary.

        Returns:
            None
        """
        self.data_folder = data_folder
        self.box = box
        self.signal_label = signal_label
        self.isosbestic_label = isosbestic_label
        self.event_labels = list(event_labels)
        self.downsample = 1 if downsample is None else downsample

        self.annotation_file = annotation_file
        self.annotation_handler = annotation_handler

        self.metadata = {'source' : str(self.data_folder), 'box' : str(self.box)}

    # --- data extraction ---
    def extract_data(self) -> dict[str, Any]:
        """Load data from TDT and extract streams and events.

        Downsamples the signal and isosbestic streams before packaging them
        into the experiment input dictionary.

        Returns:
            dict[str, Any]: Dictionary containing the raw signal,
                isosbestic signal, time vector, frequency, events, and
                metadata needed to construct a `PhotometryExperiment`.
        """
        tdt_obj = tdt.read_block(self.data_folder, verbose=False)

        # rip data out of TDT object
        sig = tdt_obj.streams[self.signal_label + self.box].data
        iso = tdt_obj.streams[self.isosbestic_label + self.box].data
        fs = tdt_obj.streams[self.signal_label + self.box].fs

        raw_signal: np.ndarray = downsample_1d(np.asarray(sig, dtype=np.float32), factor=self.downsample)
        raw_isosbestic: np.ndarray = downsample_1d(np.asarray(iso, dtype=np.float32), factor=self.downsample)
        frequency = float(fs) / self.downsample

        n = raw_signal.size
        time: np.ndarray = np.arange(n, dtype=float) / frequency

        events = self.extract_events(tdt_obj)
        del tdt_obj

        # load annotations
        if self.annotation_file is not None:
            annotation_fpath = os.path.join(self.data_folder, self.annotation_file)
            annots = self.read_annotation(file=annotation_fpath, handler=self.annotation_handler, parent_key=self.box)
            self.metadata.update(annots)

        data = dict(
            raw_signal=raw_signal,
            raw_isosbestic=raw_isosbestic,
            time=time,
            frequency=frequency,
            events=events,
            metadata=self.metadata
        )
        return data

    # --- event extraction ---
    def extract_events(self, tdt_obj) -> dict:
        """Extract event timestamps from a TDT block object.

        Args:
            tdt_obj: Object returned by `tdt.read_block()`.

        Returns:
            dict: Mapping of event labels to timestamp arrays.
        """
        # extract event timestamps for requested labels
        events = {}
        self.metadata['missing_events'] = []
        for label in self.event_labels:
            # some sessions may lack a label entirely if no events are recorded
            if hasattr(tdt_obj.epocs, self.box + label):
                ep = tdt_obj.epocs[self.box + label]
                events[label] = np.asarray(ep.onset)
            else:
                events[label] = np.array([], dtype=float)
                self.metadata['missing_events'].append(label)
        return events
    
class CSVLoader(PhotometryLoader):
    """Extract photometry data from CSV-based inputs."""

    def __init__(
            self,
            csv: str,
            time_col: str = 'time',
            signal_col: str = 'signal',
            isosbestic_col: str | None = 'isosbestic',
            event_cols: str | list[str] | None = None,
            downsample: int | None = None,
            annotation_file: str | None = None,
            annotation_handler: Literal['json', 'yaml'] | AnnotationHandler = 'json',
            ) -> None:
        """Initialize a CSV photometry loader.

        Args:
            csv (str): Path to the CSV file containing photometry data.
            time_col (str, optional): Column name containing time values.
                Defaults to ``'time'``.
            signal_col (str, optional): Column name containing the signal
                values. Defaults to ``'signal'``.
            isosbestic_col (str | None, optional): Column name containing the
                isosbestic values. Defaults to ``'isosbestic'``.
            event_cols (str | list[str] | None, optional): Column names containing
                an binary indication that the event is present at that timepoint
            downsample (int | None, optional): Downsampling factor for the raw
                streams (mean pooling). If None, no downsampling is performed.
                Defaults to ``None``.
            annotation_file (str, optional): JSON file within the TDT folder
                that contains experiment metadata. For TDT, the annotations
                should be a nested dictionary with the box prefix being the
                first key.
            annotation_handler (Literal['json', 'yaml'] | AnnotationHandler):
                a string specifying how to read the annotation file or a
                custom function that takes in the file path and loader object
                and returns a dictionary.

        Returns:
            None
        """
        # save fpaths and params
        self.csv = csv
        self.time_col = time_col
        self.sig_col = signal_col
        self.iso_col = isosbestic_col
        
        if isinstance(event_cols, str):
            event_cols = [event_cols]
        self.event_cols = event_cols

        self.downsample = 1 if downsample is None else downsample

        self.annotation_file = annotation_file
        self.annotation_handler = annotation_handler

        self.metadata = {'source' : str(self.csv)}

    def extract_data(self) -> dict[str, Any]:
        """Load signal, time, and event data from CSV and JSON files.

        Returns:
            dict[str, Any]: Dictionary containing the raw signal,
                isosbestic signal, time vector, frequency, events, and
                metadata needed to construct a `PhotometryExperiment`.
        """        
        # load csv
        df = pd.read_csv(self.csv)

        # load required
        if self.sig_col in df:
            raw_signal = downsample_1d(df[self.sig_col].to_numpy(), self.downsample)
        else:
            raise ValueError(f"Column for signal timepoints ({self.sig_col}) is not in {self.csv}")
        
        if self.time_col in df: 
            time = downsample_1d(df[self.time_col].to_numpy(), self.downsample)
        else: 
            raise KeyError(f"Column for timepoints ({self.time_col}) not found in CSV.")
        
        # load optional
        if self.iso_col in df: 
            raw_isosbestic = downsample_1d(df[self.iso_col].to_numpy(), self.downsample)
        else: 
            raw_isosbestic = None

        events = {}
        if self.event_cols is not None:
            missing = [col for col in self.event_cols if col not in df.columns]
            if missing:
                raise KeyError(f"Event columns ({missing}) not found in CSV.")
            
            for col in self.event_cols:
                present = df[col].fillna(False).astype(bool).to_numpy()
                events[col] = np.asarray(time[present], dtype=float)
            
        # load annotations
        if self.annotation_file is not None:
            annots = self.read_annotation(file=self.annotation_file, handler=self.annotation_handler, parent_key=None)
            self.metadata.update(annots)
        
        # package results
        data = dict(
            raw_signal = raw_signal,
            raw_isosbestic = raw_isosbestic,
            time = time,
            frequency = None,
            events = events,
            metadata = self.metadata,
        )
        return data
