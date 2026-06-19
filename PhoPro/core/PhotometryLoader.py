"""Load continuous photometry recordings into experiment objects."""

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
from ..utils.operations import downsample_signal, downsample_time

AnnotationHandler = Callable[[str, "PhotometryLoader"], dict[str, Any]]

##############################
#region --- ABSTRACT CLASS ---
##############################
class PhotometryLoader(ABC):
    """Abstract base class for photometry data loaders."""

    def load(self, exp_cls: type[PhotometryExperiment] = PhotometryExperiment) -> PhotometryExperiment:
        """Load data and return a `PhotometryExperiment` instance.

        Parameters
        ----------
        exp_cls : type[PhotometryExperiment], default=PhotometryExperiment
            Experiment class used to construct the loaded object.

        Returns
        -------
        PhotometryExperiment
            Loaded experiment object.
        """
        data = self.extract_data()
        return exp_cls(**data)

    @abstractmethod
    def extract_data(self) -> dict[str, Any]:
        """Extract loader-specific data for `PhotometryExperiment` construction.

        Returns
        -------
        dict[str, Any]
            Keyword arguments accepted by `PhotometryExperiment`.
        """
        pass

    def read_annotation(
            self,
            file: str | None,
            handler: Literal['json', 'yaml'] | AnnotationHandler,
            parent_key: str | None = None
            ) -> dict:
        """Load experiment metadata from annotation file.

        Parameters
        ----------
        file : str or None
            Path to the annotation file. If ``None``, an empty dictionary is
            returned.
        handler : {'json', 'yaml'} or AnnotationHandler
            Built-in annotation format name or a callable that accepts the file
            path and this loader and returns a dictionary.
        parent_key : str or None, default=None
            Optional top-level key to select from the loaded annotation
            dictionary.

        Returns
        -------
        dict
            Loaded annotation metadata.

        Raises
        ------
        ValueError
            If the file is missing, the handler is unknown, ``parent_key`` is
            absent, or the loaded annotations are not a dictionary.
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

#endregion

###################
#region --- TDT ---
###################
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
            downsample_kwargs: dict = {},
            annotation_file: str | None = None,
            annotation_handler: Literal['json', 'yaml'] | AnnotationHandler = 'json',
            ):
        """Initialize a TDT photometry loader.

        Parameters
        ----------
        data_folder : str
            Path to the TDT block folder.
        box : str
            TDT box identifier appended to stream and epoc labels.
        event_labels : list[str]
            Event labels to extract from epocs.
        signal_label : str
            Base label for the signal stream.
        isosbestic_label : str
            Base label for the isosbestic stream.
        downsample : int or None, default=None
            Downsampling factor for raw streams. If ``None``, no downsampling
            is performed.
        downsample_kwargs : dict, default={}
            Additional keyword arguments passed to the downsampling helpers.
        annotation_file : str or None, default=None
            Annotation filename inside ``data_folder``. The loaded annotation
            dictionary is indexed by ``box`` before metadata are added.
        annotation_handler : {'json', 'yaml'} or AnnotationHandler, default='json'
            Built-in annotation format name or custom annotation reader.
        """
        self.data_folder = data_folder
        self.box = box
        self.signal_label = signal_label
        self.isosbestic_label = isosbestic_label
        self.event_labels = list(event_labels)
        self.downsample = 1 if downsample is None else downsample
        self.downsample_kwargs = downsample_kwargs

        self.annotation_file = annotation_file
        self.annotation_handler = annotation_handler

        self.metadata = {'source' : str(self.data_folder), 'box' : str(self.box)}

    # --- data extraction ---
    def extract_data(self) -> dict[str, Any]:
        """Load data from TDT and extract streams and events.

        Downsamples the signal and isosbestic streams before packaging them
        into the experiment input dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing raw signal, raw isosbestic signal, time
            vector, sampling frequency, events, and metadata.
        """
        tdt_obj = tdt.read_block(self.data_folder, verbose=False)

        # rip data out of TDT object
        sig = tdt_obj.streams[self.signal_label + self.box].data
        iso = tdt_obj.streams[self.isosbestic_label + self.box].data
        fs = tdt_obj.streams[self.signal_label + self.box].fs
        start_time = tdt_obj.streamts[self.signal_label + self.box].start_time

        # downsample raw signals
        raw_signal: np.ndarray = downsample_signal(np.asarray(sig, dtype=np.float32), factor=self.downsample, **self.downsample_kwargs)
        raw_isosbestic: np.ndarray = downsample_signal(np.asarray(iso, dtype=np.float32), factor=self.downsample, **self.downsample_kwargs)

        # contruct time
        n_times = raw_signal.size
        raw_time = (start_time + np.arange(n_times, dtype=float)) / float(fs)
        time = downsample_time(np.asarray(raw_time, dtype=np.float32), factor=self.downsample, **self.downsample_kwargs)
        frequency = time.size / (time[-1] - time[0])

        # extract events
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

        Parameters
        ----------
        tdt_obj
            Object returned by `tdt.read_block`.

        Returns
        -------
        dict
            Mapping from requested event labels to onset timestamp arrays.
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

#endregion

###################
#region --- CSV ---
###################
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
            downsample_kwargs: dict = {},
            annotation_file: str | None = None,
            annotation_handler: Literal['json', 'yaml'] | AnnotationHandler = 'json',
            ) -> None:
        """Initialize a CSV photometry loader.

        Parameters
        ----------
        csv : str
            Path to the CSV file containing photometry data.
        time_col : str, default='time'
            Column containing time values.
        signal_col : str, default='signal'
            Column containing signal values.
        isosbestic_col : str or None, default='isosbestic'
            Column containing isosbestic values. If absent from the CSV, the
            loaded experiment is single-channel.
        event_cols : str, list[str], or None, default=None
            Column or columns containing truthy values where events occur.
        downsample : int or None, default=None
            Downsampling factor for the raw arrays. If ``None``, no
            downsampling is performed.
        downsample_kwargs : dict, default={}
            Additional keyword arguments passed to the downsampling helpers.
        annotation_file : str or None, default=None
            Path to an annotation file to merge into metadata.
        annotation_handler : {'json', 'yaml'} or AnnotationHandler, default='json'
            Built-in annotation format name or custom annotation reader.
        """
        # save fpaths and params
        self.csv = csv
        self.time_col = time_col
        self.sig_col = signal_col
        self.iso_col = isosbestic_col

        if isinstance(event_cols, str):
            event_cols = [event_cols]
        self.event_cols = event_cols

        self.downsample = downsample
        self.downsample_kwargs = downsample_kwargs

        self.annotation_file = annotation_file
        self.annotation_handler = annotation_handler

        self.metadata = {'source' : str(self.csv)}

    def extract_data(self) -> dict[str, Any]:
        """Load signal, time, and event data from CSV and JSON files.

        Returns
        -------
        dict[str, Any]
            Dictionary containing raw signal, optional raw isosbestic signal,
            time vector, events, and metadata.

        Raises
        ------
        ValueError
            If the configured signal column is absent.
        KeyError
            If the configured time column or event columns are absent.
        """
        # load csv
        df = pd.read_csv(self.csv)

        # load required
        if self.sig_col in df:
            raw_signal = downsample_signal(df[self.sig_col].to_numpy(), factor=self.downsample, **self.downsample_kwargs)
        else:
            raise ValueError(f"Column for signal timepoints ({self.sig_col}) is not in {self.csv}")

        if self.time_col in df:
            time = downsample_time(df[self.time_col].to_numpy(), factor=self.downsample, **self.downsample_kwargs)
        else:
            raise KeyError(f"Column for timepoints ({self.time_col}) not found in CSV.")

        # load optional
        if self.iso_col in df:
            raw_isosbestic = downsample_signal(df[self.iso_col].to_numpy(), factor=self.downsample, **self.downsample_kwargs)
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
