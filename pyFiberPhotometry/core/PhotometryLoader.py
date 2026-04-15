from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import json
import tdt

from .PhotometryExperiment import PhotometryExperiment
from ..utils.ops import downsample_1d

class PhotometryLoader(ABC):
    """Abstract base class for photometry data loaders."""

    @abstractmethod
    def load(self) -> PhotometryExperiment:
        """Load data and return a `PhotometryExperiment` instance.

        Returns:
            PhotometryExperiment: Loaded experiment object.
        """
        pass

class TDTLoader(PhotometryLoader):
    """Extract photometry data from TDT folder format."""

    def __init__(
            self,
            data_folder: str,
            box: str,
            event_labels: list[str],
            signal_label: str,
            isosbestic_label: str,
            downsample: int = 10,
            ):
        """Initialize a TDT photometry loader.

        Args:
            data_folder (str): Path to the TDT block folder.
            box (str): TDT box identifier used in stream and epoc labels.
            event_labels (list[str]): Event labels to extract from epocs.
            signal_label (str): Base label for the signal channel.
            isosbestic_label (str): Base label for the isosbestic channel.
            downsample (int, optional): Downsampling factor for the raw
                streams (mean pooling). Defaults to ``10``.

        Returns:
            None
        """
        self.data_folder = data_folder
        self.box = box
        self.signal_label = signal_label
        self.isosbestic_label = isosbestic_label
        self.event_labels = list(event_labels)
        self.downsample = downsample

        self.metadata = {}

    # --- loader method ---
    def load(self) -> PhotometryExperiment:
        """Load TDT data and return a `PhotometryExperiment` instance.

        Returns:
            PhotometryExperiment: Loaded experiment object.
        """
        data = self.extract_data()
        obj = PhotometryExperiment(**data)
        return obj 


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
        tdt_obj = tdt.read_block(self.data_folder)

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
            events_json: str | None = None,
            downsample: int = 10,
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
            events_json (str | None, optional): Path to a JSON file mapping
                event labels to timestamps. Defaults to ``None``.
            downsample (int, optional): Downsampling factor applied to the CSV
                series. Defaults to ``10``.

        Returns:
            None
        """
        # save fpaths and params
        self.csv = csv
        self.time_col = time_col,
        self.sig_col = signal_col,
        self.iso_col = isosbestic_col,
        
        self.events_json = events_json
        self.downsample = downsample

        self.metadata = {}

    def extract_data(self) -> dict[str, Any]:
        """Load signal, time, and event data from CSV and JSON files.

        Returns:
            dict[str, Any]: Dictionary containing the raw signal,
                isosbestic signal, time vector, frequency, events, and
                metadata needed to construct a `PhotometryExperiment`.
        """
        # load events
        with open(self.events_json, 'r') as f:
            events: dict = json.load(f)
        events = {str(event) : np.asarray(timestamps) for event, timestamps in events.items()}
        
        # load csv
        df = pd.read_csv(self.csv)

        if self.sig_col in df:
            raw_signal = downsample_1d(df[self.sig_col].to_numpy(), self.downsample)
        else:
            raise ValueError(f"Column for signal timepoints ({self.sig_col}) is not in {self.csv}")
        
        if self.iso_col in df: 
            raw_isosbestic = downsample_1d(df[self.iso_col].to_numpy(), self.downsample)
        else: 
            raw_isosbestic = None

        if self.time_col in df: 
            time = downsample_1d(df[self.iso_col].to_numpy(), self.downsample)
        else: 
            time = None
        
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
    
    def load(self) -> PhotometryExperiment:
        """Load CSV-based data and return a `PhotometryExperiment` instance.

        Returns:
            PhotometryExperiment: Loaded experiment object.
        """
        data = self.extract_data()
        return PhotometryExperiment(**data)
