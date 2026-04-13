from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import tdt

from .PhotometryExperiment import PhotometryExperiment
from ..utils.ops import downsample_1d

class PhotometryLoader(ABC):
    @abstractmethod
    def load(self) -> PhotometryExperiment:
        pass

class TDTLoader(PhotometryLoader):
    '''
    Extracts photometry data from TDT folder format.
    '''
    def __init__(
            self,
            data_folder: str,
            box: str,
            event_labels: list[str],
            signal_label: str,
            isosbestic_label: str,
            downsample: int = 10,
            ):
        """
        Args:
            data_folder (str): Path to the TDT block folder.
            box (str): TDT box identifier used in stream and epoc labels.
            event_labels (list[str]): Event labels to extract from epocs.
            signal_label (str): Base label for the signal channel.
            isosbestic_label (str): Base label for the isosbestic channel.
            downsample (int): downsampling factor for the raw streams (mean pooling).
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
        data = self.extract_data()
        obj = PhotometryExperiment(**data)
        return obj 


    # --- data extraction ---
    def extract_data(self) -> dict[str, Any]:
        """
        Load data from TDT, extract streams and events, and downsample signals.
        Returns:
            PhotometryExperiment
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
        """
        Used by `self.extract_data` to get event timestamps from tdt_obj.
        Args:
            tdt_obj (tdt): Object from `tdt.read_block()`.
        Returns:
            dict
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
    def __init__(
            self,
            signal_csv: str,
            isosbestic_signal_file: str | None = None,
            events: str | None = None,
        ) -> None:
        pass