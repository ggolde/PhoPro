from __future__ import annotations
from typing import Literal, Any
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import json
import os
import re
import logging

from ..core import PhotometryData, PhotometryExperiment

class SUGA_PhotometryExperiment(PhotometryExperiment):
    """
    SUGA-specific implementation of PhotometryExperiment for TDT-based SUGA sessions.
    """
    def __init__(
        self,
        data_folder: str,
        box: str = "A",
        event_labels: list[str] = ["OC", "CC"],
        signal_label: str = "_500",
        isosbestic_label: str = "_450",
        annot_filename: str = "annotations.json",
        ):
        """
        Initialize an SUGA_PhotometryExperiment.
        Args:
            data_folder (str): Path to the TDT block folder.
            box (str): TDT box identifier.
            event_labels (list[str]): Event labels to extract for SUGA from annotation file.
            signal_label (str): Base label for the signal stream.
            isosbestic_label (str): Base label for the isosbestic stream.
            annot_filename (str): File name of the notes file in the data folder.
        Returns:
            SUGA_PhotometryExperiment
        """
        super().__init__(data_folder, box, event_labels, signal_label, isosbestic_label, annot_filename)

        self.parse_annotations(self.notes_filename)
        self.metadata['folder'] = self.data_folder
        self.id = (
            f"{self.metadata.get('rat', 'UnknownRat')}_"
            f"Box{self.box}_"
            f"{self.metadata.get('treatment', 'UnknownTreatment')}"
            f"{data_folder.split('-')[-1]}"
        )
        self.metadata['uid'] = self.id


    def parse_annotations(self, filename: str = 'annotations.json') -> None:
        """
        Parse a generated .json annotation file and populate self.metadata. 
        Annotation is expected to be a nested dictionary with the first keys being the box e.g. 'A',
        and the second level containing any amount of annotations, but must include
        'rat', 'treatment', 'OC', and 'CC'.
        Raises error if annotation file not found.
        Args:
            filename (str): Generated annotation file name relative to the data folder.
        Returns:
            None
        """
        self.metadata['box'] = self.box
        annot_file = os.path.join(self.data_folder, filename)

        if not os.path.exists(annot_file):
            raise ValueError(f'Annotation file {annot_file} not found')

        with open(annot_file, 'r') as f:
            annot = json.load(f)
            self.metadata.update(annot[self.box])
        
        # check for rat and treatment
        rat = self.metadata.get('rat')
        treatment = self.metadata.get('treatment')
        if (rat is None) or (treatment is None):
            raise ValueError(f'Rat or treatment not found in notes file')
        
    def extract_events(self, tdt_obj) -> dict:
        '''
        Get event timestamps from `self.metadata`.

        Args:
            tdt_obj (tdt): Unused, events taken from `self.metadata`.
        Returns:
            events (dict): dictionary containing events and event timestamps as np.ndarray's.
        '''
        events = {}
        self.metadata['missing_events'] = []
        for label in self.event_labels:
            # some sessions may lack a label entirely if no events are recorded
            if self.metadata.get(label, None) is not None:
                events[label] = np.asarray([self.metadata[label]], dtype=float)
            else:
                events[label] = np.array([], dtype=float)
                self.metadata['missing_events'].append(label)
        return events
    
    # --- pipeline API ---
    def run_pipeline(
        self,
        logger: logging.Logger | None = None,
        downsample: int = 10,
        cutoff_frequency: float = 3.0,
        order: int = 4,
        preprocess_method: str = 'dF/F',
        preprocess_normalization: Literal['nullZ', 'none'] = 'none',
        c: float = 3,
        maxiter: int = 200,
        fit_using: Literal['OLS', 'IRLS', 'IRLS_no_intercept'] = 'IRLS',
        
        align_to: str = 'OC',
        center_on: list[str] = ['OC'],
        trial_bounds: tuple[float, float] = (-100.0, 3000.0),
        baseline_bounds: tuple[float, float] = (-100, -10),
        event_tolerences: dict[str, tuple[float, float]] = {'CC':(-1000.0, 1000.0)},
        normalization: Literal['zscore', 'zero', 'none'] = 'zero',
        check_overlap: bool = False,
        time_error_threshold: float = 0.01,
        ) -> None:
        """
        Run full SUGA pipeline: extract, preprocess, window trials, label, QC, and optionally save.
        Args:
            logger (logging.Logger | None): Logger for status messages.
            downsample (int): Downsampling factor for raw TDT streams.
            cutoff_frequency (float): Low-pass filter cutoff in Hz.
            order (int): Butterworth filter order.
            preprocess_method (str): Preprocessing method, e.g. 'dF/F' or 'dF'.
            preprocess_normalization (str): Normalization for whole processed signal, 'nullZ' or 'none'.
            c (float): Tukey-biweight IRLS parameter, lower means more aggressive downweighting. 1.4 <= c <= 3 is recommended.
            maxiter (int): Maximum iterations for IRLS isosbestic to signal fitting.
            fit_using (Literal['OLS', 'IRLS', 'IRLS_no_intercept']): model used to fit isosbestic.
            align_to (str): Primary event label used to align trials.
            center_on (list[str]): Events used to refine trial centers.
            trial_bounds (tuple[float, float]): Trial window bounds relative to center.
            baseline_bounds (tuple[float, float]): Baseline window bounds relative to center.
            event_tolerences (dict[str, tuple[float, float]]): Time tolerances for event annotation.
            normalization (Literal['zscore', 'zero', 'none']): Trial normalization method.
            check_overlap (bool): Whether to throw an error multiple ``center_on`` events are found in the same trial.
            time_error_threshold (float): Threshold on timing error for sanity check.
        Returns:
            None
        """
        # step 0: logging
        log = logger or logging.getLogger(__name__)
        log.info(f"Starting Pipeline {self.id}...")
        
        # step 1: extract raw data from TDT
        log.info(f"Extracting raw data from TDT block, downsampling x{downsample}...")
        self.extract_data(downsample=downsample)
        if len(self.metadata['missing_events']) != 0:
            log.warning(f"Requested events {self.metadata['missing_events']} are missing")
        if align_to in self.metadata['missing_events']:
            raise ValueError(f'There are no {align_to} events present in data!')

        # step 2: preprocess signal with lowpass filter and dF/F strategy
        log.info(f"Preprocessing signal...")
        self.preprocess_signal(
            cutoff_frequency=cutoff_frequency, 
            order=order, 
            method=preprocess_method, 
            normalization=preprocess_normalization,
            c=c,
            maxiter=maxiter,
            fit_using=fit_using,
        )
        log.info(f"Done. Fitted isosbestic R2 = {self.metadata['isosbestic_fit']['r2_val']:.4f}")

        # step 3: extract per-trial data
        log.info(f"Extracting per-trial data...")
        self.extract_trial_data(
            align_to=align_to,
            center_on=center_on,
            trial_bounds=trial_bounds,
            baseline_bounds=baseline_bounds,
            event_tolerences=event_tolerences,
            normalization=normalization,
            time_error_threshold=time_error_threshold,
            check_overlap=check_overlap,
        )
        log.info(f"Done. Extracted {self.trial_data.n_trials} trials of {self.trial_data.n_times} size each.")

        # step 4: annotate and clean per-trial data
        log.info(f"Annotating trial data...")
        self.trial_data.add_obs_columns(self.metadata, keys=['rat', 'box', 'treatment', 'date', 'uid'])

        log.info(f"Done. {self.trial_data.n_trials} trials remain.")
        log.info(f"Pipeline complete.")
    
def SUGA_process_whole_directory(
    data_dir: str,
    output_dir: str,
    log_file: str | None = None,
    save_dashboards: bool = False,

    trial_data_file: str = 'trials.h5ad',
    dashboard_folder: str = 'dashboard_plots',

    boxes: list[str] = ['A', 'B'],
    event_labels: list[str] = ('OC', 'CC'),
    signal_label: str = '_500',
    isosbestic_label: str = '_450',
    annotation_filename: str = 'annotations.json',

    pipeline_kwargs: dict[str, Any] = {},
    ) -> PhotometryData:
    """
    Runs full RDT pipeline on all folders in a directory and combines the results.
    Args:
        data_dir (str): Directory containing the TDT data folders.
        output_dir (str): Directory to save processed data in.
        log_file (str) : Path to log file.
        save_dashboards (bool) : Whether to save graphical dashboard of signal and isosbestic for each experiment.

        trial_data_file (str) : Name for trial data output file.
        dashboard_folder (str) : Name for folder that dashboards will be saved to in ``output_dir``.

        boxes (str): TDT box identifiers to extract data from.
        event_labels (list[str]): Event labels to extract for RDT.
        signal_label (str): Base label for the signal stream.
        isosbestic_label (str): Base label for the isosbestic stream.
        annotation_filename (str): File name of the annotations file (externally generated) in the data folder.

        pipeline_kwargs (dict[str, Any]): Arguments to be passed to ``run_pipeline()``, see function for more details.
    Returns:
        RDT_PhotometryData object containing all trials extracted
    """
    # set up logger
    log = logging.getLogger(__name__)
    if log_file is not None:
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, force=True)

    log.info("Starting data ripping process...")

    # create list of tdt data folders, ignore non-directories
    tdt_folders_list = [os.path.join(data_dir, foldername) for foldername in os.listdir(data_dir)]
    tdt_folders_list = [folderpath for folderpath in tdt_folders_list if not os.path.isfile(folderpath)]

    # concat savefiles
    trial_data_path = os.path.join(output_dir, trial_data_file)
    dashboard_folder_abs = os.path.join(output_dir, dashboard_folder)

    # delete previous files (only if they are not folders)
    for path in [trial_data_path]:
        if os.path.isfile(path) and os.path.exists(path): os.remove(path)

    # create dashboard folder if needed
    if save_dashboards and (not os.path.exists(dashboard_folder_abs)):
        os.mkdir(dashboard_folder_abs)
    
    # init tracking metrics
    n_experiments = int(len(boxes)*len(tdt_folders_list))
    n_errors = 0
    i = 1    

    trial_data: PhotometryData = None

    # loop through every TDT folder add box
    for tdt_folder in tdt_folders_list:
        for box in boxes:
            log.info(f"Processing {tdt_folder}, box {box} ({i} / {n_experiments})...")
            try:
                exp = SUGA_PhotometryExperiment(
                    tdt_folder, 
                    box=box,
                    event_labels=event_labels,
                    signal_label=signal_label,
                    isosbestic_label=isosbestic_label,
                    annot_filename=annotation_filename
                    )
                exp.run_pipeline(
                    logger=log,
                    **pipeline_kwargs
                    )
                
                if i == 1:
                    log.info(f"Creating first trial object...")
                    trial_data = PhotometryData(exp.trial_data.adata.copy())
                else:
                    log.info(f"Appending to trial_data object...")
                    trial_data.combine_obj(exp.trial_data, inplace=True)
                
                if save_dashboards == True:
                    log.info(f"Plotting and saving dashboard...")
                    save_path = os.path.join(dashboard_folder_abs, getattr(exp, 'id', 'Unnamed') + '.svg')
                    exp.dashboard(save=save_path)

                log.info(f"Experiment info for {exp.id}")
                log.info(
                    f"Rat: {exp.metadata['rat']}\n"
                    f"treatment: {exp.metadata['treatment']}\n"
                    f"n_trials: {exp.trial_data.n_trials}\n"
                    f"n_times: {exp.trial_data.n_times}\n"
                )
                
                del exp
                i += 1

            except Exception as e:
                n_errors += 1
                log.error(f"Error processing {tdt_folder}, box {box}: \n\t {e}\n")
                i += 1
                continue

    log.info('Saving trial data...')
    trial_data.write_h5ad(trial_data_path)

    log.info(
        f'Data processing complete with {n_errors} errors '
        f'out of {n_experiments} experiments'
    )

    return trial_data

# --- ANNOTATING ---
ASSIGN_RE = re.compile(
    r"""
    ^\s*
    (?:BOX\s*)?
    (?P<group>[A-Z])          # A, B, ...
    \s*:?\s*
    (?:SUGA\s*)?              # optional 'SUGA'
    (?P<rat_num>\d+)
    \s*:?\s*
    (?P<treatment>[A-Za-z0-9_-]+)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

OC_RE = re.compile(
    r"""
    ^\s*
    (?:SUGA\s*)?              # optional 'SUGA'
    (?P<rat_num>\d+)
    \s+OC
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

NOTE_RE = re.compile(
    r'Note-\d+:\s*(?P<time>\d{1,2}:\d{2}:\d{2}(?:am|pm))\s*\[.*?\]\s*"(?P<event>[^"]+)"',
    re.IGNORECASE,
)

START_RE = re.compile(
    r"Start:\s*(?P<time>\d{1,2}:\d{2}:\d{2}(?:am|pm))\s+(?P<date>\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)


def _normalize_event(event: str) -> str:
    """Uppercase, remove duplicate whitespace, normalize separators."""
    event = event.strip().upper()
    event = event.replace(":", " ")
    event = re.sub(r"\s+", " ", event)
    return event


def parse_experiment_text(text: str) -> dict[str, dict[str, str]]:
    """
    Parse one experiment note text into a structured dict.

    Returns
    -------
    dict
        Example:
        {
            "A": {
                "rat": "SUGA16",
                "treatment": "SALINE",
                "start": "9:12:14am",
                "OC": "9:28:24am",
                "CC": "9:29:53am"
            },
            ...
        }
    """
    start_match = START_RE.search(text)
    if not start_match:
        raise ValueError("Could not find experiment Start line.")

    experiment_start = start_match.group("time")
    experiment_date = start_match.group("date")

    notes: list[dict[str, str]] = []
    for m in NOTE_RE.finditer(text):
        notes.append(
            {
                "time": m.group("time"),
                "event_raw": m.group("event"),
                "event": _normalize_event(m.group("event")),
            }
        )

    # ---- pass 1: infer all group -> rat/treatment assignments ----
    data: dict[str, dict[str, str]] = {}
    rat_to_group: dict[str, str] = {}

    for note in notes:
        event = note["event"]
        m = ASSIGN_RE.match(event)
        if m:
            group = m.group("group").upper()
            rat = f"SUGA{m.group('rat_num')}"
            treatment = m.group("treatment").upper()

            data[group] = {
                "rat": rat,
                "treatment": treatment,
                "date": experiment_date,
                "start": experiment_start,
            }
            rat_to_group[rat] = group

    # ---- pass 2: attach OC and CC in note order ----
    pending_cc = deque()  # queue of groups that have OC but not yet CC

    for note in notes:
        event = note["event"]
        time = note["time"]

        # skip assignment events in second pass
        if ASSIGN_RE.match(event):
            continue

        # OC event
        oc_match = OC_RE.match(event)
        if oc_match:
            rat = f"SUGA{oc_match.group('rat_num')}"
            group = rat_to_group.get(rat)

            if group is None:
                continue  # unknown rat, ignore safely

            # Ignore duplicate OC for same group before a CC is assigned
            already_waiting = group in pending_cc
            if already_waiting:
                continue

            # Keep first OC only
            if "OC" not in data[group]:
                data[group]["OC"] = time

            pending_cc.append(group)
            continue

        # CC event
        if event == "CC":
            if pending_cc:
                group = pending_cc.popleft()
                if "CC" not in data[group]:
                    data[group]["CC"] = time
        
    # ---- convert OC / CC timestamps to seconds from start ----
    time_fmt = "%I:%M:%S%p"

    start_dt = datetime.strptime(experiment_start.upper(), time_fmt)

    for group in data.values():

        if "OC" in group:
            oc_dt = datetime.strptime(group["OC"].upper(), time_fmt)
            group["OC"] = float((oc_dt - start_dt).total_seconds())

        if "CC" in group:
            cc_dt = datetime.strptime(group["CC"].upper(), time_fmt)
            group["CC"] = float((cc_dt - start_dt).total_seconds())

    return data


def parse_experiment_file(txt_path: str | Path, json_out: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Parse a single experiment text file."""
    txt_path = Path(txt_path)
    text = txt_path.read_text(encoding="utf-8")
    parsed = parse_experiment_text(text)

    if json_out is not None:
        json_out = Path(json_out)
        json_out.write_text(json.dumps(parsed, indent=4), encoding="utf-8")

    return parsed

def SUGA_annotate_directory(data_dir: str, log_file: str | None = None, note_file_name: str = 'Notes.txt', annotation_file_name: str = 'annotations.json') -> None:
    """Annotate directory of TDT folders"""
    # set up logger
    log = logging.getLogger(__name__)
    if log_file is not None:
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, force=True)

    # create list of tdt data folders, ignore non-directories
    tdt_folders_list = [os.path.join(data_dir, foldername) for foldername in os.listdir(data_dir)]
    tdt_folders_list = [folderpath for folderpath in tdt_folders_list if not os.path.isfile(folderpath)]

    n_errors = 0

    for i, tdt_folder in enumerate(tdt_folders_list):
        try:
            log.info(f'Annotating {tdt_folder} ({i+1}/{len(tdt_folders_list)})...')

            note_file_path = os.path.join(tdt_folder, note_file_name)
            annotation_file_path = os.path.join(tdt_folder, annotation_file_name)

            if not os.path.exists(note_file_path):
                raise RuntimeWarning(f'{tdt_folder} does not have notes file: {note_file_name}')
            
            annots = parse_experiment_file(note_file_path, annotation_file_path)

            if len(annots) == 0:
                raise ValueError('Empty Annotation')
            for box, annot in annots.items():
                keys = list(annot.keys())
                for test in ['rat', 'OC', 'CC']:
                    if test not in keys:
                        raise Warning(f'{test} not found in Box {box}')

        except Exception as e:
            log.error(f"Error annotating {tdt_folder}: \n\t {e}\n")
            n_errors += 1
    
    log.info(f'Annotation complete with {n_errors} errors of out {len(tdt_folders_list)} folders!')

    return None