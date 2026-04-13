from __future__ import annotations
from typing import Literal, Any
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import re
import logging

from ..core import PhotometryData, PhotometryExperiment, TDTLoader
from ..analysis.artifact import detect_artifacts_two_channel_ODS, intervals_to_flat_idx


class SUGA_PhotometryExperiment(PhotometryExperiment):
    artifact_df: pd.DataFrame
    artifact_info: dict[str, float]

    """
    SUGA-specific implementation of PhotometryExperiment for TDT-based SUGA sessions.
    """
    def __init__(
        self,
        data_folder: str,
        box: str = "A",
        event_labels: list[str] = [],
        signal_label: str = "_500",
        isosbestic_label: str = "_450",
        annot_filename: str = "annotations.json",
        downsample: int = 10,
        trim_first: int = 1000,
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
            downsample (int): factor to downsample raw signals by.
        Returns:
            SUGA_PhotometryExperiment
        """
        loader = TDTLoader(
            data_folder=data_folder,
            box=box,
            event_labels=event_labels,
            signal_label=signal_label,
            isosbestic_label=isosbestic_label,
            downsample=downsample,
        )

        super().__init__(**loader.extract_data())

        self.raw_signal = self.raw_signal[trim_first:]
        self.raw_isosbestic = self.raw_isosbestic[trim_first:]
        self.time = self.time[trim_first:]

        self.data_folder = data_folder
        self.box = box
        self.event_labels = event_labels
        self.signal_label = signal_label
        self.isosbestic_label = isosbestic_label
        self.annot_filename = annot_filename

        self.parse_annotations(annot_filename)
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
        
        # assign events
        self.events['OC'] = np.asarray([self.metadata['OC']])
        self.events['CC'] = np.asarray([self.metadata['CC']])
    
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
        fit_using: Literal['OLS', 'IRLS', 'IRLS_no_intercept', 'OLS_no_intercept'] = 'IRLS',
        detrend_bleaching: bool = False,
        
        align_to: str = 'OC',
        center_on: list[str] = ['OC'],
        trial_bounds: tuple[float, float] = (-100.0, 3000.0),
        baseline_bounds: tuple[float, float] = (-100, -10),
        event_tolerences: dict[str, tuple[float, float]] = {'CC':(-1000.0, 1000.0)},
        trial_normalization: Literal['zscore', 'zero', 'none'] = 'zero',
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
            fit_using (Literal['OLS', 'IRLS', 'IRLS_no_intercept', 'OLS_no_intercept']): model used to fit isosbestic.
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
        if len(self.metadata['missing_events']) != 0:
            log.warning(f"Requested events {self.metadata['missing_events']} are missing")
        if align_to not in self.events.keys():
            raise ValueError(f'There are no {align_to} events present in data!')

        # step 2: preprocess signal with lowpass filter and dF/F strategy
        log.info(f"Preprocessing signal...")
        self.preprocess_signal(
            cutoff_frequency=cutoff_frequency, 
            order=order, 
            iso_correction_method=preprocess_method, 
            signal_normalization=preprocess_normalization,
            c=c,
            maxiter=maxiter,
            fit_using=fit_using,
            detrend_bleaching=detrend_bleaching
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
            trial_normalization=trial_normalization,
            time_error_threshold=time_error_threshold,
            check_overlap=check_overlap,
        )
        log.info(f"Done. Extracted {self.trial_data.n_trials} trials of {self.trial_data.n_times} size each.")

        # step 4: annotate and clean per-trial data
        log.info(f"Annotating trial data...")
        self.trial_data.add_obs_columns(self.metadata, keys=['rat', 'box', 'treatment', 'date', 'uid'])

        log.info(f"Done. {self.trial_data.n_trials} trials remain.")
        log.info(f"Pipeline complete.")

    # --- artifact ---
    def detect_artifacts(
            self,
            score_threshold: float = 10,
            expand_left_sec: float = 1,
            expand_right_sec: float = 1,
            buffer_sec: float = 2,
            jump_artifact_cutoff: float = 20,
            ignore_first_n: int = 1000,
            iso_cor_cutoff: float = 0.7,
            norm_before: Literal['none', 'zscore'] = 'none',
            n_chunks: int = 100,
            detrend: bool = True,
            ) -> None:
        
        sig = self.processed_sig.copy()
        iso = self.fitted_isosbestic.copy()

        if detrend:
            sig = self.detrend_photobleching(sig, 5)
            iso = self.detrend_photobleching(iso, 5)
        
        match norm_before:
            case 'none':
                pass
            case 'zscore':
                sig = (sig - sig.mean()) / sig.std()
                iso = (iso - iso.mean()) / iso.std()

        df = detect_artifacts_two_channel_ODS(
            signal = sig[ignore_first_n:],
            isosbestic = iso[ignore_first_n:],
            time = self.time[ignore_first_n:],
            score_threshold = score_threshold,
            expand_left_sec = expand_left_sec,
            expand_right_sec = expand_right_sec,
            buffer_sec = buffer_sec,
            jump_artifact_cutoff = jump_artifact_cutoff,
            n_chunks=n_chunks,
        )
        df = df.loc[df['isosbestic_cor'] >= iso_cor_cutoff]
        self.artifact_df = df

        info = {}
        info['artifact_score'] = self.calc_artifact_score()
        info['n_artifacts'] = df.shape[0]
        info['n_jumps'] = df.loc[df['type'] == 'jump'].sum()
        info['n_spikes'] = df.loc[df['type'] == 'spike'].sum()

        self.metadata.update(info)
        self.art_sig = sig.copy()
        self.art_iso = iso.copy()

    def calc_artifact_score(self) -> float:
        return np.sum(self.artifact_df['duration'] * self.artifact_df['amplitude'].abs())
    
    def plot_with_artifacts(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(self.time, self.art_sig, c='tab:blue', label='Exp', zorder=1)
        ax.plot(self.time, self.art_iso, c='tab:orange', label='Iso', zorder=0)
        
        df = self.artifact_df
        spike_bounds = df.loc[df['type'] == 'spike', ['start_idx', 'stop_idx']].to_numpy()
        jump_bounds = df.loc[df['type'] == 'jump', ['start_idx', 'stop_idx']].to_numpy()

        spike_idxs = intervals_to_flat_idx(spike_bounds)
        jump_idxs = intervals_to_flat_idx(jump_bounds)

        ax.scatter(self.time[spike_idxs], self.art_sig[spike_idxs], color='tab:red', label='Spike', zorder=2, s=2)
        ax.scatter(self.time[jump_idxs], self.art_sig[jump_idxs], color='pink', label='Jump', zorder=2, s=2)

        return ax
    
def SUGA_process_whole_directory(
    data_dir: str,
    output_dir: str,
    log_file: str | None = None,
    save_dashboards: bool = False,
    detect_artifacts: bool = False,

    trial_data_file: str = 'trials.h5ad',
    artifact_file: str = 'artifacts.csv',
    dashboard_folder: str = 'dashboard_plots',

    downsample: int = 20,
    boxes: list[str] = ['A', 'B'],
    event_labels: list[str] = ('OC', 'CC'),
    signal_label: str = '_500',
    isosbestic_label: str = '_450',
    annotation_filename: str = 'annotations.json',

    pipeline_kwargs: dict[str, Any] = {},
    artifact_kwargs: dict[str, Any] = {},
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
    artifact_path = os.path.join(output_dir, artifact_file)

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
    artifact_dfs: list[pd.DataFrame] = []

    # loop through every TDT folder and box
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
                    annot_filename=annotation_filename,
                    downsample=downsample,
                )
                exp.run_pipeline(
                    logger=log,
                    **pipeline_kwargs
                )
                log.info(f"Finished processing.")

                if detect_artifacts:
                    log.info(f"Detecting artifacts...")

                    exp.detect_artifacts(
                        **artifact_kwargs
                    )
                    exp.trial_data.add_obs_columns(
                        add_from=exp.metadata,
                        keys=['artifact_score', 'n_artifacts', 'n_jumps', 'n_spikes']
                    )

                    log.info(f"{exp.metadata['n_artifacts']} artifacts detected with score of {exp.metadata['artifact_score']}")
                    artifact_dfs.append(exp.artifact_df.assign(uid=exp.id))
                
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
                log.error(f"Error processing {tdt_folder}, box {box}: \n\t {e}")
                log.error(e, exc_info=True)
                i += 1
                continue

    log.info('Saving trial data...')
    trial_data.write_h5ad(trial_data_path)

    if detect_artifacts:
        log.info(f"Saving artifact information...")
        artifact_df = pd.concat(artifact_dfs)
        artifact_df.to_csv(artifact_path, index=False)

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