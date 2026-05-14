from typing import Callable, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from scipy.interpolate import PchipInterpolator

# --- utility ---
def median_abs_deviation(vals: np.ndarray, axis: int = 0) -> float | np.ndarray:
    '''Calculate the median absolute deviation along an axis'''
    med = np.median(vals, axis=axis, keepdims=True)
    abs_dev = np.abs(vals - med)
    mad = 1.4826 * np.median(abs_dev, axis=axis)
    return mad

def mad_score(vals: np.ndarray) -> np.ndarray:
    '''Calculate the MAD-score i.e. the Robust Z-score'''
    return 0.6745 * (vals - np.median(vals)) / median_abs_deviation(vals)

def z_score(vals: np.ndarray) -> np.ndarray:
    '''Calculate the Z-score'''
    return (vals - np.mean(vals)) / np.std(vals)

def boolean_mask_to_intervals(arr: np.ndarray) -> np.ndarray:
    '''Convert boolean mask to 2D array of index intervals of contigious True's'''
    arr = np.asarray(arr, dtype=bool)
    padded = np.r_[False, arr, False]
    diff = np.diff(padded.astype(int))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return np.column_stack([starts, ends])

def intervals_to_flat_idx(intervals: np.ndarray) -> np.ndarray:
    '''Convert 2D array of interval bounds to flat index'''
    if intervals.size == 0: return np.ndarray([], dtype=int)
    else: return np.concatenate([np.arange(start, stop) for start, stop in intervals]).ravel()

def intervals_to_bool_mask(arr: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    '''Convert 2D array of interval bouunds to boolean mask for input `arr`'''
    idxs = intervals_to_flat_idx(intervals)
    arr_idx = np.arange(arr.size)
    return np.isin(arr_idx, idxs)

# --- base classes ---
@dataclass(slots=True)
class ArtifactResult:
    '''Summarizes detected artifacts'''
    df: pd.DataFrame
    REQUIRED_COLUMNS = {
        "type",
        "start_idx",
        "stop_idx",
    }

    def __post_init__(self) -> None:
        if 'type' not in self.df.columns.to_list():
            self.df['type'] = 'unlabeled'

        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required artifact columns: {sorted(missing)}")

    @property   
    def types(self) -> list:
        return self.df['type'].unique().tolist()
    
    @property
    def groups(self) -> dict[str, pd.DataFrame]:
        res: dict[str, pd.DataFrame] = dict(tuple(self.df.groupby('type')))
        return res
    
    @property
    def group_intervals(self) -> dict[str, np.ndarray]:
        out = {}
        for label, idxs in self.df.groupby('type').indices.items():
            out[label] = self.df.filter(['start_idx', 'stop_idx']).iloc[idxs].to_numpy()
        return out
    
    @property
    def intervals(self) -> np.ndarray:
        return self.df[['start_idx', 'stop_idx']].to_numpy()
    
    @property
    def score(self) -> float:
        return np.sum(np.log10(self.df['amplitude']) * self.df['duration'])
    
    def calculate_metrics(
            self, 
            signal: np.ndarray, 
            time: np.ndarray,
            buffer: int = 100,
            ) -> None:
        n_artifacts = self.intervals.shape[0]

        # calc duration
        start_time = time[self.intervals[:, 0]] 
        stop_time = time[self.intervals[:, 1]]
        duration = stop_time - start_time

        # calc baseline before and after
        bs_before, bs_after, bs_diff = self.calc_baseline_change(signal=signal, buffer=buffer)

        # calc amplitude
        amplitude = np.zeros(shape=n_artifacts, dtype=float)
        for i, (start, stop) in enumerate(self.intervals):
            zeroed_sig = signal[start:stop] - bs_before[i]
            amplitude[i] = zeroed_sig[np.argmax(np.abs(zeroed_sig))]

        # save results
        self.df = self.df.assign(
            amplitude = amplitude,
            duration = duration,
            baseline_change = bs_diff,
            start_time = start_time,
            stop_time = stop_time,
        )

    def calc_baseline_change(
            self,
            signal: np.ndarray,
            buffer: int = 100,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # calc baseline before and after
        idx_before = self.intervals[:, 0][:, np.newaxis] + np.arange(-buffer+1, 1)
        idx_after = self.intervals[:, 1][:, np.newaxis] + np.arange(buffer)

        bs_before = np.median(signal[idx_before], axis=1)
        bs_after = np.median(signal[idx_after], axis=1)
        bs_diff = bs_after - bs_before

        return bs_before, bs_after, bs_diff

        

class ArtifactDetector(ABC):
    @abstractmethod
    def detect(self, signal: np.ndarray, reference: np.ndarray | None, time: np.ndarray,) -> ArtifactResult:
        pass

    def _calc_artifact_correlation(
            self,
            signal: np.ndarray,
            reference: np.ndarray,
            artifact_intervals: np.ndarray,
            ) -> np.ndarray:
        cor = np.zeros(shape=artifact_intervals.shape[0])
        for i, (start, stop) in enumerate(artifact_intervals):
            idxs = np.arange(start, stop)
            res = spearmanr(signal[idxs], reference[idxs])
            cor[i] = res.statistic # type: ignore
        return cor

class ArtifactCorrector:
    @abstractmethod
    def correct(self, signal: np.ndarray, time: np.ndarray, artifacts: ArtifactResult,) -> np.ndarray:
        pass
    
    def _correct_jump_artifacts(
            self,
            signal: np.ndarray,
            baseline_jumps: np.ndarray,
            stops: np.ndarray,
            ) -> np.ndarray:
        cum_drifts = np.r_[0.0, baseline_jumps.cumsum()]
        intervals = np.r_[0, stops, len(signal)]
        windows = [slice(intervals[i], intervals[i+1]) for i in range(0, intervals.size - 1)]

        corrected = signal.copy()
        for window, drift in zip(windows, cum_drifts):
            corrected[window] = corrected[window] - drift
        return corrected
    
# --- detectors ---
class ODS_Detector(ArtifactDetector):
    '''Detect spike and jump artifacts based on outlier derivatives'''
    def __init__(
            self,
            score_func: Callable = mad_score,
            score_threshold: float = 10,
            jump_score_threshold: float = 10,
            reference_cor_cutoff: float = 0.5,
            expand_sec: tuple[float, float] = (1, 2),
            buffer_sec: float = 1,
            n_chunks: int = 100,
            ) -> None:
        # --- save params ---
        self.score_func = score_func
        self.score_threshold = score_threshold
        self.jump_score_threshold = jump_score_threshold
        self.reference_cor_cutoff = reference_cor_cutoff

        self.expand_sec = expand_sec
        self.buffer_sec = buffer_sec
        self.n_chunks = n_chunks

    # --- main ---
    def detect(
            self,
            signal: np.ndarray,
            reference: np.ndarray | None,
            time: np.ndarray,
            ) -> ArtifactResult:
        # convert time params to indexs
        freq = len(signal) / (time[-1] - time[0])
        expand_left = int(self.expand_sec[0] * freq)
        expand_right = int(self.expand_sec[1] * freq)
        buffer = int(self.buffer_sec * freq)

        # detect outlier derivatives
        artifact_intervals = self._get_outlying_derivatives(
            signal=signal,
            score_threshold=self.score_threshold,
            expand_left=expand_left,
            expand_right=expand_right,
        )

        # package results
        artifacts = ArtifactResult(pd.DataFrame(
            artifact_intervals, columns=['start_idx', 'stop_idx']
        ))

        # calc artifact metrics
        artifacts.calculate_metrics(
            signal=signal,
            time=time,
            buffer=buffer,
        )

        # label artifacts jumps and spikes
        jump_scores = self._score_baseline_diffs_locally(
            signal=signal,
            artifacts=artifacts,
            n_chunks=self.n_chunks,
        )

        artifacts.df['jump_score'] = jump_scores
        artifacts.df.loc[np.abs(jump_scores) < self.jump_score_threshold, 'type'] = 'spike'
        artifacts.df.loc[np.abs(jump_scores) >= self.jump_score_threshold, 'type'] = 'jump'

        # if present calculate reference correlation
        if reference is not None:
            artifacts.df['reference_cor'] = self._calc_artifact_correlation(
                signal=signal,
                reference=reference,
                artifact_intervals=artifacts.intervals,
            )

            artifacts.df = artifacts.df.loc[
                artifacts.df['reference_cor'] >= self.reference_cor_cutoff
            ]
        
        return artifacts
    
    def _get_outlying_derivatives(
            self,
            signal: np.ndarray,
            score_threshold: float = 10,
            expand_left: float = 1,
            expand_right: float = 2,
            ) -> np.ndarray:
        # calc derivative
        delta = np.r_[0.0, np.diff(signal)]

        # find artifact boundaries
        scores = self.score_func(delta)
        outliers = (scores > score_threshold) | (scores < -score_threshold)
        intervals = boolean_mask_to_intervals(outliers)

        # expand artifact boundaries
        intervals[:, 0] = intervals[:, 0] - expand_left
        intervals[:, 1] = intervals[:, 1] + expand_right

        # combine overlaping artifacts
        intervals = boolean_mask_to_intervals(intervals_to_bool_mask(signal, intervals))
        return intervals 
    
    def _score_baseline_diffs_locally(
            self,
            signal: np.ndarray,
            artifacts: ArtifactResult,
            n_chunks: int = 100,
            ) -> np.ndarray:
        df = artifacts.df.copy()

        # calc artifact durations and centers
        art_idx_dur = df['stop_idx'] - df['start_idx']
        art_centers = df[['start_idx', 'stop_idx']].mean(axis=1).astype(int).to_numpy()

        # calc derivative
        derv = np.diff(signal, prepend=signal[0])

        # chunk signal and diff arrays
        chunk_size = int(np.ceil(signal.size / n_chunks))
        chunk_labels = np.clip(np.arange(signal.size) // chunk_size, 0, n_chunks - 1)
        art_chunks = np.clip(art_centers // chunk_size, 0, n_chunks - 1)
        
        # build chunk derv metrics
        derv_medians = np.array([np.median(derv[chunk_labels == c]) for c in range(n_chunks)])
        derv_mads = np.array([median_abs_deviation(derv[chunk_labels == c]) for c in range(n_chunks)])

        # score baseline diffs
        bs_diffs = df['baseline_change'].to_numpy() / art_idx_dur
        bs_diff_scores = 0.6745 * (bs_diffs - derv_medians[art_chunks]) / median_abs_deviation(derv_mads[art_chunks])
        return bs_diff_scores

# --- correctors ---
class Spline_Corrector(ArtifactCorrector):
    '''Correct spike and jump artifacts using a PCHIP shape-preserving interpolator'''
    def __init__(
            self,
            anchor_sec: tuple[float, float] = (0.2, 0.2),
            pre_norm: Literal['zscore', 'none'] = 'zscore',
            correct_spikes: bool = True,
            correct_jumps: bool = True,
            ) -> None:
        self.anchor_sec = anchor_sec
        self.pre_norm = pre_norm
        self.correct_spikes = correct_spikes
        self.correct_jumps = correct_jumps
        super().__init__()

    def correct(
            self,
            signal: np.ndarray,
            time: np.ndarray,
            artifacts: ArtifactResult,
            ) -> np.ndarray:
        # copy signal
        corrected = signal.copy()

        # convert time params to index
        freq = len(signal) / (time[-1] - time[0])
        anchor_left = int(self.anchor_sec[0] * freq)
        anchor_right = int(self.anchor_sec[1] * freq)

        # fill spike artifacts with spline if desired
        if self.correct_spikes:
            spikes = artifacts.groups['spike']
            for i, row in spikes.iterrows():
                corrected = self._fill_artifact_with_cubic_spline(
                    signal=corrected,
                    time=time,
                    start_idx=row['start_idx'],
                    stop_idx=row['stop_idx'],
                    anchor_left=anchor_left,
                    anchor_right=anchor_right,
                )

        # correct jump artifacts if desired
        if self.correct_jumps:
            # recalc baseline jumps
            bs_before, bs_after, bs_diff = artifacts.calc_baseline_change(signal=corrected)
            jump_mask = artifacts.df['type'] == 'jump'

            # apply baseline correction
            corrected = self._correct_jump_artifacts(
                signal=corrected,
                baseline_jumps=bs_diff[jump_mask],
                stops=artifacts.df.loc[jump_mask, 'stop_idx'],
            )

            # apply post spike correction
            for i, row in artifacts.df.loc[jump_mask].iterrows():
                corrected = self._fill_artifact_with_cubic_spline(
                    signal=corrected,
                    time=time,
                    start_idx=row['start_idx'],
                    stop_idx=row['stop_idx'],
                    anchor_left=anchor_left,
                    anchor_right=anchor_right,
                )


        # undue normalization transformation
        #corrected = (corrected * og_std) + og_mean
        
        return corrected

    def _fill_artifact_with_cubic_spline(
            self,
            signal: np.ndarray,
            time: np.ndarray,
            start_idx: int,
            stop_idx: int,
            anchor_left: int,
            anchor_right: int,
            ) -> np.ndarray:
        # construct anchors
        idxs = np.concatenate(
            [np.arange(start_idx - anchor_left, start_idx), np.arange(stop_idx, stop_idx + anchor_right)]
        )
        anchors_y = signal[idxs]
        anchors_x = time[idxs]

        # apply spline
        spline = PchipInterpolator(x=anchors_x, y=anchors_y)
        patch_idxs = np.arange(start_idx, stop_idx)
        patch = spline(time[patch_idxs])
        signal[patch_idxs] = patch
        return signal