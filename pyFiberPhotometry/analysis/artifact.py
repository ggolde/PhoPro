import numpy as np
import pandas as pd

from typing import Callable
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

# --- artifact detection ---
def detect_artifacts_two_channel_ODS(
        signal: np.ndarray,
        isosbestic: np.ndarray,
        time: np.ndarray,
        score_func: Callable = mad_score,
        score_threshold: float = 10,
        expand_left_sec: float = 1,
        expand_right_sec: float = 2,
        buffer_sec: float = 1,
        jump_artifact_cutoff: float = 20,
        n_chunks: int = 100,
        ) -> pd.DataFrame:
    '''
    Detect spike and jump artifacts by detecting outlying derivative scores (ODS). 
    Args:
        signal (np.ndarray): experimental signal, photobleach detrended signal is preferred over raw.
        isosbestic (np.ndarray): isosbestic signal, photobleach detrended signal is preferred over raw.
        time (np.ndarray): time series for signal.
        score_func (Callable): function for derivative scoring.
        score_threshold (float): absolute value cutoff for a derivative score to be considered an outlier.
        expand_left_sec (float): how many seconds to expand artifact boundaries to the left.
        expand_right_sec (float): how many seconds to expand artifact boundaries to the right.
        buffer_sec (float): how many seconds before and after the artifact used to calculate baseline difference.
        jump_artifact_cutoff (float): absolute value cutoff for a baseline difference score to be considered significant,
         and the artifact to be labeled a jump, instead of a spike.
        n_chunks (int): number of chunks used to split signal into local enviroments
    Returns:
        tuple[np.ndarray, pd.DataFrame]: array of fitted curve and DataFrame of parameters and metrics.
    '''
    # convert seconds to indexs
    freq = len(signal) / (time[-1] - time[0])
    expand_left = int(expand_left_sec * freq)
    expand_right = int(expand_right_sec * freq)
    buffer = int(buffer_sec * freq)

    # detect artifacts
    artifact_intervals = detect_outlier_derivatives(
        signal=signal,
        score_func=score_func,
        score_threshold=score_threshold,
        expand_left=expand_left,
        expand_right=expand_right,
    )

    # calc artifact metrics and label
    artifact_df = calc_artifact_metrics(
        signal=signal,
        time=time,
        artifact_intervals=artifact_intervals,
        buffer=buffer,
        n_chunks=n_chunks,
        jump_artifact_cutoff=jump_artifact_cutoff,
    )

    # find cor with isosbestic
    artifact_df['isosbestic_cor'] = calc_artifact_correlation(signal, isosbestic, artifact_intervals)

    return artifact_df

def detect_outlier_derivatives(
        signal: np.ndarray,
        score_func = mad_score, 
        score_threshold: float = 10,
        expand_left: float = 1,
        expand_right: float = 1,
        ) -> np.ndarray:

    # calc derivative
    delta = np.r_[0.0, np.diff(signal)]

    # find artifact boundaries
    scores = score_func(delta)
    outliers = (scores > score_threshold) | (scores < -score_threshold)
    intervals = boolean_mask_to_intervals(outliers)

    # expand artifact boundaries
    intervals[:, 0] = intervals[:, 0] - expand_left
    intervals[:, 1] = intervals[:, 1] + expand_right

    # combine overlaping artifacts
    intervals = boolean_mask_to_intervals(intervals_to_bool_mask(signal, intervals))
    return intervals

def calc_artifact_correlation(
        arr1: np.ndarray,
        arr2: np.ndarray,
        artifact_intervals: np.ndarray,
        ) -> np.ndarray:
    cor = np.zeros(shape=artifact_intervals.shape[0])
    for i, (start, stop) in enumerate(artifact_intervals):
        idxs = np.arange(start, stop)
        res = spearmanr(arr1[idxs], arr2[idxs])
        cor[i] = res.statistic # type: ignore
    return cor

def score_baseline_diffs_locally(
        signal: np.ndarray,
        artifact_df: pd.DataFrame,
        n_chunks: int = 100,
        ) -> np.ndarray:
    df = artifact_df.copy()

    # calc artifact durations and centers
    art_idx_dur = df['stop_idx'] - df['start_idx']
    avg_idx_dur = int(art_idx_dur.mean())
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

def calc_artifact_metrics(
        signal: np.ndarray, 
        time: np.ndarray, 
        artifact_intervals: np.ndarray, 
        buffer: int = 100, 
        n_chunks: int = 100, 
        jump_artifact_cutoff: float = 5
        ) -> pd.DataFrame:
    n_artifacts = artifact_intervals.shape[0]

    # calc duration
    duration = time[artifact_intervals[:, 1]] - time[artifact_intervals[:, 0]] 

    # calc baseline before and after
    idx_before = artifact_intervals[:, 0][:, np.newaxis] + np.arange(-buffer+1, 1)
    idx_after = artifact_intervals[:, 1][:, np.newaxis] + np.arange(buffer)

    bs_before = np.mean(signal[idx_before], axis=1)
    bs_after = np.mean(signal[idx_after], axis=1)
    bs_diff = bs_after - bs_before

    # calc amplitude
    amplitude = np.zeros(shape=n_artifacts, dtype=float)
    for i, (start, stop) in enumerate(artifact_intervals):
        zeroed_sig = signal[start:stop] - bs_before[i]
        amplitude[i] = zeroed_sig[np.argmax(np.abs(zeroed_sig))]
    
    # package result
    res = pd.DataFrame(dict(
        type = 'unlabeled',
        amplitude = amplitude,
        duration = duration,
        baseline_change = bs_diff,
        baseline_before = bs_before,
        baseline_after = bs_before,
        start_idx = artifact_intervals[:, 0],
        stop_idx = artifact_intervals[:, 1],
    ))

    # score baseline diff locally
    res['baseline_change_score'] = score_baseline_diffs_locally(
        signal=signal,
        artifact_df=res,
        n_chunks=n_chunks,
    )

    # assign labels
    res.loc[res['baseline_change_score'].abs() < jump_artifact_cutoff, 'type'] = 'spike'
    res.loc[res['baseline_change_score'].abs() >= jump_artifact_cutoff, 'type'] = 'jump'
    
    return res

# --- artifact correction ---
def correct_jump_artifacts(
        signal: np.ndarray,
        baseline_jumps: np.ndarray,
        starts: np.ndarray,
        stops: np.ndarray,
        ) -> np.ndarray:
    cum_drifts = np.r_[0.0, baseline_jumps.cumsum()]
    intervals = np.r_[0, stops, len(signal)]
    windows = [slice(intervals[i], intervals[i+1]) for i in range(0, intervals.size - 1)]

    corrected = signal.copy()
    for window, drift in zip(windows, cum_drifts):
        corrected[window] = corrected[window] - drift
    return corrected


def fill_artifact_with_cubic_spline(
        signal: np.ndarray, 
        time: np.ndarray, 
        start: int, 
        stop: int, 
        anchor_left_sec: float = 1, 
        anchor_right_sec: float = 1, 
        ) -> np.ndarray:
    
    freq = len(signal) / (time[-1] - time[0])
    n_anchor_left = int(anchor_left_sec * freq)
    n_anchor_right = int(anchor_right_sec * freq)
    
    idxs = np.concatenate(
        [np.arange(start - n_anchor_left, start), np.arange(stop, stop + n_anchor_right)]
    )

    anchors_y = signal[idxs]
    anchors_x = time[idxs]

    spline = PchipInterpolator(x=anchors_x, y=anchors_y)
    patch_idxs = np.arange(start, stop)
    patch = spline(time[patch_idxs])
    
    signal[patch_idxs] = patch
    return signal

def correct_artifacts(
        signal: np.ndarray, 
        time: np.ndarray, 
        artifacts: pd.DataFrame, 
        anchor_left_sec: float = 1, 
        anchor_right_sec: float = 1,
        correct_jumps: bool = True,
        ) -> np.ndarray:
    corrected = signal.copy()

    if correct_jumps:
        jump_artifacts = artifacts.loc[artifacts['type'] == 'jump']
        corrected = correct_jump_artifacts(
            corrected, 
            baseline_jumps=jump_artifacts['baseline_change'], 
            starts=jump_artifacts['start_idx'], 
            stops=jump_artifacts['start_idx']
        )

    for i, row in artifacts.iterrows():
        corrected = fill_artifact_with_cubic_spline(
            corrected, time, 
            start=row['start_idx'], stop=row['stop_idx'], 
            anchor_left_sec=anchor_left_sec, anchor_right_sec=anchor_right_sec
        )
    return corrected