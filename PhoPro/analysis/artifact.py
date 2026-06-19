"""Artifact detection and correction utilities for photometry traces."""

from typing import Callable, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from scipy.interpolate import PchipInterpolator

#######################
#region --- UTILITY ---
#######################
def median_abs_deviation(vals: np.ndarray, axis: int = 0) -> float | np.ndarray:
    """Calculate median absolute deviation along an axis.

    Parameters
    ----------
    vals : np.ndarray
        Input values.
    axis : int, default=0
        Axis along which to calculate the statistic.

    Returns
    -------
    float or np.ndarray
        MAD values scaled by ``1.4826``.
    """
    med = np.median(vals, axis=axis, keepdims=True)
    abs_dev = np.abs(vals - med)
    mad = 1.4826 * np.median(abs_dev, axis=axis)
    return mad

def mad_score(vals: np.ndarray) -> np.ndarray:
    """Calculate robust MAD scores.

    Parameters
    ----------
    vals : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        Robust Z-like scores based on the median absolute deviation.
    """
    return 0.6745 * (vals - np.median(vals)) / median_abs_deviation(vals)

def z_score(vals: np.ndarray) -> np.ndarray:
    """Calculate Z-scores.

    Parameters
    ----------
    vals : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        Standard Z-scores.
    """
    return (vals - np.mean(vals)) / np.std(vals)

def boolean_mask_to_intervals(arr: np.ndarray) -> np.ndarray:
    """Convert a boolean mask to half-open true intervals.

    Parameters
    ----------
    arr : np.ndarray
        One-dimensional boolean-like mask.

    Returns
    -------
    np.ndarray
        Integer array with shape ``(n_intervals, 2)`` containing half-open
        ``[start_idx, stop_idx)`` intervals.
    """
    arr = np.asarray(arr, dtype=bool)
    padded = np.r_[False, arr, False]
    diff = np.diff(padded.astype(int))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return np.column_stack([starts, ends])

def intervals_to_flat_idx(intervals: np.ndarray) -> np.ndarray:
    """Convert half-open intervals to flat indexes.

    Parameters
    ----------
    intervals : np.ndarray
        Integer array with shape ``(n_intervals, 2)`` containing half-open
        intervals.

    Returns
    -------
    np.ndarray
        One-dimensional integer indexes covered by ``intervals``.
    """
    if intervals.size == 0: return np.ndarray([], dtype=int)
    else: return np.concatenate([np.arange(start, stop) for start, stop in intervals]).ravel()

def intervals_to_bool_mask(arr: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Convert intervals to a boolean mask aligned to an input array.

    Parameters
    ----------
    arr : np.ndarray
        Input array used to determine mask length.
    intervals : np.ndarray
        Integer array with shape ``(n_intervals, 2)`` containing half-open
        intervals.

    Returns
    -------
    np.ndarray
        Boolean mask with length ``arr.size``.
    """
    idxs = intervals_to_flat_idx(intervals)
    arr_idx = np.arange(arr.size)
    return np.isin(arr_idx, idxs)

#endregion

############################
#region --- RESULT CLASS ---
############################
@dataclass(slots=True)
class ArtifactResult:
    """Summarize detected artifacts."""

    df: pd.DataFrame
    REQUIRED_COLUMNS = {
        "type",
        "start_idx",
        "stop_idx",
    }

    def __post_init__(self) -> None:
        """Validate artifact table columns."""
        if 'type' not in self.df.columns.to_list():
            self.df['type'] = 'unlabeled'

        missing = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required artifact columns: {sorted(missing)}")

    @property
    def types(self) -> list:
        """Artifact type labels present in the result."""
        return self.df['type'].unique().tolist()

    @property
    def groups(self) -> dict[str, pd.DataFrame]:
        """Artifact rows grouped by type."""
        res: dict[str, pd.DataFrame] = dict(tuple(self.df.groupby('type')))
        return res

    @property
    def group_intervals(self) -> dict[str, np.ndarray]:
        """Artifact intervals grouped by type."""
        out = {}
        for label, idxs in self.df.groupby('type').indices.items():
            out[label] = self.df.filter(['start_idx', 'stop_idx']).iloc[idxs].to_numpy()
        return out

    @property
    def intervals(self) -> np.ndarray:
        """Artifact intervals as ``start_idx`` and ``stop_idx`` columns."""
        return self.df[['start_idx', 'stop_idx']].to_numpy()

    @property
    def score(self) -> float:
        """Aggregate artifact score based on amplitude and duration."""
        return np.sum(np.log10(self.df['amplitude']) * self.df['duration'])

    def calculate_metrics(
            self,
            signal: np.ndarray,
            time: np.ndarray,
            buffer: int = 100,
            ) -> None:
        """Calculate and append artifact metrics.

        Parameters
        ----------
        signal : np.ndarray
            Signal containing the artifacts.
        time : np.ndarray
            Time values aligned to ``signal``.
        buffer : int, default=100
            Number of samples before and after each artifact used to estimate
            baseline change.
        """
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
        """Calculate pre/post baseline medians around artifacts.

        Parameters
        ----------
        signal : np.ndarray
            Signal containing the artifacts.
        buffer : int, default=100
            Number of samples before and after each artifact used to estimate
            baseline values.

        Returns
        -------
        bs_before : np.ndarray
            Median baseline before each artifact.
        bs_after : np.ndarray
            Median baseline after each artifact.
        bs_diff : np.ndarray
            ``bs_after - bs_before`` for each artifact.
        """

        # calc baseline before and after
        idx_before = self.intervals[:, 0][:, np.newaxis] + np.arange(-buffer+1, 1)
        idx_after = self.intervals[:, 1][:, np.newaxis] + np.arange(buffer)

        bs_before = np.median(signal[idx_before], axis=1)
        bs_after = np.median(signal[idx_after], axis=1)
        bs_diff = bs_after - bs_before

        return bs_before, bs_after, bs_diff

#endregion

################################
#region --- ABSTRACT CLASSES ---
################################
class ArtifactDetector(ABC):
    """Base class for artifact detectors."""

    @abstractmethod
    def detect(self, signal: np.ndarray, reference: np.ndarray | None, time: np.ndarray,) -> ArtifactResult:
        """Detect artifacts in a signal.

        Parameters
        ----------
        signal : np.ndarray
            Signal to inspect.
        reference : np.ndarray or None
            Optional reference trace.
        time : np.ndarray
            Time values aligned to ``signal``.

        Returns
        -------
        ArtifactResult
            Detected artifacts.
        """
        pass

    def _calc_artifact_correlation(
            self,
            signal: np.ndarray,
            reference: np.ndarray,
            artifact_intervals: np.ndarray,
            ) -> np.ndarray:
        """Calculate signal-reference correlation inside artifact intervals."""
        cor = np.zeros(shape=artifact_intervals.shape[0])
        for i, (start, stop) in enumerate(artifact_intervals):
            idxs = np.arange(start, stop)
            res = spearmanr(signal[idxs], reference[idxs])
            cor[i] = res.statistic # type: ignore
        return cor

class ArtifactCorrector:
    """Base class for artifact correctors."""

    @abstractmethod
    def correct(self, signal: np.ndarray, time: np.ndarray, artifacts: ArtifactResult,) -> np.ndarray:
        """Correct artifacts in a signal.

        Parameters
        ----------
        signal : np.ndarray
            Signal to correct.
        time : np.ndarray
            Time values aligned to ``signal``.
        artifacts : ArtifactResult
            Detected artifacts.

        Returns
        -------
        np.ndarray
            Corrected signal.
        """
        pass

    def _correct_jump_artifacts(
            self,
            signal: np.ndarray,
            baseline_jumps: np.ndarray,
            stops: np.ndarray,
            ) -> np.ndarray:
        """Apply cumulative baseline-jump correction."""
        cum_drifts = np.r_[0.0, baseline_jumps.cumsum()]
        intervals = np.r_[0, stops, len(signal)]
        windows = [slice(intervals[i], intervals[i+1]) for i in range(0, intervals.size - 1)]

        corrected = signal.copy()
        for window, drift in zip(windows, cum_drifts):
            corrected[window] = corrected[window] - drift
        return corrected

#endregion

#########################
#region --- DETECTORS ---
#########################
class ODS_Detector(ArtifactDetector):
    """Detect spike and jump artifacts from outlier derivatives."""

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
        """Initialize an outlier-derivative artifact detector.

        Parameters
        ----------
        score_func : Callable, default=mad_score
            Function used to score signal derivatives.
        score_threshold : float, default=10
            Absolute derivative score threshold for artifact detection.
        jump_score_threshold : float, default=10
            Absolute local baseline-difference score threshold used to label
            jumps versus spikes.
        reference_cor_cutoff : float, default=0.5
            Minimum signal-reference Spearman correlation required to keep an
            artifact when a reference trace is supplied.
        expand_sec : tuple[float, float], default=(1, 2)
            Seconds added to the left and right of detected derivative
            outliers.
        buffer_sec : float, default=1
            Seconds before and after artifacts used for baseline metrics.
        n_chunks : int, default=100
            Number of chunks used for local derivative scoring.
        """
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
        """Detect spike and jump artifacts.

        Parameters
        ----------
        signal : np.ndarray
            Signal to inspect.
        reference : np.ndarray or None
            Optional reference trace used to filter artifacts by correlation.
        time : np.ndarray
            Time values aligned to ``signal``.

        Returns
        -------
        ArtifactResult
            Detected and labeled artifacts.
        """
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
        """Find expanded intervals around outlier derivatives."""
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
        """Score artifact baseline jumps against local derivative statistics."""
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

#endregion

##########################
#region --- CORRECTORS ---
##########################
class Spline_Corrector(ArtifactCorrector):
    """Correct artifacts with PCHIP interpolation and jump drift removal."""

    def __init__(
            self,
            anchor_sec: tuple[float, float] = (0.2, 0.2),
            pre_norm: Literal['zscore', 'none'] = 'zscore',
            correct_spikes: bool = True,
            correct_jumps: bool = True,
            ) -> None:
        """Initialize a spline artifact corrector.

        Parameters
        ----------
        anchor_sec : tuple[float, float], default=(0.2, 0.2)
            Seconds of anchor signal used to the left and right of each
            artifact for interpolation.
        pre_norm : {'zscore', 'none'}, default='zscore'
            Stored normalization option. The current implementation does not
            apply the normalization transform.
        correct_spikes : bool, default=True
            Whether to interpolate spike artifacts.
        correct_jumps : bool, default=True
            Whether to apply jump drift correction and interpolate jump
            intervals.
        """
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
        """Correct artifacts in a signal.

        Parameters
        ----------
        signal : np.ndarray
            Signal to correct.
        time : np.ndarray
            Time values aligned to ``signal``.
        artifacts : ArtifactResult
            Detected artifacts with ``type``, ``start_idx``, and ``stop_idx``
            columns.

        Returns
        -------
        np.ndarray
            Corrected signal.
        """
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
        """Fill one artifact interval with a PCHIP spline."""
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

#endregion
