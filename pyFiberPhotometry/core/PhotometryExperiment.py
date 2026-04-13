from __future__ import annotations
from typing import Any, Literal
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import pandas as pd

from .PhotometeryData import PhotometryData

from ..utils.ops import (
    downsample_1d, reconstruct_time_points, 
    zscore_signal, center_signal, mad_norm_signal, amp_norm_signal,
    neg_bi_exponential_5
)
from ..utils.stats import OLS_fit, IRLS_fit, fit_photobleaching

class PhotometryExperiment:
    id: str
    metadata: dict[str, Any]
    events: dict[str, np.ndarray]
    raw_signal: np.ndarray
    raw_isosbestic: np.ndarray
    frequency: float
    time: np.ndarray
    trial_data: PhotometryData

    """
    Handles processing of raw photometry data.
    """
    def __init__( 
            self,
            raw_signal: np.ndarray,
            raw_isosbestic: np.ndarray,
            time: np.ndarray,
            frequency: float | None = None,
            events: dict = {},
            metadata: dict = {},
            ):
        self.raw_signal = raw_signal
        self.raw_isosbestic = raw_isosbestic
        self.time = time
        self.events = events
        self.metadata = metadata

        self.frequency = raw_signal.size / (self.time.max() - self.time.min()) if frequency is None else frequency
        self.metadata['frequency'] = frequency


    # --- import methods ---
    @classmethod
    def load_TDT(
            cls,
            data_folder: str,
            box: str,
            event_labels: list[str],
            signal_label: str,
            isosbestic_label: str,
            downsample: int = 10,
            ) -> PhotometryExperiment:
        """
        Load photometry data from TDT format.
        Args:
            data_folder (str): Path to the TDT block folder.
            box (str): TDT box identifier used in stream and epoc labels.
            event_labels (list[str]): Event labels to extract from epocs.
            signal_label (str): Base label for the signal channel.
            isosbestic_label (str): Base label for the isosbestic channel.
            downsample (int): downsampling factor for the raw streams (mean pooling).
        """
        from .PhotometryLoader import TDTLoader
        loader = TDTLoader(
            data_folder=data_folder,
            box=box,
            event_labels=event_labels,
            signal_label=signal_label,
            isosbestic_label=isosbestic_label,
            downsample=downsample,
        )
        return loader.load()
                
    # --- pipeline API ---
    def run_pipeline(self) -> None:
        """
        Run the full processing pipeline for 1 experiment (to be implemented in child classes).
        Args:
            None
        Returns:
            None
        """
        return
    
    def preprocess_signal(
            self,
            cutoff_frequency: float = 3.0, 
            order: int = 4,
            iso_correction_method: Literal['dF/F', 'dF', 'none'] = 'dF/F',
            signal_normalization: Literal['zscore', 'nullZ', 'none'] = 'none',
            fit_using: Literal['OLS', 'IRLS', 'IRLS_no_intercept', 'OLS_no_intercept'] = 'IRLS',
            maxiter: int = 1000,
            c: float | None = 3,
            detrend_bleaching: bool = False,
            ) -> None:
        """
        Low-pass filter and preprocess the signal using isosbestic fitting.
        Args:
            cutoff_frequency (float): Low-pass cutoff frequency in Hz.
            order (int): Butterworth filter order.
            iso_correction_method Literal['dF/F', 'dF', 'none']: Isosbestic correction method, 'dF/F' or 'dF'.
            signal_normalization (Literal['zscore', 'nullZ', 'none']): Method for whole signal normalization, 'none' recommended.
            fit_using (Literal['OLS', 'IRLS', 'IRLS_no_intercept', 'OLS_no_intercept']): model used to fit isosbestic.
            maxiter (int): maximum iterations of IRLS isosbestic fit.
            c (float): constant for IRLS fits, smaller values mean more agressive downweighting.
                1.4 <= c <= 3 is recommended unless there is large global drift is experimental signal, in which case c >= 10 is better. 
            detrend_bleaching (bool): whether or not to detrend signals for photobleaching using a (sig - bleach) / bleach scheme.
        Returns:
            None
        """
        # filter high frequency noise
        filt_sig = self.low_frequency_pass_butter(
            self.raw_signal, 
            self.frequency,
            cutoff_frequency=cutoff_frequency,
            order=order
        )
        # check if isosbestic present
        if self.raw_isosbestic is not None:
            filt_iso = self.low_frequency_pass_butter(
                self.raw_isosbestic, 
                self.frequency,
                cutoff_frequency=cutoff_frequency,
                order=order
            )
            
            # trim to minimum length
            min_len = min(filt_sig.size, filt_iso.size)
            filt_sig = filt_sig[:min_len]
            filt_iso = filt_iso[:min_len]

            # detrend photobleaching (optional)
            if detrend_bleaching:
                filt_sig = self.detrend_photobleching(filt_sig)
                filt_iso = self.detrend_photobleching(filt_iso)

            # fit isosbestic to signal
            fitted_iso, r2_val, coeff = self.fit_isosbestic_to_signal(
                filt_sig, 
                filt_iso,
                maxiter=maxiter,
                fit_using=fit_using,
                c=c,
            )
            # add relevant metadata
            self.metadata['isosbestic_fit'] = {'r2_val' : r2_val, 'coeffs' : coeff}
            self.fitted_isosbestic = fitted_iso
        else:
            filt_iso = None
            fitted_iso = None
        
        # save processed signals
        self.processed_sig = filt_sig.copy()
        self.processed_iso = filt_iso.copy()

        # procesing / normalization methods
        self.metadata['signal_processing_method'] = iso_correction_method 
        match iso_correction_method:
            case 'dF':
                self.signal = (filt_sig - fitted_iso)
            case 'dF/F':
                self.signal = (filt_sig - fitted_iso) / np.maximum(fitted_iso, np.finfo(np.float32).eps)
            case 'none':
                self.signal = filt_sig
            case _:
                raise ValueError(f'{iso_correction_method} isosbestic correction method not recognized!')

        match signal_normalization:
            case 'zscore':
                self.signal = (self.signal - np.mean(self.signal)) / np.std(self.signal)
            case 'nullZ':
                self.signal = self.signal / np.std(self.signal)
            case 'none':
                pass
            case _:
                raise ValueError(f'{signal_normalization} whole signal normalization method not recognized!')

        self.signal = self.signal.astype(np.float32, copy=False)
        return
    
    def extract_trial_data(
            self,
            align_to: str,
            center_on: list[str],
            trial_bounds: tuple[float, float],
            baseline_bounds: tuple[float, float] | None = None,
            event_tolerences: dict[str, tuple[float, float]] = {},
            trial_normalization: Literal['zscore', 'zero', 'mad', 'amp', 'none'] = 'none',
            check_overlap: bool = True,
            time_error_threshold: float = 0.1,
            event_conflict_logic: Literal['first', 'last', 'mean'] = 'first',
            ) -> None:
        """
        Build trial-wise windows, normalize, and store trial data.
        Args:
            align_to (str): Event label used to align and identify trial, should be one per trial.
            center_on (list[str]): Event labels to center trial windows on.
            trial_bounds (list[float, float]): Trial window bounds relative to ``center_on`` events.
            baseline_bounds (list[float, float]): Baseline window bounds relative to ``align_to`` event.
            event_tolerences (dict[str, tuple[float, float]]): Time tolerances for event annotation, relative to ``align_to``.
            trial_normalization (Literal['zscore', 'zero', 'none']): Normalization method for trial signals based on baselines.
            check_overlap (bool): Whether to throw an error multiple ``center_on`` events are found in the same trial.
            time_error_threshold (float): Maximum allowed mean timing error.
            event_conflict_logic (Literal['first', 'last', 'mean']): Logic for choosing center on event timestamps if multiple of the same event are present.
        Returns:
            None
        """        
        # validate inputs
        if align_to not in self.events: 
            raise KeyError(f"align_to '{align_to}' not found in events: {list(self.events)}")

        missing = [lab for lab in center_on if lab not in self.events]
        if missing: raise KeyError(f"center_on labels not found in events: {missing}")

        if baseline_bounds is None:
            calc_baselines = False
            if trial_normalization in ['zscore', 'zero']:
                raise ValueError(f'Baseline bounds have to be specified to for normalization method {trial_normalization}')
        else:
            calc_baselines = True

        # build trials around align_to event
        align_events = self.events[align_to].copy()
        if align_events.size == 0: raise ValueError(f"No '{align_to}' events found.")
        
        # annotate events based on tolerences
        events_selected = self.annotate_intervals(
            align_to=align_to,
            series=self.time, 
            centers=align_events,
            events=self.events, 
            tolorences=event_tolerences,
            logic=event_conflict_logic
            )
        
        # find window centers using the selected events
        trial_window_centers = self.find_window_centers(
            center_on=center_on, 
            align_on=align_to, 
            events=events_selected,
            check_overlap=check_overlap,
            )
        baseline_window_centers = align_events
        
        # construct trial and baseline windows
        raw_trial_signals, trial_times, trial_events = self.create_windows(
            signal=self.signal,
            time=self.time,
            events=events_selected,
            centers=trial_window_centers,
            bounds=trial_bounds
        )
        if calc_baselines:
            baseline_signals, baseline_times, baseline_events = self.create_windows(
                signal=self.signal,
                time=self.time,
                events=events_selected,
                centers=baseline_window_centers,
                bounds=baseline_bounds
            )
        else:
            baseline_signals, baseline_times, baseline_events = None, None, None

        # apply trial-wise normalization method
        match trial_normalization:
            case 'zscore':
                trial_signals = zscore_signal(raw_trial_signals, baseline_signals)
            case 'zero':
                trial_signals = center_signal(raw_trial_signals, baseline_signals)
            case 'mad':
                trial_signals = mad_norm_signal(raw_trial_signals, baseline_signals)
            case 'amp':
                trial_signals = amp_norm_signal(raw_trial_signals, baseline_signals)
            case 'none':
                trial_signals = raw_trial_signals.copy()
            case _:
                raise ValueError(f'{trial_normalization} trial-wise normalization method not recognized!')
        
        # reconstruct times for consistency
        trial_time_points = reconstruct_time_points(trial_bounds, self.frequency)

        if calc_baselines:
            baseline_time_points = reconstruct_time_points(baseline_bounds, self.frequency)
        else:
            baseline_time_points = None

        # save trial data
        self.raw_trial_signal = raw_trial_signals
        self.trial_data = PhotometryData.from_arrays(
            obs=pd.DataFrame(trial_events),
            data=trial_signals,
            time_points=trial_time_points,
            metadata=self.metadata.copy(),
        )

        if calc_baselines:
            self.baseline_data = PhotometryData.from_arrays(
                obs=pd.DataFrame(baseline_events),
                data=baseline_signals,
                time_points=baseline_time_points,
                metadata=self.metadata.copy(),
            )

        # check error in times
        time_err = trial_times.std(axis=0).mean()
        if time_err > time_error_threshold:
            raise ValueError(f'Error in rounded times is too high: {time_err} > {time_error_threshold}')

    # --- signal processing ---
    def low_frequency_pass_butter(
            self,
            signal: np.ndarray, 
            sample_frequency: float, 
            cutoff_frequency: float = 30.0, 
            order: int = 4,
            axis: int = 0,
        ) -> np.ndarray:
        """
        Apply a low-pass Butterworth filter to a 1D signal. Usually already done by photometry machine at 20 or 30 Hz.
        Args:
            signal (np.ndarray): Input signal array.
            sample_frequency (float): Sampling frequency in Hz.
            cutoff_frequency (float): Low-pass cutoff frequency in Hz.
            order (int): Butterworth filter order.
            axis (int): axis of array to perform a lowpass on.
        Returns:
            np.ndarray: Filtered signal.
        """        
        normalized_frequency = cutoff_frequency / (sample_frequency / 2)
        sos = butter(order, normalized_frequency, btype='low', output='sos') 
        return sosfiltfilt(sos, signal, axis=axis, padtype='odd', padlen=None)
    
    def fit_isosbestic_to_signal(
            self, 
            signal: np.ndarray, 
            isosbestic: np.ndarray, 
            maxiter: int = 1000,
            fit_using: Literal['OLS', 'IRLS', 'IRLS_no_intercept', 'OLS_no_intercept'] = 'IRLS',
            c: float | None = None,
            ) -> tuple[np.ndarray, float, Any]:
        """
        Fit the isosbestic channel to the signal using IRLS.
        Args:
            signal (np.ndarray): Filtered signal trace.
            isosbestic (np.ndarray): Filtered isosbestic trace.
            maxiter (int): maximum iterations of isosbestic fit.
            fit_using (Literal['OLS', 'IRLS', 'IRLS_no_intercept']): model used to fit isosbestic.
            c (float): constant for IRLS fits, smaller values mean more agressive downweighting.
                1.4 <= c <= 3 is recommended. 
        Returns:
            tuple[np.ndarray, float, np.ndarry]: Fitted isosbestic, R² value, and fit coefficients.
        """
        # fit using specified model
        match fit_using:
            case 'OLS':
                fitted_iso, params = OLS_fit(signal, isosbestic, add_intercept=True)
            case 'IRLS':
                fitted_iso, params = IRLS_fit(signal, isosbestic, maxiter=maxiter, c=c, add_intercept=True)
            case 'OLS_no_intercept':
                fitted_iso, params = OLS_fit(signal, isosbestic, add_intercept=False)
            case 'IRLS_no_intercept':
                fitted_iso, params = IRLS_fit(signal, isosbestic, maxiter=maxiter, c=c, add_intercept=False)
            case _:
                raise ValueError(f'{fit_using} fitting method not recognized!')

        # cheack output
        if np.isnan(fitted_iso).any():
            raise ValueError(f'NaN values in fitted isosbestic.')
        r2_val = r2_score(signal, fitted_iso)
        return fitted_iso, r2_val, params
    
    def detrend_photobleching(
            self,
            signal: np.ndarray,
            window_dur: float = 5,
            ) -> np.ndarray:
        '''
        Detrend photobleaching from a signal using a (signal - bleach) / bleach scheme.
        Args:
            signal (np.ndarray): 1D array of signal.
            window_dur (float): length of window in seconds used for sliding window median downsampling.
        Returns:
            detrended_signal (np.ndarray): (signal - bleach) / bleach.
        '''
        fitted_curve, fitted_params = self.fit_photobleaching_curve(signal=signal, window_dur=window_dur)
        return (signal - fitted_curve) / fitted_curve
    
    def fit_photobleaching_curve(
            self,
            signal: np.ndarray, 
            window_dur: float =  5,
            ) -> tuple[np.ndarray, list[float]]:
        """
        Fit a negative bi-exponential to photobleaching trend in raw signal and isosbestic
        using a sliding window median downsampling scheme.
        Args:
            signal (np.ndarray): array of signal to fit photobleaching curve to.
            window (float): length of window in seconds used for sliding window median downsampling.
        Returns:
            tuple[np.ndarray, pd.DataFrame]: array of fitted curve and DataFrame of parameters and metrics.
        """
        window_len = int(window_dur * self.frequency)
        fitted_curve, params = fit_photobleaching(signal, self.time, window=window_len)
        r2_val = r2_score(signal, fitted_curve)
        mse_val = mean_squared_error(signal, fitted_curve)
        
        return fitted_curve, params
    
    # --- extraction of trial-wise data ---
    def find_interval_bounds(self, series: np.ndarray, centers: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
        """
        Compute index bounds for windows around specified centers.
        Args:
            series (np.ndarray): Monotonic time-like series.
            centers (np.ndarray): Center times for each interval.
            bounds (tuple[int, int]): Relative lower and upper bounds in the same units as series.
        Returns:
            np.ndarray: Array of shape (n_intervals, 2) with left and right indices.
        """        
        low, high = bounds
        left_idxs = np.searchsorted(series, centers + low, side='left')
        right_idxs = np.searchsorted(series, centers + high, side='right')
        return np.c_[left_idxs, right_idxs]
    
    def find_timestamp_in_intervals(
            self, 
            timestamps: np.ndarray, 
            time_intervals: np.ndarray, 
            logic: Literal['first', 'last', 'mean'] = 'first',
            ) -> np.ndarray:
        """
        Find timestamps that falls within each time interval and select with customizable logic.
        Args:
            timestamps (np.ndarray): Sorted 1D array of event timestamps.
            time_intervals (np.ndarray): Array of shape (n_trials, 2) with [start, end] bounds.
            logic (Literal['first', 'last', 'mean'] | int): Logic for choosing event timestamps if multiple are present.
        Returns:
            np.ndarray: Array of first timestamps per interval or NaN if none exist.
        """        
        # tests which events are in interval
        timestamps = np.sort(timestamps)
        lo_idx = np.searchsorted(timestamps, time_intervals[:, 0], side="left")
        hi_idx = np.searchsorted(timestamps, time_intervals[:, 1], side="right")
        in_interval = lo_idx < hi_idx

        # perform choice logic
        out = np.full(len(time_intervals), np.nan, float)
        match logic:
            case 'first':
                out[in_interval] = timestamps[lo_idx[in_interval]]
            case 'last':
                out[in_interval] = timestamps[hi_idx[in_interval]]
            case 'mean':
                counts = hi_idx - lo_idx
                in_interval = counts > 0

                csum = np.concatenate(([0.0], np.cumsum(timestamps)))
                sums = csum[hi_idx] - csum[lo_idx]

                out[in_interval] = sums[in_interval] / counts[in_interval]
            case _:
                raise ValueError(f'Multiple event selection logic, {logic}, is not recognized')
        return out
    
    def annotate_intervals(
            self, 
            align_to: str, 
            series: np.ndarray, 
            centers: np.ndarray, 
            events: dict[str, np.ndarray], 
            tolorences: dict[str, np.ndarray],
            logic: Literal['first', 'last', 'mean'] = 'first',
            ) -> dict[str, np.ndarray]:
        """
        Annotate intervals around centers with event timestamps within tolerances.
        Args:
            align_to (str): Label of the primary alignment event.
            series (np.ndarray): Monotonic time-like series.
            centers (np.ndarray): Center times for each trial.
            events (dict[str, np.ndarray]): Mapping of event labels to timestamps.
            tolorences (dict[str, np.ndarray]): Mapping of labels to time tolerances.
            logic (Literal['first', 'last', 'mean'] | int): Logic for choosing event timestamps if multiple are present.
        Returns:
            dict[str, np.ndarray]: Mapping from labels to aligned event times per trial.
        """        
        out = {}
        out[align_to] = centers
        for label, bounds in tolorences.items():
            timestamps = events.get(label)
            interval_bounds = self.find_interval_bounds(series=series, centers=centers, bounds=bounds)
            time_intervals = series[interval_bounds]
            out[label] = self.find_timestamp_in_intervals(timestamps=timestamps, time_intervals=time_intervals, logic=logic)
        return out

    def create_windows(self, signal: np.ndarray, time: np.ndarray, events: dict[str, np.ndarray], centers: np.ndarray, bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Slice signal and time into fixed-length windows around centers.
        Args:
            signal (np.ndarray): Full preprocessed signal trace.
            time (np.ndarray): Associated time vector.
            events (dict[str, np.ndarray]): Per-label event times for each trial.
            centers (np.ndarray): Window center times for each trial.
            bounds (tuple[int, int]): Relative bounds [low, high] around centers.
        Returns:
            tuple[np.ndarray, np.ndarray, dict]: Signal windows, centered time windows, and centered events.
        """        
        # find target window bounds
        low, high = bounds
        left_idxs = np.searchsorted(time, centers + low, side='left')

        # calculate minimum window size to ensure stable sizes
        target_len = np.floor((bounds[1] - bounds[0]) * self.frequency).astype(int)
        window_idxs = left_idxs[:, None] + np.arange(target_len)[None, :]

        # slice and stack + center time and events
        signal_windows = signal[window_idxs]
        time_windows = time[window_idxs] - centers[:, np.newaxis]
        events_centered = {k : v - centers for k, v in events.items()}
        return signal_windows, time_windows, events_centered

    def find_window_centers(self, center_on: str | list[str], align_on: str, events: dict[str, np.ndarray], check_overlap: bool = True) -> np.ndarray:
        """
        Determine window centers based on center_on events or fallback to align_on.
        Args:
            center_on (str | list[str]): Event labels used as preferred centers.
            align_on (str): Fallback event label used when center_on is missing.
            events (dict[str, np.ndarray]): Mapping from labels to event times per trial.
            check_overlap (bool): Whether to throw an error multiple ``center_on`` events are found in the same trial.
        Returns:
            np.ndarray: Center times per trial.
        """        
        # center_on events should be non-overlaping
        # if no center_on events present, center on align_on
        if isinstance(center_on, str):
            center_on = [center_on]

        centers = events[align_on].copy()
        overlap = np.full_like(centers, True, dtype=bool)
        for label in center_on:
            arr = events[label]
            event_not_nan = ~np.isnan(arr)
            centers[event_not_nan] = arr[event_not_nan]
            overlap &= event_not_nan
        if check_overlap and overlap.any():
            at_idxs = np.where(overlap)
            culprits = {k : events[k][at_idxs] for k in center_on}
            culprits_in_og_events = {k : self.events[k][np.searchsorted(self.events[k], culprits[k])] for k in center_on}
            raise ValueError(f'Center_on events over lap in trials {np.where(overlap)}, with culprits: {culprits_in_og_events}')
        return centers
    
    # --- poor signal checks ---
    def median_centered_abs_max_check(self, trial_signal: np.ndarray, threshold: float = 0.075):
        median_centered = trial_signal - np.median(trial_signal, axis=1, keepdims=True)
        abs_max = np.abs(median_centered).max(axis=1)
        is_poor_signal = np.mean(abs_max) < threshold
        return is_poor_signal

    # --- graphing ---
    def dashboard(self, save: str | None = None) -> None:
        """
        Quickly plot the raw, fitted, and processed signal, isosbestic, and fitted photobleaching curve (if avaliable).
        Args:
            save (str, None): Path to save figure, if None figure does not save.
        Returns:
            None.
        """
        fig, (ax1, ax2) = plt.subplots(
            ncols=1, nrows=2, 
            sharex=True, figsize=(6, 6), dpi=140,
            gridspec_kw={'height_ratios': [3, 1]})
        fig.tight_layout()

        downsample_factor = 20
        x = downsample_1d(self.time, downsample_factor)
        raw_sig = downsample_1d(self.raw_signal, downsample_factor)
        raw_iso = downsample_1d(self.raw_isosbestic, downsample_factor)
        fit_iso = downsample_1d(self.fitted_isosbestic, downsample_factor)
        fit_params = getattr(self, 'fitted_params', None)
        final_sig = downsample_1d(self.signal, downsample_factor)
        
        # raw and fitted signals
        ax1: matplotlib.axes.Axes
        ax1.plot(x, raw_sig, label='Raw Signal', c='#1f77b4')
        ax1.plot(x, raw_iso, label='Raw Iso.', c="#4B4B4B", alpha=0.9)
        ax1.plot(x, fit_iso, label='Fitted Iso.', c='#ff7f0e', alpha=0.9)
        if fit_params is not None:
            ax1.plot(x, neg_bi_exponential_5(x, *fit_params), label='Fitted Curve', c="#920000")
        ax1.legend()

        # processed signal
        ax2: matplotlib.axes.Axes
        y_pad_factor = 2.5
        middle_third = np.array_split(final_sig, 3)[1]
        y_high = np.max(middle_third)
        y_low = np.min(middle_third)
        print(y_high)
        ax2.plot(x, final_sig, label='Processed Signal', c='#1f77b4')
        ax2.set_ylim(bottom=y_low*y_pad_factor, top=y_high*y_pad_factor)
        ax2.legend()

        # annotate
        ax1.set_title(
        f"Dashboard for {getattr(self, 'id', 'Unnamed')}"
        )
        ax1.set_ylabel('Signal amplitude (a.u.)')
        ax2.set_ylabel(f"{self.metadata.get('signal_processing_method', 'NOT FOUND')}")
        ax2.set_xlabel('Time (s)')

        if save is not None:
            plt.savefig(save, bbox_inches='tight')