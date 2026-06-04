from __future__ import annotations
from collections.abc import Sequence
from numbers import Real
from typing import Any, Literal, Callable
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import pandas as pd

from .PhotometeryData import PhotometryData

from ..analysis.artifact import ArtifactResult, ArtifactDetector, ArtifactCorrector
from ..utils.ops import (
    downsample_1d, reconstruct_time_points, 
    zscore_signal, center_signal, mad_norm_signal, amp_norm_signal,
    neg_bi_exponential_5
)
from ..utils.stats import OLS_fit, IRLS_fit, fit_photobleaching
from ..utils.window import create_windows_interp_1D, create_windows_nearest_1D, WindowResult

class PhotometryExperiment:
    """Handle processing of raw photometry data."""

    id: str
    metadata: dict[str, Any]
    events: dict[str, np.ndarray]
    raw_signal: np.ndarray
    raw_isosbestic: np.ndarray | None
    time: np.ndarray
    frequency: float
    artifacts: ArtifactResult | None

    signal: np.ndarray
    fitted_ref: np.ndarray
    trial_data: PhotometryData

    def __init__( 
            self,
            raw_signal: np.ndarray,
            raw_isosbestic: np.ndarray | None,
            time: np.ndarray,
            events: dict = {},
            metadata: dict = {},
            frequency: float | None = None,
            ):
        """Initialize a photometry experiment.

        Args:
            raw_signal (np.ndarray): Raw signal channel values.
            raw_isosbestic (np.ndarray): Raw isosbestic channel values. If
                ``None``, experiment is assumed to be single channel.
            time (np.ndarray): Time points corresponding to the raw signals.
            events (dict, optional): Mapping of event labels to timestamp
                arrays. Defaults to ``{}``.
            metadata (dict, optional): Additional experiment metadata. Defaults
                to ``{}``.
            frequency (float | None, optional): Sampling frequency in Hz. If
                ``None`` (default), it is estimated from ``raw_signal`` and
                ``time``.
        """
        # save inputs
        self.raw_signal = raw_signal
        self.raw_isosbestic = raw_isosbestic
        self.time = time
        self.events = events
        self.metadata = metadata
        self.frequency = raw_signal.size / (self.time.max() - self.time.min() + 1) if frequency is None else frequency

        self.signal = None
        self.fitted_ref = None
        self.trial_data = None
        
        # add relevant metadata
        self.metadata['frequency'] = self.frequency
        self.metadata['channel_mode'] = self.channel_mode

    # --- properties ---
    @property
    def has_isosbestic(self) -> bool: return self.raw_isosbestic is not None
    @property
    def channel_mode(self) -> Literal['single', 'dual']: return 'dual' if self.has_isosbestic else 'single'

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
        """Load photometry data from TDT format.

        Args:
            data_folder (str): Path to the TDT block folder.
            box (str): TDT box identifier used in stream and epoc labels.
            event_labels (list[str]): Event labels to extract from epocs.
            signal_label (str): Base label for the signal channel.
            isosbestic_label (str): Base label for the isosbestic channel.
            downsample (int, optional): Downsampling factor for the raw streams
                (mean pooling). Defaults to ``10``.

        Returns:
            PhotometryExperiment: Loaded experiment instance.
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
    def preprocess_signal(
            self,
            cutoff_frequency: float = 3.0, 
            order: int = 4,
            signal_normalization: Literal['zscore', 'nullZ', 'none'] | Callable = 'none',
            correction_method: Literal['dF/F', 'dF', 'dB/B', 'dB', 'none'] | Callable = 'dF/F',
            fit_using: Literal['OLS', 'IRLS', 'IRLS_no_intercept', 'OLS_no_intercept'] | Callable = 'IRLS',
            maxiter: int = 1000,
            c: float | None = 3,
            artifact_detector: ArtifactDetector | None = None,
            artifact_corrector: ArtifactCorrector | None = None,
            ) -> None:
        """Low-pass filter and preprocess the signal using isosbestic fitting.

        Args:
            cutoff_frequency (float, optional): Low-pass cutoff frequency in Hz.
                Values between 1 and 5 recommended. Defaults to ``3.0``.

            order (int, optional): Butterworth filter order. 
                Values >3 are recommended. Defaults to ``4``.

            correction_method (Literal['dF/F', 'dF', 'dB/B', 'dB', 'none'] | Callable, optional):
                Reference trace correction method:  
                * ``'dF/F'`` and ``'dB/B`` = (signal - fitted reference) / fitted reference
                * ``'dF'`` and ``'dB'`` = (signal - fitted reference) 
                For dual channel experiments the reference is the isosbestic, for single channel
                it is a fit photobleaching curve. Use ``'dF/F'`` or ``'dF'`` for dual channel
                and ``'dB/B'`` or ``'dB'`` for single channel.
                A custom function that takes in the positional arguements ``signal``, ``fitted_reference``
                and returns a 1D ``np.ndarray`` can also be passed. Defaults to ``'dF/F'``.

            signal_normalization (Literal['zscore', 'nullZ', 'none'] | Callable, optional): 
                Method for whole-signal normalization; ``'none'``
                for dF/F and ``'zscore'`` for dF is recommended. 
                A custom function that takes in the positional arguements ``signal``
                and returns a 1D ``np.ndarray`` can also be passed.
                Defaults to ``'none'``.

            fit_using (Literal['OLS', 'IRLS', 'IRLS_no_intercept', 'OLS_no_intercept'] | Callable, optional): 
                Model used to fit isosbestic to experimental signal. IRLS methods recommended. 
                Use a no intercept type model if large global change is present in the experimental signal.
                A custom function that takes in the positional arguements ``signal``, ``isosbestic``
                and returns a 1D ``np.ndarray`` and a sequence of params can also be passed.
                Note custom functions will only apply to isosbestic fits (i.e. dual channel experiments).
                Defaults to ``'IRLS'``.

            maxiter (int, optional): Maximum iterations of the IRLS isosbestic
                fit. Defaults to ``1000``.

            c (float | None, optional): Constant for IRLS fits; smaller values
                mean more agressive downweighting. ``1.4 <= c <= 3`` is
                recommended unless there is large global drift in the
                experimental signal, in which case large values (>5) are better.
                Defaults to ``3``.

            artifact_detector (ArtifactDetector | None, optional): Detector object with method
                ``.detect(signal, reference, time)`` used to detect artifacts.
                If ``None``, detection is skipped. Defaults to ``None``.

            artifact_corrector (ArtifactCorrector | None, optional): Corrector object with method
                ``.correct(signal, time, artifacts)`` used to correct artifacts.
                If ``None``, correction is skipped. Defaults to ``None``.

        Returns:
            None

        Raises:
            ValueError: 
                * If ``correction_method`` is incompatible with ``self.channel_mode``
                * If ``artifact_corrector`` specified but not ``artifact_detector``
        """
        # validate inputs
        dual_channel_methods = ['dF/F', 'dF']
        single_channel_methods = ['dB/B', 'dB']

        if self.channel_mode == 'single' and correction_method in dual_channel_methods:
            raise ValueError(f'Correction methods {",".join(dual_channel_methods)} are for dual channel experiments.')
        
        if self.channel_mode == 'dual' and correction_method in single_channel_methods:
            raise ValueError(f'Correction methods {",".join(single_channel_methods)} are for single channel experiments.')
        
        if artifact_corrector is not None and artifact_detector is None:
            raise ValueError(f'artifact_detector not specified but artifact_corrector is.')

        # apply lowpass butterworth filter
        filt_sig = self.low_frequency_pass_butter(
            self.raw_signal, 
            self.frequency,
            cutoff_frequency=cutoff_frequency,
            order=order
        )

        if self.channel_mode == 'dual':
            # apply lowpass to isosbestic
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

            # fit isosbestic to signal
            fitted_ref, r2_val, coeffs = self.fit_isosbestic_to_signal(
                filt_sig, 
                filt_iso,
                maxiter=maxiter,
                fit_using=fit_using,
                c=c,
            )
            reference_type = 'isosbestic'
        
        elif self.channel_mode == 'single':
            # fit photobleaching curve
            fitted_ref, r2_val, coeffs = self.fit_photobleaching_curve(
                signal=filt_sig
            )
            reference_type = 'photobleaching'

        else:
            raise ValueError(f"Channel mode, {self.channel_mode}, not recognized.")

        # save relevant data
        self.metadata['reference_fit'] = dict(
            type = reference_type,
            r2_val = r2_val,
            coeffs = coeffs,
        )
        self.filt_sig = filt_sig
        self.fitted_ref = fitted_ref
    
        # apply reference curve correction
        signal = self._apply_correction_method(
            correction_method=correction_method,
            signal=filt_sig,
            fitted_ref=fitted_ref,
        )

        # apply signal normalization
        signal = self._apply_signal_normalization(
            signal_normalization=signal_normalization,
            signal=signal,
        )

        # detect artifacts
        artifacts = None
        if artifact_detector is not None:
            reference = None if self.channel_mode == 'single' else self.fitted_ref

            artifacts = self._detect_artifacts(
                detector=artifact_detector,
                signal=signal,
                reference=reference,
                time=self.time,
            )

            # correct artifacts
            if artifact_corrector is not None:
                signal = self._correct_artifacts(
                    corrector=artifact_corrector,
                    artifacts=artifacts,
                    signal=signal,
                    time=self.time,
                )

        # save results and end pipeline
        self.signal = signal
        self.artifacts = artifacts

        if callable(correction_method): self.metadata['correction_method'] = correction_method.__name__
        else: self.metadata['correction_method'] = correction_method

        return
        
    def extract_trial_data(
            self,
            align_to: str | Sequence[str] | float | Sequence[float],
            trial_bounds: tuple[float, float],
            center_on: str | Sequence[str] | None = None,
            baseline_bounds: tuple[float, float] | None = None,
            event_tolerences: dict[str, tuple[float, float] | None] = {},
            trial_normalization: Literal['zscore', 'zero', 'mad', 'amp', 'none'] | Callable = 'none',
            check_overlap: bool = True,
            time_error_threshold: float = 0.1,
            window_alignment: Literal['nearest', 'interp'] = 'nearest',
            invalid_window_policy: Literal['drop', 'error'] = 'drop',
            event_conflict_logic: Literal['first', 'last', 'mean'] = 'first',
            ) -> None:
        """Build trial-wise windows, normalize, and store trial data.

        Args:
            align_to (str | Sequence[str] | float | Sequence[float]): 
                Event label(s) or timestamp(s) used to build windows for
                and identify trials. Should be one per trial. If multiple
                event labels, an ``align_event`` column will be added to
                ``self.trial_data`` spcifying which alignment event the 
                trial originated from.

            center_on (str | Sequence[str] | None = None): 
                Event label(s) to center trial windows on.
                Events should be mutually exclusive (i.e. lever press choice).
                If no ``center_on`` events are present within an identified
                trial, ``align_to`` will be centered on, as is also the case
                when ``center_on`` is ``None``.

            trial_bounds (tuple[float, float]): Trial window bounds relative to
                ``center_on`` or ``align_to`` events.

            baseline_bounds (tuple[float, float] | None, optional): Baseline
                window bounds relative to ``align_to`` event used for per-trial
                normalizations. Defaults to ``None``.

            event_tolerences (dict[str, tuple[float, float] | None] | None, optional): 
                Time tolerances for event annotation, relative to ``align_to``.
                Defaults to ``{}``. If ``center_on`` events not specified,
                there are automatically added with tolerence equal to trial_bounds.

            trial_normalization (Literal['zscore', 'zero', 'mad', 'amp', 'none'] | Callable, optional):
                Normalization method for trial signals based on baselines.
                A custom function that takes in the positional arguements 
                ``trial_signals``, ``baseline_signals`` and returns a 2D 
                ``np.ndarray`` of shape (n_trials, n_times) can also be passed.
                Defaults to ``'none'``.

            check_overlap (bool, optional): Whether to throw an error when
                multiple ``center_on`` events are found in the same trial.
                Defaults to ``True``.

            time_error_threshold (float, optional): Maximum allowed mean timing
                error. Defaults to ``0.1``.

            window_alignment (Literal['nearest', 'interp'], optional):
                Stategy for aligning trial times. In ``'nearest'`` mode, events times
                are rounded to the nearest sampling times, giving a maximum
                event alignment error of +/- 0.5/frequency. In ``'interp'`` mode,
                signals are linearly interpolated to an exact event-centered time
                grid, removing event alignment error but introduction signal
                interpolation error. Use ``'nearest'`` if event times are already
                locked to time sampling points.

            invalid_window_policy (Literal['drop', 'error'], optional):
                Policy for handling trials whose windows extend outside the signal
                bounds. ``'drop'`` drops the invalid windows while ``'error'`` raises
                an error if any invalid windows are present.

            event_conflict_logic (Literal['first', 'last', 'mean'], optional):
                Logic for choosing event timestamps if multiple of the
                same event are present within the same trial and within tolerence. 
                Defaults to ``'first'``.

        Returns:
            None

        Raises:
            ValueError: If ``baseline_bounds`` is ``None`` and ``trial_normalization``
                requires baselines.
        """
        # validate inputs
        if baseline_bounds is None:
            calc_baselines = False
            if trial_normalization in ['zscore', 'zero', 'mad', 'amp']:
                raise ValueError(
                    f'Baseline bounds have to be specified for normalization method {trial_normalization}'
                )
        else:
            calc_baselines = True

        # handle alignment
        align_label, align_events, align_obs = self._resolve_alignment(align_to)
        
        # coerce centering
        center_on = self._coerce_centering(center_on)
        
        # coerce tolerences
        event_tolerences = self._coerce_tolerances(event_tolerences, center_on, trial_bounds)
        
        # annotate events based on tolerences
        events_selected = self._annotate_intervals(
            align_label=align_label,
            series=self.time,
            centers=align_events,
            events=self.events, 
            tolorences=event_tolerences,
            logic=event_conflict_logic
        )
        
        # find window centers using the selected events
        trial_window_centers = self._find_window_centers(
            center_on=center_on, 
            align_events=align_events, 
            events=events_selected,
            check_overlap=check_overlap,
        )
        
        # construct trial windows
        trial_windows = self._create_windows(
            signal=self.signal,
            time=self.time,
            events=events_selected,
            centers=trial_window_centers,
            bounds=trial_bounds,
            strategy=window_alignment,
        )

        # do the same for baseline
        if calc_baselines:
            baseline_windows = self._create_windows(
                signal=self.signal,
                time=self.time,
                events=events_selected,
                centers=align_events,
                bounds=baseline_bounds,
                strategy=window_alignment,
            )
        else:
            baseline_windows = None

        # handle invalid windows
        valid_trial_mask = self._handle_invalid_windows(
            trial_windows=trial_windows,
            baseline_windows=baseline_windows,
            policy=invalid_window_policy,
        )

        # apply trial-wise normalization method
        processed_trial_signals = self._apply_trial_normalization(
            trial_normalization=trial_normalization, 
            trial_windows=trial_windows, 
            baseline_windows=baseline_windows,
        )

        # save trial data
        trial_obs = pd.concat(
                [align_obs.loc[valid_trial_mask].reset_index(drop=True),
                 pd.DataFrame(trial_windows.events).reset_index(drop=True)], 
                axis=1,
            )
        
        self.trial_data = PhotometryData.from_arrays(
            obs=trial_obs,
            data=processed_trial_signals,
            time_points=trial_windows.time_grid,
            metadata=self.metadata.copy(),
        )

        # assign trial numbers
        self.trial_data.obs.insert(0, 'trial_num', np.arange(self.trial_data.n_trials, dtype=int) + 1)

        # save baseline data if applicable
        if calc_baselines:
            baseline_obs = pd.concat(
                [align_obs.loc[valid_trial_mask].reset_index(drop=True),
                 pd.DataFrame(baseline_windows.events).reset_index(drop=True)], 
                axis=1,
            )
            self.baseline_data = PhotometryData.from_arrays(
                obs=baseline_obs,
                data=baseline_windows.signals,
                time_points=baseline_windows.time_grid,
                metadata=self.metadata.copy(),
            )

        # check error in times
        time_err = trial_windows.times.std(axis=0).mean()
        if time_err > time_error_threshold:
            raise ValueError(f'Error in rounded times is too high: {time_err} > {time_error_threshold}')
        
    def _coerce_tolerances(
            self,
            event_tolerences: dict[str, tuple[float, float] | None] | None,
            center_on: list[str],
            trial_bounds: tuple[float, float],
            ) -> dict[str, tuple[float, float]]:
        
        event_tolerences = dict(event_tolerences or {})
        # ensure center on events present
        for label in center_on:
            event_tolerences.setdefault(label, trial_bounds)

        # change None to maximum tolerance
        for label, tolerance in event_tolerences.items():
            if tolerance is None:
                event_tolerences[label] = trial_bounds

        new_tolerences: dict[str, tuple[float, float]] = event_tolerences
        return new_tolerences

    # --- signal processing ---
    def low_frequency_pass_butter(
            self,
            signal: np.ndarray, 
            sample_frequency: float, 
            cutoff_frequency: float = 30.0, 
            order: int = 4,
            axis: int = 0,
        ) -> np.ndarray:
        """Apply a low-pass Butterworth filter to a signal.

        Args:
            signal (np.ndarray): Input signal array.
            sample_frequency (float): Sampling frequency in Hz.
            cutoff_frequency (float, optional): Low-pass cutoff frequency in Hz.
                Defaults to ``30.0``.
            order (int, optional): Butterworth filter order. Defaults to ``4``.
            axis (int, optional): Axis of the array to perform a low-pass on.
                Defaults to ``0``.

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
            fit_using: Literal['OLS', 'IRLS', 'IRLS_no_intercept', 'OLS_no_intercept'] | Callable = 'IRLS',
            maxiter: int = 1000,
            c: float | None = None,
            ) -> tuple[np.ndarray, float, Any]:
        """Fit the isosbestic channel to the signal.

        Args:
            signal (np.ndarray): Filtered signal trace.
            isosbestic (np.ndarray): Filtered isosbestic trace.
            fit_using (Literal['OLS', 'IRLS', 'IRLS_no_intercept', 'OLS_no_intercept'] | Callable, optional): 
                Model used to fit isosbestic. Defaults to ``'IRLS'``.
            maxiter (int, optional): Maximum iterations of IRLS isosbestic fit.
                Defaults to ``1000``.
            c (float | None, optional): Constant for IRLS fits; smaller values
                mean more agressive downweighting. ``1.4 <= c <= 3`` is
                recommended. Defaults to ``None``.

        Returns:
            tuple[np.ndarray, float, Any]: Fitted isosbestic, R-squared value,
                and fit coefficients.
        """
        # fit using specified model
        match fit_using:
            case func if callable(func):
                fitted_iso, params = fit_using(signal, isosbestic)
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
    
    def fit_photobleaching_curve(
            self,
            signal: np.ndarray, 
            window_dur: float =  5,
            ) -> tuple[np.ndarray, float, list[float]]:
        """Fit a curve with a negative bi-exponential photobleaching model.
        Uses a soft_l1 least squares to fit to the sliding-window median 
        downsampled signal.

        Args:
            signal (np.ndarray): Signal array to fit the photobleaching curve to.
            window_dur (float, optional): Length of the window in seconds used
                for sliding-window median downsampling. Defaults to ``5``.

        Returns:
            tuple[np.ndarray, list[float]]: Fitted curve and fitted parameter
            values.
        """
        window_len = int(window_dur * self.frequency)
        fitted_curve, params = fit_photobleaching(signal, self.time, window=window_len)
        r2_val = r2_score(signal, fitted_curve)

        if np.isnan(fitted_curve).any():
            raise ValueError(f'Fitted photobleaching curve produced NaNs.')
        
        return fitted_curve, r2_val, params
    
    # --- hidden signal preprocess helpers ---
    def _apply_correction_method(
            self,
            correction_method: Literal['dF/F', 'dF', 'dB/B', 'dB', 'none'] | Callable,
            signal: np.ndarray,
            fitted_ref: np.ndarray,
            ) -> np.ndarray:
        
        match correction_method:
            case func if callable(func):
                out: np.ndarray = correction_method(signal, fitted_ref)
            case 'dF' | 'dB':
                out = (signal - fitted_ref)
            case 'dF/F' | 'dB/B':
                out = (signal - fitted_ref) / np.maximum(fitted_ref, np.finfo(np.float32).eps)
            case 'none':
                out = signal
            case _:
                raise ValueError(f'{correction_method} isosbestic correction method not recognized!')
        return out
    
    def _apply_signal_normalization(
            self,
            signal_normalization: Literal['zscore', 'nullZ', 'none'] | Callable,
            signal : np.ndarray,
            ) -> np.ndarray:
        
        match signal_normalization:
            case func if callable(func):
                out: np.ndarray = signal_normalization(signal)
            case 'zscore':
                out = (signal - np.mean(signal)) / np.std(signal)
            case 'nullZ':
                out = signal / np.std(signal)
            case 'none':
                out = signal
            case _:
                raise ValueError(f'{signal_normalization} whole signal normalization method not recognized!')
        return out
    
    def _detect_artifacts(
            self,
            detector: ArtifactDetector,
            signal: np.ndarray,
            reference: np.ndarray,
            time: np.ndarray,
            ) -> ArtifactResult:
        
        artifacts = detector.detect(
            signal=signal,
            reference=reference,
            time=time,
        )
        return artifacts
    
    def _correct_artifacts(
            self,
            corrector: ArtifactCorrector,
            artifacts: ArtifactResult,
            signal: np.ndarray,
            time: np.ndarray,
            ) -> np.ndarray:
        
        corrected = corrector.correct(
            signal=signal,
            time=time,
            artifacts=artifacts,
        )
        return corrected
    
    # --- trial extraction helpers ---
    def _apply_trial_normalization(
            self, 
            trial_normalization: Literal['zscore', 'zero', 'mad', 'amp', 'none'] | Callable, 
            trial_windows: WindowResult, 
            baseline_windows: WindowResult | None,
            ) -> np.ndarray:
        # unpack
        raw_trial_signals = trial_windows.signals
        baseline_signals = None if baseline_windows is None else baseline_windows.signals

        # execute
        match trial_normalization:
            case func if callable(func):
                trial_signals = trial_normalization(raw_trial_signals, baseline_signals)
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
        return trial_signals

    def _find_interval_bounds(self, series: np.ndarray, centers: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
        """Compute index bounds for windows around specified centers.

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
    
    def _find_timestamp_in_intervals(
            self, 
            timestamps: np.ndarray, 
            time_intervals: np.ndarray, 
            logic: Literal['first', 'last', 'mean'] = 'first',
            ) -> np.ndarray:
        """Find timestamps within each time interval using customizable logic.

        Args:
            timestamps (np.ndarray): Sorted 1D array of event timestamps.
            time_intervals (np.ndarray): Array of shape (n_trials, 2) with [start, end] bounds.
            logic (Literal['first', 'last', 'mean'], optional): Logic for
                choosing event timestamps if multiple are present. Defaults to
                ``'first'``.

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
                out[in_interval] = timestamps[hi_idx[in_interval] - 1]
            case 'mean':
                counts = hi_idx - lo_idx
                in_interval = counts > 0

                csum = np.concatenate(([0.0], np.cumsum(timestamps)))
                sums = csum[hi_idx] - csum[lo_idx]

                out[in_interval] = sums[in_interval] / counts[in_interval]
            case _:
                raise ValueError(f'Multiple event selection logic, {logic}, is not recognized')
        return out
    
    def _resolve_alignment(
            self,
            align_to: str | Sequence[str] | float | Sequence[float],
            ) -> tuple[str, np.ndarray, pd.DataFrame]:
        # set default manual alignment label   
        ALIGNMENT_LABEL = 'ALIGNMENTS' 

        # single event
        if isinstance(align_to, str):
            if align_to not in self.events: 
                raise KeyError(
                    f"align_to '{align_to}' not found in events. "
                    f"Available events: {list(self.events)}"
                )
            
            align_events = self.events[align_to].copy()
            align_obs = pd.DataFrame(index=np.arange(align_events.size))
            align_label = align_to

        # single timestamp
        elif isinstance(align_to, Real) and not isinstance(align_to, bool):
            align_events = np.asarray([align_to])
            align_obs = pd.DataFrame(index=np.arange(align_events.size))
            align_label = ALIGNMENT_LABEL

        # multiple events
        elif isinstance(align_to, Sequence) and all(isinstance(v, str) for v in align_to):
            if len(align_to) == 0:
                raise ValueError("align_to must contain at least one event label.")

            missing = [label for label in align_to if label not in self.events]
            if missing:
                raise KeyError(
                    f"align_to labels not found in events: {missing}. "
                    f"Available events: {list(self.events)}"
                )

            times = np.concatenate([self.events[label] for label in align_to])
            sources = np.concatenate([
                np.repeat(label, len(self.events[label]))
                for label in align_to
            ])

            order = np.argsort(times, kind="stable")

            align_events = times[order]
            align_obs = pd.DataFrame(
                {"align_event": sources[order]},
                index=np.arange(align_events.size),
            )
            align_label = ALIGNMENT_LABEL

        # multiple timestamps
        elif isinstance(align_to, Sequence) and all(isinstance(v, (float, int)) for v in align_to):
            if len(align_to) == 0:
                raise ValueError("align_to must contain at least one timestamp.")

            align_events = np.asarray(align_to, dtype=float)
            align_obs = pd.DataFrame(index=np.arange(align_events.size))
            align_label = ALIGNMENT_LABEL

        else:
            raise ValueError(f'Invalid input type for align_to: {type(align_to)}.')

        # validate
        if align_events.size == 0:
            raise ValueError(f"No align_to ({align_to}) events found.")

        if isinstance(align_events, np.ndarray) and (align_events.ndim != 1):
            raise ValueError(f'align_events is not 1D-array, type: {type(align_events)}, ndim: {align_events.ndim}.')
    
        return align_label, align_events, align_obs
    
    def _coerce_centering(
            self, 
            center_on: str | Sequence[str] | None,
            ) -> list[str]:
        if center_on is None:
            center_out = []

        elif isinstance(center_on, str):
            if center_on not in self.events:
                raise KeyError(
                f"center_on label ({center_on}) not found in events. "
                f"Available events: {list(self.events)}"
            ) 
            center_out = [center_on]

        elif isinstance(center_on, Sequence):
            missing = [label for label in center_on if label not in self.events]
            if missing:
                raise KeyError(
                f"center_on labels ({missing}) not found in events. "
                f"Available events: {list(self.events)}"
            )
            center_out = list(center_on)
        return center_out
    
    def _annotate_intervals(
            self, 
            align_label: str, 
            series: np.ndarray, 
            centers: np.ndarray, 
            events: dict[str, np.ndarray], 
            tolorences: dict[str, np.ndarray],
            logic: Literal['first', 'last', 'mean'] = 'first',
            ) -> dict[str, np.ndarray]:
        """Annotate intervals around centers with event timestamps.

        Args:
            align_label (str): Label of the primary alignment event.
            series (np.ndarray): Monotonic time-like series.
            centers (np.ndarray): Center times for each trial.
            events (dict[str, np.ndarray]): Mapping of event labels to timestamps.
            tolorences (dict[str, np.ndarray]): Mapping of labels to time tolerances.
            logic (Literal['first', 'last', 'mean'], optional): Logic for
                choosing event timestamps if multiple are present. Defaults to
                ``'first'``.

        Returns:
            dict[str, np.ndarray]: Mapping from labels to aligned event times per trial.
        """
        out = {align_label: centers.copy()}
        tmin, tmax = series[0], series[-1]

        for label, bounds in tolorences.items():
            low, high = map(float, bounds)
            if low > high:
                raise ValueError(f"Invalid bounds for {label}: {bounds}")

            timestamps = events.get(label)
            if timestamps is None:
                out[label] = np.full(centers.shape, np.nan, dtype=float)
                continue

            timestamps = np.asarray(timestamps, dtype=float)
            if timestamps.ndim != 1:
                raise ValueError(f"Event timestamps for {label} must be 1D")

            time_intervals = np.column_stack((centers + low, centers + high))
            time_intervals[:, 0] = np.maximum(time_intervals[:, 0], tmin)
            time_intervals[:, 1] = np.minimum(time_intervals[:, 1], tmax)

            invalid = time_intervals[:, 0] > time_intervals[:, 1]
            selected = np.full(centers.shape, np.nan, dtype=float)

            if timestamps.size > 0 and np.any(~invalid):
                selected[~invalid] = self._find_timestamp_in_intervals(
                    timestamps=timestamps,
                    time_intervals=time_intervals[~invalid],
                    logic=logic,
                )

            out[label] = selected
        return out
    
    def _create_windows(
            self,
            signal: np.ndarray,
            time: np.ndarray,
            events: dict[str, np.ndarray],
            centers: np.ndarray,
            bounds: tuple[float, float],
            strategy: Literal['nearest', 'interp'] = 'nearest',
            ) -> WindowResult:
        # execute windowing
        match strategy:
            case 'nearest':
                return create_windows_nearest_1D(signal, time, events, centers, bounds, self.frequency)
            case 'interp':
                return create_windows_interp_1D(signal, time, events, centers, bounds, self.frequency)
            case _:
                raise ValueError(f'Window alignment strategy {strategy} not recognized.')
            
    def _handle_invalid_windows(
            self,
            trial_windows: WindowResult,
            baseline_windows: WindowResult | None,
            policy: Literal['drop', 'error'] = 'drop'
            ) -> np.ndarray:
        # find valid trials
        if baseline_windows is None:
            invalid_mask = trial_windows.invalid_mask
        else:
            invalid_mask = trial_windows.invalid_mask | baseline_windows.invalid_mask
        valid_mask = ~invalid_mask

        # apply policy
        if invalid_mask.any():
            invalid_idxs = np.flatnonzero(invalid_mask).tolist()
            self.metadata['invalid_windows'] = invalid_idxs

            match policy:
                case 'drop':
                    trial_windows.apply_mask(valid_mask)
                    if baseline_windows is not None:
                        baseline_windows.apply_mask(valid_mask)

                case 'error':
                    raise ValueError(
                        f'Invalid trial windows that extend outside signal range at trial indicies {invalid_idxs}'
                    )
                case _:
                    raise ValueError(f'Invalid window policy {policy} not recognized.')
        else:
            self.metadata['invalid_windows'] = None
        
        # confirm no invalid windows left
        assert not trial_windows.invalid_mask.any()

        # confirm some trials remain
        if trial_windows.signals.size == 0:
            raise ValueError(f'No trials remain after dropping invalid windows.')
        
        return valid_mask

    def _find_window_centers(
            self, 
            center_on: list[str], 
            align_events: np.ndarray, 
            events: dict[str, np.ndarray], 
            check_overlap: bool = True
            ) -> np.ndarray:
        """Determine window centers based on ``center_on`` events.

        Falls back to ``align_on`` when ``center_on`` is missing.

        Args:
            center_on (list[str]): Event labels used as preferred centers.
            align_events (np.ndarray): Timestamps of alignment timestamps.
            events (dict[str, np.ndarray]): Mapping from labels to event times per trial.
            check_overlap (bool, optional): Whether to throw an error when
                multiple ``center_on`` events are found in the same trial.
                Defaults to ``True``.

        Returns:
            np.ndarray: Center times per trial.
        """
        # center_on events should be non-overlaping
        # if no center_on events present, center on align_on
        centers = align_events.copy()
        present_count = np.zeros_like(centers, dtype=int)

        if len(center_on) == 0:
            return centers
        
        for label in center_on:
            arr = events[label]
            event_not_nan = ~np.isnan(arr)
            centers[event_not_nan] = arr[event_not_nan]
            present_count += event_not_nan.astype(int)

        overlap = present_count > 1

        if check_overlap and overlap.any():
            at_idxs = np.where(overlap)
            culprits = {k : events[k][at_idxs] for k in center_on}
            culprits_in_og_events = {k : self.events[k][np.searchsorted(self.events[k], culprits[k])] for k in center_on}
            raise ValueError(f'Center_on events over lap in trials {np.where(overlap)}, with culprits: {culprits_in_og_events}')
        return centers
    
    # --- poor signal checks ---
    def median_centered_abs_max_check(
            self, 
            trial_signal: np.ndarray, 
            threshold: float = 0.075
            ):
        """Check whether a trial signal should be flagged as poor quality

        Tests a threshold for minimum MAD (robust Z-score).
        Threshold needs to be tuned for specific experiments.
        Generally not recommended.

        Args:
            trial_signal (np.ndarray): Trial-by-time signal array.
            threshold (float, optional): Mean absolute-max threshold applied
                after median centering. Defaults to ``0.075``.

        Returns:
            is_poor_signal (bool): ``True`` if the signal is classified as poor quality.
        """
        median_centered = trial_signal - np.median(trial_signal, axis=1, keepdims=True)
        abs_max = np.abs(median_centered).max(axis=1)
        is_poor_signal = np.mean(abs_max) < threshold
        return is_poor_signal

    # --- graphing ---
    def dashboard(self, save: str | None = None, downsample: int = 20) -> None:
        """Plot a quick dashboard for the experiment.

        Plots the raw, fitted, and, if ``.preprocess_signal()`` has been run, 
        processed signal, isosbestic trace, and the fitted photobleaching curve 
        if available.

        Args:
            save (str | None, optional): Path to save the figure. If ``None``,
                the figure is not saved. Defaults to ``None``.
            downsample (int | optional): Downsample factor for signals before
                plotting. Defaults to 20.

        Returns:
            None
        """
        # determine is self.process_singal has been run
        if self.fitted_ref is not None:
            fig = self.dashboard_full()
        else:
            fig = self.dashboard_raw()

        if save is not None:
            plt.savefig(save, bbox_inches='tight')

    def dashboard_raw(self, downsample: int = 20):
        # down sample series
        x = downsample_1d(self.time, downsample)
        raw_sig = downsample_1d(self.raw_signal, downsample)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
        fig.tight_layout()
        
        # raw signals
        ax: matplotlib.axes.Axes
        ax.plot(x, raw_sig, label='Raw Signal', c='#1f77b4')

        # plot isosbestic if it is avaliable
        if self.raw_isosbestic is not None:
            raw_iso = downsample_1d(self.raw_isosbestic, downsample)
            ax.plot(x, raw_iso, label='Raw Iso.', c="#4B4B4B", alpha=0.9)
        
        ax.legend()

        # annotate
        ax.set_title(
        f"Dashboard for {getattr(self, 'id', 'Unnamed')}"
        )
        ax.set_ylabel('Signal amplitude (a.u.)')
        ax.set_xlabel('Time (s)')

        return fig
    
    def dashboard_full(self, downsample: int = 20):
        if self.signal is None:
            raise ValueError(f"Process signal not avaliable, please run self.preprocess_signal() first.")


        x = downsample_1d(self.time, downsample)
        raw_sig = downsample_1d(self.raw_signal, downsample)
        fit_params = getattr(self, 'fitted_params', None)
        final_sig = downsample_1d(self.signal, downsample)

        fig, (ax1, ax2) = plt.subplots(
            ncols=1, nrows=2, 
            sharex=True, figsize=(6, 6), dpi=140,
            gridspec_kw={'height_ratios': [3, 1]})
        fig.tight_layout()
        
        # raw and fitted signals
        ax1: matplotlib.axes.Axes
        ax1.plot(x, raw_sig, label='Raw Signal', c='#1f77b4')

        if self.channel_mode == 'dual':
            raw_ref = downsample_1d(self.raw_isosbestic, downsample)
            fit_ref = downsample_1d(self.fitted_ref, downsample)
            ax1.plot(x, raw_ref, label='Raw Iso.', c="#4B4B4B", alpha=0.9)
            ax1.plot(x, fit_ref, label='Fitted Iso.', c='#ff7f0e', alpha=0.9)

        elif self.channel_mode == 'single':
            fit_ref = downsample_1d(self.fitted_ref, downsample)
            ax1.plot(x, fit_ref, label='Fitted Bleaching', c='#ff7f0e', alpha=0.9)

        if fit_params is not None:
            ax1.plot(x, neg_bi_exponential_5(x, *fit_params), label='Fitted Curve', c="#920000")
        ax1.legend()

        # processed signal
        ax2: matplotlib.axes.Axes
        y_pad_factor = 2.5
        middle_third = np.array_split(final_sig, 3)[1]
        y_high = np.max(middle_third)
        y_low = np.min(middle_third)
        ax2.plot(x, final_sig, label='Processed Signal', c='#1f77b4')
        ax2.set_ylim(bottom=y_low*y_pad_factor, top=y_high*y_pad_factor)
        ax2.legend()

        # annotate
        ax1.set_title(
        f"Dashboard for {getattr(self, 'id', 'Unnamed')}"
        )
        ax1.set_ylabel('Signal amplitude (a.u.)')
        ax2.set_ylabel(f"{self.metadata.get('correction_method', 'NOT FOUND')}")
        ax2.set_xlabel('Time (s)')

        return fig
