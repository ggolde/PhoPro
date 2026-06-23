"""High-level simulator for synthetic fiber photometry recordings."""

from __future__ import annotations

from typing import Callable, Any, Self, Literal, Sequence

import numpy as np
import pandas as pd
from plotnine import ggplot

from ..core.PhotometryExperiment import PhotometryExperiment
from ..core.PhotometeryData import PhotometryData
from ..utils import operations, graphing, window

from .kernels import(
    gamma_kernel,
    gaussian_kernel,
    exp_decay_kernel,
    alpha_kernel,
    diff_of_exp_kernel,
    sum_of_exp_kernel,
    smooth_trapezoid_kernel,
    domed_trapezoid_kernel,
)

from .layers import (
    TimeBase, 
    PhotobleachingLayer,
    EventLayer, EventSpec,
    NoiseShotLayer, NoiseGaussianLayer,
    NeuralDynamicNoiseLayer,
    MovementAttenuationLayer,
    ArtifactSpikeLayer, ArtifactJumpLayer,
)

class SimulatedPhotometry:
    """Build and export simulated dual-channel photometry traces."""

    ############################
    #region --- CONSTRUCTOR ---
    ############################
    def __init__(
            self,
            # layer objects
            timebase: TimeBase,
            bleaching_exp: PhotobleachingLayer,
            bleaching_iso: PhotobleachingLayer,
            event_layer: EventLayer,
            noise_shot_exp: NoiseShotLayer,
            noise_shot_iso: NoiseShotLayer,
            noise_gaussian_exp: NoiseGaussianLayer,
            noise_gaussian_iso: NoiseGaussianLayer,
            dynamic_noise: NeuralDynamicNoiseLayer,
            movement: MovementAttenuationLayer,
            artifact_spike: ArtifactSpikeLayer,
            artifact_jump: ArtifactJumpLayer,
            # other
            seed: int | None = None,
            iso_event_leakage: float | None = None,
            ) -> None:
        """Initialize a simulated photometry recording from layer objects.

        Parameters
        ----------
        timebase : TimeBase
            Timebase used to render all traces.
        bleaching_exp : PhotobleachingLayer
            Photobleaching layer for the experimental channel.
        bleaching_iso : PhotobleachingLayer
            Photobleaching layer for the isosbestic channel.
        event_layer : EventLayer
            Event layer rendered into the experimental channel.
        noise_shot_exp : NoiseShotLayer
            Shot-noise layer in the experimental channel.
        noise_shot_iso : NoiseShotLayer
            Shot-noise layer in the isosbestic channel.
        noise_gaussian_exp : NoiseGaussianLayer
            Additive Gaussian noise layer in experimental channel.
        noise_gaussian_iso : NoiseGaussianLayer
            Additive Gaussian noise layer in isosbestic channel.
        movement : MovementAttenuationLayer
            Multiplicative movement attenuation layer.
        artifact_spike : ArtifactSpikeLayer
            Multiplicative spike artifact layer.
        artifact_jump : ArtifactJumpLayer
            Multiplicative jump artifact layer.
        seed : int or None, default=None
            Random seed used when rendering stochastic layers.
        iso_event_leakage : float or None, default=None
            Fraction of the event trace mixed into the isosbestic channel.
            ``None`` is treated as ``0.0``.
        """
        # layers
        self.timebase = timebase

        self.bleaching_exp = bleaching_exp
        self.bleaching_iso = bleaching_iso
        
        self.event_layer = event_layer
        self.dynamic_noise = dynamic_noise

        self.noise_shot_exp = noise_shot_exp
        self.noise_shot_iso = noise_shot_iso

        self.noise_gaussian_exp = noise_gaussian_exp
        self.noise_gaussian_iso = noise_gaussian_iso
        
        self.movement = movement
        self.artifact_spike = artifact_spike
        self.artifact_jump = artifact_jump

        # other
        self.iso_event_leakage = 0.0 if iso_event_leakage is None else iso_event_leakage
        self.seed = seed

        # build layers
        self.build_traces()

    @classmethod
    def from_parameters(
            cls,

            length_sec: float = 1000,
            frequency: float = 100,

            event_label: str = 'trial_cue',
            n_events: int | None = 10,
            event_kernel: Callable = gamma_kernel,
            event_amplitude: float = 0.02,
            event_kernel_params: dict[str, Any] = {'shape_k': 3, 'tau_sec': 1.0},
            event_buffer_sec: float | None = 10.0,

            bleaching_params_exp: dict[str, float] = {'alpha1':50, 'alpha2':20, 'tau1':300, 'tau2':10000, 'B_floor':10},
            bleaching_params_iso: dict[str, float] | None = None,
            iso_bleach_scale: float | None = 0.8,
            iso_bleach_offset: float | None = None,

            photons_per_unit_exp: int | None = 1e5,
            photons_per_unit_iso: int | None = None,

            gaussian_noise_scale_exp: float | None = 0.2,
            gaussian_noise_scale_iso: float | None = None,

            dynamic_noise_amplitude: float | None = None,
            dynamic_noise_center: float = 0.0,
            dynamic_noise_frequency: float = 1.0,

            movement_attenuation: float | None = 0.5,
            attenuation_cutoff_hz: float = 0.1,

            n_spike_artifacts: int | None = None,
            spike_amplitude_range: tuple[float, float] = (-0.5, -0.1),
            spike_decay_range: tuple[float, float] = (0.01, 0.05),

            n_jump_artifacts: int | None = None,
            jump_amplitude_range: tuple[float, float] = (-0.5, 0.5),
            jump_duration_range: tuple[float, float] = (10, 100),

            seed: int | None = 42,
            iso_event_leakage: float | None = None,
            max_event_duration_sec: float = 1000.0,
            ) -> Self:
        """Construct a simulator from parameter dictionaries and scalars.

        Parameters
        ----------
        length_sec : float, default=1000
            Simulated recording duration, in seconds.
        frequency : float, default=100
            Sampling frequency in Hz.
        event_label : str, default='trial_cue'
            Label for the initial evenly spaced event.
        n_events : int or None, default=10
            Number of initial events to create.
        event_kernel : Callable, default=gamma_kernel
            Kernel function used for the initial event.
        event_amplitude : float, default=0.02
            Amplitude passed to ``event_kernel``.
        event_kernel_params : dict[str, Any], default={'shape_k': 3, 'tau_sec': 1.0}
            Additional keyword arguments passed to ``event_kernel``.
        event_buffer_sec : float or None, default=10.0
            Left and right buffer used when placing evenly spaced events.
        bleaching_params_exp : dict[str, float]
            Parameters used to construct the experimental photobleaching layer.
        bleaching_params_iso : dict[str, float] or None, default=None
            Parameters used to construct a separate isosbestic photobleaching
            layer.
        iso_bleach_scale : float or None, default=0.8
            Scale applied to experimental bleaching parameters when building
            the isosbestic layer.
        iso_bleach_offset : float or None, default=None
            Offset applied to experimental bleaching parameters when building
            the isosbestic layer.
        photons_per_unit_exp : int or None, default=1e5
            Photon-count scale for shot noise in the experimental channel.
        photons_per_unit_iso : int or None, default=None
            Photon-count scale for shot noise in the isosbestic channel.
            If ``None``, it is set equal to ``photons_per_unit_exp``.
        gaussian_noise_scale_exp : float or None, default=0.2
            Standard deviation of additive Gaussian noise in the experimental channel.
        gaussion_noise_scale_iso : float or None, default=None
            Standard deviation of additive Gaussian noise in the isosbestic channel.
            If ``None``, it is set equal to ``gaussian_noise_scale_exp``.
        dynamic_noise_amplitude : float or None, default=None
            Standard deviation of random neural dynamic noise.
        dynamic_noise_center : float, default=0.0
            Mean of random neural dynamic noise.
        dynamic_noise_frequency : float, default 1.0
            Frequency of random neural dynamic noise. 
            Should be > experiment frequency.
        movement_attenuation : float, default=0.5
            Attenuation strength for movement artifacts.
        attenuation_cutoff_hz : float, default=0.1
            Low-pass cutoff used to generate movement artifacts.
        n_spike_artifacts : int or None, default=None
            Number of spike artifacts. ``None`` is treated as ``0``.
        spike_amplitude_range : tuple[float, float], default=(-0.5, -0.1)
            Range sampled for spike artifact amplitudes.
        spike_decay_range : tuple[float, float], default=(0.01, 0.05)
            Range sampled for spike artifact decay constants.
        n_jump_artifacts : int or None, default=None
            Number of jump artifacts. ``None`` is treated as ``0``.
        jump_amplitude_range : tuple[float, float], default=(-0.5, 0.5)
            Range sampled for jump artifact amplitudes.
        jump_duration_range : tuple[float, float], default=(10, 100)
            Range sampled for jump artifact durations, in seconds.
        seed : int or None, default=42
            Random seed used when rendering stochastic layers.
        iso_event_leakage : float or None, default=None
            Fraction of the event trace mixed into the isosbestic channel.
        max_event_duration_sec : float, default=1000.0
            Max duration of event peaks before values are cutoff in seconds. 
            A cutoff is required to generate finite kernels. 

        Returns
        -------
        SimulatedPhotometry
            Fully rendered simulator object.

        Raises
        ------
        ValueError
            If both separate isosbestic bleaching parameters and isosbestic
            scaling/offset parameters are supplied, or if neither is supplied.
        """
        # time
        timebase = TimeBase(time_sec=length_sec, frequency=frequency)

        # events
        onsets = timebase.evenly_spaced_timestamps(n_events, event_buffer_sec, event_buffer_sec)
        event_layer = EventLayer(
            specs={event_label : EventSpec(onsets, event_amplitude, event_kernel, event_kernel_params)},
            max_duration_sec=max_event_duration_sec,
        )

        # bleaching
        bleaching_exp = PhotobleachingLayer(**bleaching_params_exp)

        is_seperate_curves = (bleaching_params_iso is not None)
        is_scaled_iso = (iso_bleach_scale is not None) or (iso_bleach_offset is not None)
        if is_seperate_curves and is_scaled_iso:
            raise ValueError('Only bleaching_params_iso or (iso_bleach_scale | iso_bleach_offset) should be specified')
        elif is_seperate_curves:
            bleaching_iso = PhotobleachingLayer(**bleaching_params_iso)
        elif is_scaled_iso:
            bleaching_iso = PhotobleachingLayer(**bleaching_params_exp, scale=iso_bleach_scale, offset=iso_bleach_offset)
        else:
            raise ValueError('One of bleaching_params_iso or (iso_bleach_scale | iso_bleach_offset) has to be be specified')

        # shot noise
        photons_per_unit_iso = photons_per_unit_exp if photons_per_unit_iso is None else photons_per_unit_iso
        noise_shot_exp = NoiseShotLayer(photons_per_unit_exp)
        noise_shot_iso = NoiseShotLayer(photons_per_unit_iso)

        # gaussian noise
        gaussian_noise_scale_iso = gaussian_noise_scale_exp if gaussian_noise_scale_iso is None else gaussian_noise_scale_iso
        noise_gaussian_exp = NoiseGaussianLayer(gaussian_noise_scale_exp)
        noise_gaussian_iso = NoiseGaussianLayer(gaussian_noise_scale_iso)

        dynamic_noise = NeuralDynamicNoiseLayer(
            dynamic_noise_amplitude,
            dynamic_noise_center,
            dynamic_noise_frequency,
        )

        # movement attenuation
        movement = MovementAttenuationLayer(movement_attenuation, attenuation_cutoff_hz)

        # artifacts
        n_spike_artifacts = 0 if n_spike_artifacts is None else n_spike_artifacts
        n_jump_artifacts = 0 if n_jump_artifacts is None else n_jump_artifacts
        artifact_spike = ArtifactSpikeLayer(n_spike_artifacts, spike_amplitude_range, spike_decay_range)
        artifact_jump = ArtifactJumpLayer(n_jump_artifacts, jump_amplitude_range, jump_duration_range)

        # extras
        seed = seed
        iso_event_leakage = 0.0 if iso_event_leakage is None else iso_event_leakage

        # build instance
        return cls(
            timebase=timebase,
            bleaching_exp=bleaching_exp,
            bleaching_iso=bleaching_iso,
            event_layer=event_layer,
            noise_shot_exp=noise_shot_exp,
            noise_shot_iso=noise_shot_iso,
            noise_gaussian_exp=noise_gaussian_exp,
            noise_gaussian_iso=noise_gaussian_iso,
            dynamic_noise=dynamic_noise,
            movement=movement,
            artifact_spike=artifact_spike,
            artifact_jump=artifact_jump,
            seed=seed,
            iso_event_leakage=iso_event_leakage
        )
    
    #endregion

    ############################
    #region --- BUILD TRACES ---
    ############################
    
    def build_traces(self, seed: int | None = None) -> None:
        """Render all component traces and final fluorescence channels.

        Parameters
        ----------
        seed : int or None, default=None
            Optional seed overriding ``self.seed`` for this render.
            Recalls with the same seed will produce identical RNG-based 
            layers.
        """
        # make rng locally for repeatability
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed=seed)

        # build time
        self.time = self.timebase.render()
        self.freq = self.timebase.frequency

        # bleaching curves
        self.B_exp = self.bleaching_exp.render(self.time)
        self.B_iso = self.bleaching_iso.render(self.time)

        # events
        self.E = self.event_layer.render(self.time, self.freq)
        self.E_iso = self.iso_event_leakage * self.E

        # random neural dynamics
        self.D = self.dynamic_noise.render(self.time, self.freq, rng)
        self.D_iso = self.iso_event_leakage * self.D

        # clean traces
        self.neural_trace_exp = self.E + self.D
        self.neural_trace_iso = self.E_iso + self.D_iso

        self.clean_exp = (self.B_exp - self.bleaching_exp.B_floor) * self.neural_trace_exp + self.B_exp
        self.clean_iso = (self.B_iso - self.bleaching_iso.B_floor) * self.neural_trace_iso + self.B_iso

        # artifacts
        self.M = self.movement.render(self.time, self.freq, rng)
        self.AS = self.artifact_spike.render(self.time, self.freq, rng)
        self.AJ = self.artifact_jump.render(self.time, self.freq, rng)
        self.A = self.M * self.AS * self.AJ

        # noiseless trace
        noiseless_exp = self.clean_exp * self.A
        noiseless_iso = self.clean_iso * self.A

        # noise
        self.N_shot_exp = self.noise_shot_exp.render(noiseless_exp, rng)
        self.N_gauss_exp = self.noise_gaussian_exp.render(self.time, rng)
        self.N_exp = self.N_shot_exp + self.N_gauss_exp

        self.N_shot_iso = self.noise_shot_iso.render(noiseless_iso, rng)
        self.N_gauss_iso = self.noise_gaussian_iso.render(self.time, rng)
        self.N_iso = self.N_shot_iso + self.N_gauss_iso

        # composite final signal
        self.F_exp = noiseless_exp + self.N_exp
        self.F_iso = noiseless_iso + self.N_iso

    #endregion

    #######################
    #region --- UTILITY ---
    #######################
    def get_event_spec(self, label: str) -> EventSpec:
        """Get an event specification by label.

        Parameters
        ----------
        label : str
            Event label.

        Returns
        -------
        EventSpec
            Stored event specification.
        """
        return self.event_layer.specs[label]


    #endregion

    ##############################
    #region --- HANDLE EVENTS ---
    ##############################
    def add_event(
            self,
            label: str,
            onsets: np.ndarray,
            amplitude: float,
            kernel_func: Callable,
            kernel_params: dict[str, Any],
            ) -> None:
        """Add an event type to the simulator.

        Parameters
        ----------
        label : str
            Event label.
        onsets : np.ndarray
            Event onset times, in seconds.
        amplitude : float
            Amplitude passed to ``kernel_func``.
        kernel_func : Callable
            Kernel function used to render the event.
        kernel_params : dict[str, Any]
            Additional keyword arguments passed to ``kernel_func``.
        """
        # add event
        self.event_layer.add_event(
            label=label,
            onsets=onsets,
            amplitude=amplitude,
            kernel_func=kernel_func,
            kernel_params=kernel_params,
        )
        # recalculate layers
        self.build_traces()

    def add_event_relative_to(
            self,
            relative_to: str,
            time_range: tuple[float, float],
            labels: list[str],
            overall_prob: float | None = None,
            choice_probs: list[float] | None = None,
            amplitudes: list[float] | None = None,
            kernel_funcs: list[Callable] | None = None,
            kernel_params: list[dict[str, Any]] | None = None,
            seed: int | None = None,
            ) -> None:
        """Add events sampled relative to an existing event label.

        Parameters
        ----------
        relative_to : str
            Existing event label used as the anchor.
        time_range : tuple[float, float]
            Minimum and maximum offset, in seconds, sampled relative to each
            anchor onset.
        labels : list[str]
            New event labels to sample from.
        overall_prob : float or None, default=None
            Probability that any relative event occurs for each anchor.
            ``None`` is treated as ``1.0``.
        choice_probs : list[float] or None, default=None
            Sampling probabilities for ``labels``.
        amplitudes : list[float] or None, default=None
            Amplitudes for each label. ``None`` uses ``1.0`` for every label.
        kernel_funcs : list[Callable] or None, default=None
            Kernel functions for each label. ``None`` uses ``self.kernel_gamma``.
        kernel_params : list[dict[str, Any]] or None, default=None
            Kernel parameter dictionaries for each label. ``None`` uses empty
            dictionaries.
        seed : int or None, default=None
            Optional seed overriding ``self.seed`` for relative-event sampling.
        """
        # set up rng
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed=seed)

        # handle None inputs
        overall_prob = 1.0 if overall_prob is None else overall_prob
        amplitudes = [1.0 for s in labels] if amplitudes is None else amplitudes
        kernel_funcs = [self.kernel_gamma for s in labels] if kernel_funcs is None else kernel_funcs
        kernel_params = [{} for s in labels] if kernel_params is None else kernel_params

        # get relative_to event
        anchor_event = self.event_layer.specs[relative_to]
        anchor_onsets = anchor_event.onsets

        # generate new timestamps
        occurs = rng.random(anchor_onsets.size) < overall_prob
        new_onsets = anchor_onsets[occurs] + rng.uniform(time_range[0], time_range[1], size=occurs.sum())
        new_labels = rng.choice(labels, size=occurs.sum(), p=choice_probs)

        # clip events to be in bounds
        in_bounds = (new_onsets >= self.time[0]) & (new_onsets <= self.time[-1])
        new_onsets = new_onsets[in_bounds]
        new_labels = new_labels[in_bounds]

        # add evnets to event layer
        for i in range(len(labels)):
            self.add_event(
                label=labels[i],
                onsets = new_onsets[np.nonzero(new_labels == labels[i])],
                amplitude = amplitudes[i],
                kernel_func = kernel_funcs[i],
                kernel_params = kernel_params[i]
            )

        # recalculate layers
        self.build_traces()

    #endregion
    
    ###########################
    #region --- KERNEL APIS ---
    ###########################
    @staticmethod
    def kernel_gamma(
            time: np.ndarray,
            amplitude: float = 1.0,
            shape_k: float = 3,
            tau_sec: float = 1.0,
            ) -> np.ndarray:
        """Evaluate the simulator gamma kernel wrapper.

        Parameters
        ----------
        time : np.ndarray
            Time points, in seconds.
        amplitude : float, default=1.0
            Peak amplitude.
        shape_k : float, default=3
            Gamma shape parameter.
        tau_sec : float, default=1.0
            Decay time constant, in seconds.

        Returns
        -------
        np.ndarray
            Kernel values evaluated at ``time``.
        """
        return gamma_kernel(time, amplitude, shape_k, tau_sec)

    @staticmethod
    def kernel_exp_deacy(
            time: np.ndarray,
            amplitude: float = 1.0,
            tau_sec: float = 1.0,
            ) -> np.ndarray:
        """Evaluate the simulator exponential-decay kernel wrapper.

        Parameters
        ----------
        time : np.ndarray
            Time points, in seconds.
        amplitude : float, default=1.0
            Initial amplitude.
        tau_sec : float, default=1.0
            Decay time constant, in seconds.

        Returns
        -------
        np.ndarray
            Kernel values evaluated at ``time``.
        """
        return exp_decay_kernel(time, amplitude, tau_sec)

    @staticmethod
    def kernel_alpha(
            time: np.ndarray,
            amplitude: float = 1.0,
            tau_sec: float = 1.0,
            ) -> np.ndarray:
        """Evaluate the simulator alpha kernel wrapper.

        Parameters
        ----------
        time : np.ndarray
            Time points, in seconds.
        amplitude : float, default=1.0
            Peak amplitude.
        tau_sec : float, default=1.0
            Time constant, in seconds.

        Returns
        -------
        np.ndarray
            Kernel values evaluated at ``time``.
        """
        return alpha_kernel(time, amplitude, tau_sec)

    @staticmethod
    def kernel_diff_of_exp(
            time: np.ndarray,
            amplitude: float = 1.0,
            tau_rise_sec: float = 1.0,
            tau_decay_sec: float = 1.0,
            ) -> np.ndarray:
        """Evaluate the simulator difference-of-exponentials kernel wrapper.

        Parameters
        ----------
        time : np.ndarray
            Time points, in seconds.
        amplitude : float, default=1.0
            Peak amplitude.
        tau_rise_sec : float, default=1.0
            Rise time constant, in seconds.
        tau_decay_sec : float, default=1.0
            Decay time constant, in seconds.

        Returns
        -------
        np.ndarray
            Kernel values evaluated at ``time``.
        """
        return diff_of_exp_kernel(time, amplitude, tau_rise_sec, tau_decay_sec)

    @staticmethod
    def kernel_sum_of_exp(
            time: np.ndarray,
            amplitude: float = 1.0,
            tau_fast_sec: float = 1.0,
            tau_slow_sec: float = 1.0,
            fast_weight: float = 0.5,
            ) -> np.ndarray:
        """Evaluate the simulator sum-of-exponentials kernel wrapper.

        Parameters
        ----------
        time : np.ndarray
            Time points, in seconds.
        amplitude : float, default=1.0
            Overall amplitude multiplier.
        tau_fast_sec : float, default=1.0
            Fast decay time constant, in seconds.
        tau_slow_sec : float, default=1.0
            Slow decay time constant, in seconds.
        fast_weight : float, default=0.5
            Weight applied to the fast exponential.

        Returns
        -------
        np.ndarray
            Kernel values evaluated at ``time``.
        """
        return sum_of_exp_kernel(time, amplitude, tau_fast_sec, tau_slow_sec, fast_weight)

    @staticmethod
    def kernel_gaussian(
            time: np.ndarray,
            amplitude: float = 1.0,
            center_sec: float = 1.0,
            sigma_sec: float = 0.5,
            ) -> np.ndarray:
        """Evaluate the simulator Gaussian kernel wrapper.

        Parameters
        ----------
        time : np.ndarray
            Time points, in seconds.
        amplitude : float, default=1.0
            Peak amplitude.
        center_sec : float, default=1.0
            Center of the Gaussian, in seconds.
        sigma_sec : float, default=0.5
            Standard deviation of the Gaussian, in seconds.

        Returns
        -------
        np.ndarray
            Kernel values evaluated at ``time``.
        """
        return gaussian_kernel(time, amplitude, center_sec, sigma_sec)
    
    @staticmethod 
    def kernel_domed_trapezoid(
            time: np.ndarray,
            amplitude: float,
            rise_duration: float,
            plateau_duration: float,
            decay_duration: float,
            dome_strength: float = 0.0,
            ) -> np.ndarray:
        """
        Broad kernel with S-shaped rise, broad middle dome, and S-shaped decay.

        Parameters
        ----------
        time : np.ndarray
            Time points, in seconds.
        amplitude : float
            Peak height of dome region.
        rise_duration : float
            Duration of the S-shaped increase, in seconds.
        plateau_duration : float
            Duration of the broad middle region, in seconds.
        decay_duration : float
            Duration of the S-shaped decrease, in seconds.
        dome_strength : float
            Strength of curvature of middle "plateau" region.
            Negative values are concave, positive are convex, 
            and ``0`` is flat. Larger magnitudes are increase strength
            of curvature.

        Returns
        -------
        y : np.ndarray
            Kernel values evaluated at ``time``.
        """
        return domed_trapezoid_kernel(
            time, 
            amplitude, 
            rise_duration, 
            plateau_duration, 
            decay_duration, 
            dome_strength,
        )

    #endregion

    ######################
    #region --- EXPORT ---
    ######################
    def to_PhotometryExperiment(
            self,
            as_single_channel: bool = False,
            downsample: int | None = None,
            downsample_kwargs: dict = {},
            ):
        """Export simulated traces as a `PhotometryExperiment`.

        Parameters
        ----------
        as_single_channel : bool, default=False
            If ``True``, omit the isosbestic channel.
        downsample : int or None, default=None
            Downsampling factor applied during export.
        downsample_kwargs : dict, default={}
            Additional keyword arguments passed to the downsampling helpers.

        Returns
        -------
        PhotometryExperiment
            Experiment object containing simulated raw signal, optional
            isosbestic signal, time, frequency, and events.
        """
        EXPORT_iso = None if as_single_channel else self.F_iso
        frequency = self.freq if downsample in (None, 0, 1) else self.freq / downsample

        obj = PhotometryExperiment(
            raw_signal=operations.downsample_signal(self.F_exp, factor=downsample, **downsample_kwargs),
            raw_isosbestic=None if EXPORT_iso is None else operations.downsample_signal(EXPORT_iso, factor=downsample, **downsample_kwargs),
            time=operations.downsample_time(self.time, factor=downsample, **downsample_kwargs),
            frequency=frequency,
            events=self.event_layer.onsets_to_dict(),
        )
        return obj

    def to_PhotometryData(
            self,
            align_to: str | Sequence[str] | float | Sequence[float],
            trial_bounds: tuple[float, float],
            center_on: str | Sequence[str] | None = None,
            window_alignment: Literal['nearest', 'interp'] = 'nearest',
            invalid_window_policy: Literal['drop', 'error'] = 'drop',
            event_conflict_logic: Literal['first', 'last', 'mean'] = 'first',
            ) -> PhotometryData:
        """Extract simulated neural trace into trial-wise `PhotometryData`.

        Parameters
        ----------
        align_to : str, Sequence[str], float, or Sequence[float]
            Event label, event labels, timestamp, or timestamps used to define
            candidate trials.
        trial_bounds : tuple[float, float]
            Window bounds relative to each selected center.
        center_on : str, Sequence[str], or None, default=None
            Optional event labels used to recenter each trial.
        window_alignment : {'nearest', 'interp'}, default='nearest'
            Windowing strategy passed to
            ``PhotometryExperiment.extract_trial_data``.
        invalid_window_policy : {'drop', 'error'}, default='drop'
            Policy for windows that extend outside the signal range.
        event_conflict_logic : {'first', 'last', 'mean'}, default='first'
            Rule used when multiple events of the same label occur in one
            annotation window.

        Returns
        -------
        PhotometryData
            Trial-wise event-layer data.
        """
        # create experiment object
        exp = PhotometryExperiment(
            raw_signal=self.neural_trace_exp,
            raw_isosbestic=None,
            time=self.time,
            events=self.event_layer.timestamps_to_dict(),
        )

        # override to skip preprocessing
        exp.signal = self.E

        # use experiments extract trial infrastructure
        exp.extract_trial_data(
            align_to=align_to,
            trial_bounds=trial_bounds,
            center_on=center_on,
            baseline_bounds=None,
            window_alignment=window_alignment,
            invalid_window_policy=invalid_window_policy,
            event_conflict_logic=event_conflict_logic,
            all_events=True,
        )
        return exp.trial_data

    def to_long_dataframe(
            self,
            only_layers: list[str] | None = None,
            condensed: bool = True,
            downsample: int | None = None,
            downsample_kwargs: dict = {},
            ) -> pd.DataFrame:
        """Export simulated layers to a long dataframe.

        Parameters
        ----------
        only_layers : list[str], str, or None, default=None
            Optional layer name or collection of layer names to include.
        condensed : bool, default=True
            If ``True``, combine related components into summary layers. If
            ``False``, export movement, spike, jump, shot-noise, and
            Gaussian-noise components separately.
        downsample : int or None, default=None
            Downsampling factor applied before export.
        downsample_kwargs : dict, default={}
            Additional keyword arguments passed to the downsampling helpers.

        Returns
        -------
        pd.DataFrame
            Long dataframe with ``time``, ``layer``, ``trace``, and ``value``
            columns.
        """
        # helper
        def _df_builder_helper(arr: np.ndarray | None, layer: str, trace: str, time: np.ndarray) -> pd.DataFrame:
            """Build one long dataframe for a simulated trace component."""
            index = np.arange(time.size)
            if arr is None:
                return pd.DataFrame(index=index)
            else:
                export_arr = operations.downsample_signal(arr, factor=downsample, **downsample_kwargs)
                return pd.DataFrame({'time':time, 'layer':layer, 'trace':trace, 'value':export_arr}, index=index)

        # build export manifests
        if condensed:
            exp_manifest = dict(
                neural_trace = self.neural_trace_exp,
                photobleaching = self.B_exp,
                artifacts = self.A,
                noise = self.N_exp,
                full_signal = self.F_exp,
            )

            iso_manifest = dict(
                neural_trace = self.neural_trace_iso,
                photobleaching = self.B_iso,
                artifacts = self.A,
                noise = self.N_iso,
                full_signal = self.F_iso,
            )

        else:
            exp_manifest = dict(
                events = self.E,
                dynamic_noise = self.D,
                photobleaching = self.B_exp,
                movement_artifacts = self.M,
                spike_artifacts = self.AS,
                jump_artifacts = self.AJ,
                shot_noise = self.N_shot_exp,
                gaussian_noise = self.N_gauss_exp,
                full_signal = self.F_exp,
            )

            iso_manifest = dict(
                events = self.E_iso,
                dynamic_noise = self.D_iso,
                photobleaching = self.B_iso,
                movement_artifacts = self.M,
                spike_artifacts = self.AS,
                jump_artifacts = self.AJ,
                shot_noise = self.N_shot_iso,
                gaussian_noise = self.N_gauss_iso,
                full_signal = self.F_iso,
            )

        # filter layers
        if only_layers is not None:
            exp_manifest = {k : v for k, v in exp_manifest.items() if k in only_layers}
            iso_manifest = {k : v for k, v in iso_manifest.items() if k in only_layers}

        # downsample time
        time = operations.downsample_time(self.time, downsample, **downsample_kwargs)

        # build lists of dfs
        exp_df_list = [_df_builder_helper(arr, layer, 'experimental', time) for layer, arr in exp_manifest.items()]
        iso_df_list = [_df_builder_helper(arr, layer, 'isosbestic', time) for layer, arr in iso_manifest.items()]
        df_list = exp_df_list + iso_df_list

        # concat result and return
        df = pd.concat(df_list, axis=0, ignore_index=True)
        return df

    #endregion

    ########################
    #region --- PLOTTING ---
    ########################

    def plot_layers(
            self,
            condensed: bool = True,
            line_kwargs: dict = {},
            theme_kwargs: dict = {},
            downsample: int | None = None,
            downsample_kwargs: dict = {},
            ) -> ggplot:
        """Plot simulated component layers.

        Parameters
        ----------
        condensed : bool, default=True
            Whether to plot condensed summary layers.
        line_kwargs : dict, default={}
            Keyword arguments forwarded to line geoms.
        theme_kwargs : dict, default={}
            Keyword arguments forwarded to the plot theme helper.
        downsample : int or None, default=None
            Downsampling factor applied before plotting.
        downsample_kwargs : dict, default={}
            Additional keyword arguments passed to the downsampling helpers.

        Returns
        -------
        ggplot
            Plot object for simulated layers.
        """
        long_df = self.to_long_dataframe(
            condensed=condensed,
            downsample=downsample,
            downsample_kwargs=downsample_kwargs,
        )

        p = graphing.plot_simulated_layers(
            long_df,
            condensed,
            line_kwargs,
            theme_kwargs,
        )

        return p

    def plot_traces(
            self,
            line_kwargs: dict = {},
            theme_kwargs: dict = {},
            downsample: int | None = None,
            downsample_kwargs: dict = {},
            ) -> ggplot:
        """Plot final simulated experimental and isosbestic traces.

        Parameters
        ----------
        line_kwargs : dict, default={}
            Keyword arguments forwarded to line geoms.
        theme_kwargs : dict, default={}
            Keyword arguments forwarded to the plot theme helper.
        downsample : int or None, default=None
            Downsampling factor applied before plotting.
        downsample_kwargs : dict, default={}
            Additional keyword arguments passed to the downsampling helpers.

        Returns
        -------
        ggplot
            Plot object for final simulated traces.
        """
        long_df = self.to_long_dataframe(
            only_layers='full_signal',
            condensed=True,
            downsample=downsample,
            downsample_kwargs=downsample_kwargs,
        )

        p = graphing.plot_simulated_traces(
            long_df,
            line_kwargs,
            theme_kwargs,
        )

        return p

    #endregion
