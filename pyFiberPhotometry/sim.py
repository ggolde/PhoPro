import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

from .core import PhotometryData, PhotometryExperiment

class SimulatedPhotometryGenerator:
    """
    Generator for simulated fiber photometry data.

    Simulated data is generated with the following layers:
      1) neural_true: evenly spaced, identical finite-support pulses representing true signal
      2) B: photobleaching curve, modeled with a 5-param negative bi-exponetial
      3) M: movement attenuation artifacts
      4) noise_dependent(_exp & _iso): Gaussian noise that scales with photobleaching, 
        calculated seperately for experimental and isosbestic signal
      5) noise_independent(_exp & _iso): Gaussian noise that is independent of photobleaching, 
        calculated seperately for experimental and isosbestic signal
    
    Experimental signal = (B + (neural_true * B/B.max()) * M) + (noise_dependent_exp * B/B.mox()) + noise_independent_exp
    Isosbestic signal = (B * M) + (noise_dependent_exp * B/B.mox()) + noise_independent_exp

    Outputs (attributes):
      t: time points
      F_exp, F_iso: experimental and isosbestic signals
      neural_true, neural_norm: true and null-Z normalized neural signals
      event_times_sec: times of neural events
      B, M: photobleaching curve and movement attenuation
    """

    def __init__(
        self,
        # time
        T_sec: float = 1000, fs: float = 30.0,
        # event
        n_events: int = 100, event_dur_sec: float = 3.0,
        buffer_sec: float = 5,
        # neural signal
        A_neural: float = 0.02, tau_p_sec: float = 0.6, 
        shape_k: int = 3, 
        # photobleaching
        a1: float = 1.0, tau1_sec: float = 120.0, 
        a2: float = 0.4, tau2_sec: float = 900.0, 
        b0: float = 0.2,
        # movement attenuation
        n_artifacts: int = 60, artifact_tau_sec: float = 0.8, 
        artifact_depth_range: Tuple[float, float] = (0.02, 0.15),
        # noise
        dependent_sigma_exp: float=1e-4, dependent_sigma_iso: float=1e-4,
        independent_sigma_exp: float=1e-5, independent_sigma_iso: float=1e-5,
        seed=0,
    ):
        """
        Build simulated photometry data
        Args:
            T_sec (float): Amount of seconds in data.
            fs (float): Sampling frequency in Hz.
            n_events (int): Number of neural events.
            event_dur_sec (float): How long a neural event is in seconds.
            A_neural (float): Magnitude of neural signal.
            tau_p_sec (float): Exponential param for neural signal generation.
            shape_k (int): Controls the skewness of neural signal, must be >= 3.
            buffer_sec (float): Minimum number of second between a neural event and end of time series.
            n_artifact (int): Number of attenuation artifacts.
            artifact_tau_sec (float): Exponential param for attenuation generation.
            aritifact_depth_range (Tuple[float, float]): The fractional bounds for magnitude of attenuation.
            dependent_sigma (float): Magnitude of intensity-proportional noise for experimental and isosbestic signals independently.
            independent_sigma (float): Magnitude of intensity-independent noise for experimental and isosbestic signals independently.
            a tau_sec b0 (floats): parmas for negative bi-exponetial photobleaching model 
                = a1 * exp(-time/tau1_sec) + a2 * exp(-time/tau2_sec) + b0
        Returns:
            None
        """
        self.fs = float(fs)
        self.rng = np.random.default_rng(seed)

        # store params
        self.T_sec = float(T_sec)
        self.n_events = int(n_events)
        self.event_dur_sec = float(event_dur_sec)

        self.A_neural = float(A_neural)
        self.tau_p_sec = float(tau_p_sec)
        self.shape_k = int(shape_k)

        self.a1, self.tau1_sec = float(a1), float(tau1_sec)
        self.a2, self.tau2_sec = float(a2), float(tau2_sec)
        self.b0 = float(b0)

        self.n_artifacts = int(n_artifacts)
        self.artifact_tau_sec = float(artifact_tau_sec)
        self.artifact_depth_range = tuple(artifact_depth_range)

        self.dependent_sigma_exp = float(dependent_sigma_exp)
        self.dependent_sigma_iso = float(dependent_sigma_iso)
        self.independent_sigma_exp = float(independent_sigma_exp)
        self.independent_sigma_iso = float(independent_sigma_iso)

        # build layers
        self.t, self.N = self._timebase()
        self.pulse_kernel, self.L = self._pulse_kernel()
        self.onset_idx, self.event_times_sec = self._even_onsets(buffer_sec=buffer_sec)

        self.neural_true = self._neural_layer()
        self.neural_norm = self.neural_true / np.std(self.neural_true)
        self.B = self._photobleach_layer()

        self.decay_curve = self.B / self.B.max()
        self.neural_decaying = self.neural_true * self.decay_curve

        F_exp0 = self.B + self.neural_decaying
        F_iso0 = self.B.copy()

        self.M = self._attenuation_layer()
        F_exp1, F_iso1 = F_exp0 * self.M, F_iso0 * self.M

        self.noise_exp, self.noise_iso = self._noise_layer()

        self.F_exp = F_exp1 + self.noise_exp
        self.F_iso = F_iso1 + self.noise_iso

    # layer calculators
    def _timebase(self):
        N = int(self.T_sec * self.fs)
        if N <= 0:
            raise ValueError("T_sec * fs must be > 0.")
        t = np.arange(N) / self.fs
        return t, N

    def _pulse_kernel(self):
        L = int(self.event_dur_sec * self.fs)
        if L < 3:
            raise ValueError("event_dur_sec * fs must be >= 3 samples.")
        if self.shape_k < 2:
            raise ValueError("shape_k must be >= 2 for a smooth start at 0.")

        tau = np.arange(L) / self.fs

        # gamma-like pulse, then Hann taper to forces 0 at endpoints
        g = (tau ** (self.shape_k - 1)) * np.exp(-tau / self.tau_p_sec)
        g /= (g.max() + 1e-12)
        h = g * np.hanning(L)
        h /= (h.max() + 1e-12)
        return h, L

    def _even_onsets(self, buffer_sec=5.0):
        buffer_samp = int(round(buffer_sec * self.fs))
        lo = buffer_samp
        hi = (self.N - buffer_samp) - self.L  # last onset so pulse fits before end buffer

        if hi <= lo:
            raise ValueError(
                "Buffer too large (or session too short) for chosen event_dur_sec."
            )

        onset_idx = np.linspace(lo, hi, self.n_events, dtype=int)
        return onset_idx, onset_idx / self.fs

    def _neural_layer(self):
        neural = np.zeros(self.N)
        h = self.A_neural * self.pulse_kernel
        for i0 in self.onset_idx:
            neural[i0:i0 + self.L] += h
        return neural

    def _photobleach_layer(self):
        B = (
            self.a1 * np.exp(-self.t / self.tau1_sec)
            + self.a2 * np.exp(-self.t / self.tau2_sec)
            + self.b0
        )
        if np.any(B <= 0):
            raise ValueError("Photobleaching baseline B must be strictly positive.")
        return B

    def _attenuation_layer(self):
        # smooth dip kernel for movement/attenuation artifacts
        Lk = max(int(5 * self.artifact_tau_sec * self.fs), 3)
        tk = np.arange(Lk) / self.fs
        k = np.exp(-tk / self.artifact_tau_sec)
        k /= (k.max() + 1e-12)

        art = np.zeros(self.N)
        if self.n_artifacts > 0:
            onsets = self.rng.choice(np.arange(0, self.N - Lk), size=self.n_artifacts, replace=False)
            depths = self.rng.uniform(*self.artifact_depth_range, size=self.n_artifacts)
            for j0, d in zip(onsets, depths):
                art[j0:j0 + Lk] += d * k

        art = np.clip(art, 0.0, 0.8)
        M = 1.0 - art
        if np.any(M <= 0):
            raise ValueError("Attenuation factor M must be > 0 everywhere.")
        return M
    
    def _noise_layer(self):
        self.dependent_noise_exp = self.decay_curve * (self.rng.normal(0.0, self.dependent_sigma_exp, size=self.N))
        self.dependent_noise_iso = self.decay_curve * (self.rng.normal(0.0, self.dependent_sigma_iso, size=self.N))
        self.independent_noise_exp = self.rng.normal(0.0, self.independent_sigma_exp, size=self.N)
        self.independent_noise_iso = self.rng.normal(0.0, self.independent_sigma_iso, size=self.N)

        return (self.dependent_noise_exp + self.independent_noise_exp), (self.dependent_noise_iso + self.independent_noise_iso)
    
    # operations
    def window(self, series: np.ndarray, center: float, bounds: Tuple[float, float]):
        low, high = bounds
        window_idxs = np.searchsorted(self.t, [center + low, center + high])
        return series[window_idxs[0]:window_idxs[1]]
    
    # visualizations
    def plot_layers(self):
        fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True, sharey='row', figsize=(10, 15))
        t = self.t
        axs[0, 0].plot(t, self.neural_true)
        axs[0, 0].set_title('Experimental signal')
        axs[0, 0].set_ylabel('Nueral Signal')

        axs[0, 1].plot(t, np.zeros_like(self.neural_true))
        axs[0, 1].set_title('Isosbestic signal')

        axs[1, 0].set_ylabel('Photobleaching')
        axs[1, 0].plot(t, self.B)
        axs[1, 1].plot(t, self.B)

        axs[2, 0].set_ylabel('Attenuation')
        axs[2, 0].plot(t, self.M)
        axs[2, 1].plot(t, self.M)

        axs[3, 0].set_ylabel('Noise')
        axs[3, 0].plot(t, self.noise_exp)
        axs[3, 1].plot(t, self.noise_iso)

        axs[4, 0].set_ylabel('Full Signal')
        axs[4, 0].plot(t, self.F_exp)
        axs[4, 1].plot(t, self.F_iso)


    # export
    def to_dict(self):
        return {
            "t": self.t,
            "fs": self.fs,
            "F_exp": self.F_exp,
            "F_iso": self.F_iso,
            "neural_true": self.neural_true,
            "event_times_sec": self.event_times_sec,
            "B": self.B,
            "M": self.M,
            "pulse_kernel": self.pulse_kernel,
        }
    
    def to_PhotometryExperiment(self):
        events = {'event' : self.event_times_sec}
        return PhotometryExperiment.manual_init(
            raw_signal=self.F_exp,
            raw_isosbestic=self.F_iso,
            time=self.t,
            frequency=self.fs,
            events=events,
        )