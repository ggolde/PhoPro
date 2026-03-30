import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Sequence, Tuple

from ..core import PhotometryExperiment
from ..utils.ops import downsample_1d


class SimulatedPhotometryGenerator:
    """
    Generator for simulated fiber photometry data.

    Simulated data is generated with the following layers:
      1) event-driven neural signals rendered from timestamped events
      2) B: photobleaching curve, modeled with a 5-param negative bi-exponetial
      3) M: movement attenuation artifacts
      4) noise_dependent(_exp & _iso): Gaussian noise that scales with photobleaching,
        calculated seperately for experimental and isosbestic signal
      5) noise_independent(_exp & _iso): Gaussian noise that is independent of photobleaching,
        calculated seperately for experimental and isosbestic signal

    Experimental signal = (B + event_signal_exp + noise_exp) * M
    Isosbestic signal = (B + event_signal_iso + noise_iso) * M

    Outputs (attributes):
      t: time points
      F_exp, F_iso: experimental and isosbestic signals
      neural_true, neural_norm: true and null-Z normalized experimental event signals
      event_times_sec: times of the base neural events
      events: mapping of event labels to timestamps
      event_peak_specs: per-label peak-generation parameters
      B, M: photobleaching curve and movement attenuation
    """

    def __init__(
        self,
        # time
        T_sec: float = 1000,
        fs: float = 30.0,
        # event
        n_events: int = 100,
        event_dur_sec: float = 3.0,
        buffer_sec: float = 5,
        # neural signal
        A_neural: float = 0.02,
        tau_p_sec: float = 0.6,
        shape_k: int = 3,
        # photobleaching
        bleach_params: dict = dict(a1=50, a2=20, tau1=300, tau2=10000, b0=1),
        bleach_iso_scale: float = 0.9,
        # movement attenuation
        n_artifacts: int = 60,
        artifact_tau_sec: float = 0.8,
        artifact_depth_range: Tuple[float, float] = (0.02, 0.15),
        # noise
        dependent_sigma_exp: float = 1e-4,
        dependent_sigma_iso: float = 1e-4,
        independent_sigma_exp: float = 1e-5,
        independent_sigma_iso: float = 1e-5,
        seed=0,
        # custom artifacting
        artifact_mask: np.ndarray | None = None,
    ):
        """
        Build simulated photometry data.
        Args:
            T_sec (float): Amount of seconds in data.
            fs (float): Sampling frequency in Hz.
            n_events (int): Number of base neural events.
            event_dur_sec (float): How long a base neural event is in seconds.
            A_neural (float): Magnitude of the base event signal.
            tau_p_sec (float): Exponential param for neural signal generation.
            shape_k (int): Controls the skewness of neural signal, must be >= 2.
            buffer_sec (float): Minimum number of seconds between a base neural event and end of time series.
            n_artifacts (int): Number of attenuation artifacts.
            artifact_tau_sec (float): Exponential param for attenuation generation.
            artifact_depth_range (Tuple[float, float]): Fractional bounds for attenuation magnitude.
            dependent_sigma_* (float): Magnitude of intensity-proportional noise.
            independent_sigma_* (float): Magnitude of intensity-independent noise.
            bleach_params (dict): Params for the negative bi-exponential photobleaching model.
            artifact_mask (np.ndarray | None): Optional multiplicative attenuation mask of shape (N,).
        Returns:
            None
        """
        self.fs = float(fs)
        self.rng = np.random.default_rng(seed)

        self.T_sec = float(T_sec)
        self.n_events = int(n_events)
        self.event_dur_sec = float(event_dur_sec)
        self.buffer_sec = float(buffer_sec)

        self.A_neural = float(A_neural)
        self.tau_p_sec = float(tau_p_sec)
        self.shape_k = int(shape_k)

        self.bleach_params = bleach_params
        self.bleach_iso_scale = float(bleach_iso_scale)

        self.n_artifacts = int(n_artifacts)
        self.artifact_tau_sec = float(artifact_tau_sec)
        self.artifact_depth_range = tuple(artifact_depth_range)

        self.dependent_sigma_exp = float(dependent_sigma_exp)
        self.dependent_sigma_iso = float(dependent_sigma_iso)
        self.independent_sigma_exp = float(independent_sigma_exp)
        self.independent_sigma_iso = float(independent_sigma_iso)

        self.artifact_mask = artifact_mask

        self.events: dict[str, np.ndarray] = {}
        self.event_peak_specs: dict[str, dict[str, Any]] = {}
        self.event_signals: dict[str, dict[str, np.ndarray]] = {}

        self.build_layers()

    def build_layers(self):
        self.t, self.N = self._timebase()
        self.artifact_mask = self._validate_artifact_mask(self.artifact_mask)

        self.pulse_kernel, self.L = self._pulse_kernel()
        self.onset_idx, self.event_times_sec = self._even_onsets(buffer_sec=self.buffer_sec)
        self.events = {"event": self.event_times_sec.copy()}
        self.event_peak_specs = {"event": self._default_peak_spec()}
        self.event_signals = {}

        self.B_exp = self._photobleach_layer(**self.bleach_params)
        self.B_iso = self.bleach_iso_scale * self._photobleach_layer(**self.bleach_params)

        self.decay_exp = self.B_exp / self.B_exp.max()
        self.decay_iso = self.B_iso / self.B_iso.max()

        self.M = self._attenuation_layer() * self.artifact_mask
        self.noise_exp, self.noise_iso = self._noise_layer()

        self._refresh_event_layers()

    def _timebase(self):
        N = int(self.T_sec * self.fs)
        if N <= 0:
            raise ValueError("T_sec * fs must be > 0.")
        t = np.arange(N) / self.fs
        return t, N

    def _pulse_kernel(
        self,
        amplitude: float | None = None,
        event_dur_sec: float | None = None,
        tau_p_sec: float | None = None,
        shape_k: int | None = None,
    ):
        amplitude = self.A_neural if amplitude is None else float(amplitude)
        event_dur_sec = self.event_dur_sec if event_dur_sec is None else float(event_dur_sec)
        tau_p_sec = self.tau_p_sec if tau_p_sec is None else float(tau_p_sec)
        shape_k = self.shape_k if shape_k is None else int(shape_k)

        L = int(event_dur_sec * self.fs)
        if L < 3:
            raise ValueError("event_dur_sec * fs must be >= 3 samples.")
        if shape_k < 2:
            raise ValueError("shape_k must be >= 2 for a smooth start at 0.")
        if tau_p_sec <= 0:
            raise ValueError("tau_p_sec must be > 0.")

        tau = np.arange(L) / self.fs

        g = (tau ** (shape_k - 1)) * np.exp(-tau / tau_p_sec)
        g /= g.max() + 1e-12
        h = g * np.hanning(L)
        h /= h.max() + 1e-12
        return amplitude * h, L

    def _even_onsets(self, buffer_sec=5.0):
        buffer_samp = int(round(buffer_sec * self.fs))
        lo = buffer_samp
        hi = (self.N - buffer_samp) - self.L

        if hi <= lo:
            raise ValueError("Buffer too large (or session too short) for chosen event_dur_sec.")

        onset_idx = np.linspace(lo, hi, self.n_events, dtype=int)
        return onset_idx, onset_idx / self.fs

    def _photobleach_layer(self, a1: float, a2: float, tau1: float, tau2: float, b0: float):
        B = a1 * np.exp(-self.t / tau1) + a2 * np.exp(-self.t / tau2) + b0
        if np.any(B <= 0):
            raise ValueError("Photobleaching baseline B must be strictly positive.")
        return B

    def _attenuation_layer(self):
        Lk = max(int(5 * self.artifact_tau_sec * self.fs), 3)
        tk = np.arange(Lk) / self.fs
        k = np.exp(-tk / self.artifact_tau_sec)
        k /= k.max() + 1e-12

        art = np.zeros(self.N)
        if self.n_artifacts > 0:
            onsets = self.rng.choice(np.arange(0, self.N - Lk), size=self.n_artifacts, replace=False)
            depths = self.rng.uniform(*self.artifact_depth_range, size=self.n_artifacts)
            for j0, depth in zip(onsets, depths):
                art[j0:j0 + Lk] += depth * k

        art = np.clip(art, 0.0, 0.8)
        M = 1.0 - art
        if np.any(M <= 0):
            raise ValueError("Attenuation factor M must be > 0 everywhere.")
        return M

    def _noise_layer(self):
        self.dependent_noise_exp = self.decay_exp * self.rng.normal(0.0, self.dependent_sigma_exp, size=self.N)
        self.dependent_noise_iso = self.decay_iso * self.rng.normal(0.0, self.dependent_sigma_iso, size=self.N)
        self.independent_noise_exp = self.rng.normal(0.0, self.independent_sigma_exp, size=self.N)
        self.independent_noise_iso = self.rng.normal(0.0, self.independent_sigma_iso, size=self.N)

        return (
            self.dependent_noise_exp + self.independent_noise_exp,
            self.dependent_noise_iso + self.independent_noise_iso,
        )

    def _validate_artifact_mask(self, artifact_mask: np.ndarray | None) -> np.ndarray:
        if artifact_mask is None:
            return np.ones(self.N, dtype=float)

        artifact_mask = np.asarray(artifact_mask, dtype=float)
        if artifact_mask.shape != (self.N,):
            raise ValueError(f"artifact_mask must have shape ({self.N},), got {artifact_mask.shape}.")
        if np.any(artifact_mask < 0):
            raise ValueError("artifact_mask must be non-negative.")
        return artifact_mask

    def _default_peak_spec(self) -> dict[str, Any]:
        return {
            "amplitude": self.A_neural,
            "event_dur_sec": self.event_dur_sec,
            "tau_p_sec": self.tau_p_sec,
            "shape_k": self.shape_k,
            "channel": "exp",
            "scale_with_bleach": True,
        }

    def _normalize_peak_spec(
        self,
        spec: dict[str, Any] | None,
        fallback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = self._default_peak_spec() if fallback is None else fallback.copy()
        if spec is not None:
            normalized.update(spec)

        normalized["amplitude"] = float(normalized["amplitude"])
        normalized["event_dur_sec"] = float(normalized["event_dur_sec"])
        normalized["tau_p_sec"] = float(normalized["tau_p_sec"])
        normalized["shape_k"] = int(normalized["shape_k"])
        normalized["channel"] = str(normalized["channel"])
        normalized["scale_with_bleach"] = bool(normalized["scale_with_bleach"])

        if normalized["event_dur_sec"] <= 0:
            raise ValueError("peak_specs['event_dur_sec'] must be > 0.")
        if normalized["tau_p_sec"] <= 0:
            raise ValueError("peak_specs['tau_p_sec'] must be > 0.")
        if normalized["shape_k"] < 2:
            raise ValueError("peak_specs['shape_k'] must be >= 2.")
        if normalized["channel"] not in {"exp", "iso", "both"}:
            raise ValueError("peak_specs['channel'] must be one of: 'exp', 'iso', 'both'.")
        return normalized

    def _validate_add_event_args(
        self,
        time_range: tuple[float, float],
        overall_prob: float,
        choices: Sequence[str],
        choice_probs: Sequence[float],
        relative_to: str,
        peak_specs: dict[str, dict[str, Any]] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if relative_to not in self.events:
            raise KeyError(f"relative_to '{relative_to}' not found in events: {list(self.events)}")
        if len(time_range) != 2:
            raise ValueError("time_range must contain exactly two values.")
        if time_range[0] > time_range[1]:
            raise ValueError("time_range[0] must be <= time_range[1].")
        if not 0 <= overall_prob <= 1:
            raise ValueError("overall_prob must be between 0 and 1.")
        if len(choices) == 0:
            raise ValueError("choices must contain at least one event label.")
        if len(choices) != len(choice_probs):
            raise ValueError("choices and choice_probs must have the same length.")
        if len(set(choices)) != len(choices):
            raise ValueError("choices must be unique.")

        labels = np.asarray(choices, dtype=object)
        probs = np.asarray(choice_probs, dtype=float)
        if np.any(~np.isfinite(probs)):
            raise ValueError("choice_probs must contain only finite values.")
        if np.any(probs < 0):
            raise ValueError("choice_probs must be non-negative.")
        prob_sum = probs.sum()
        if prob_sum <= 0:
            raise ValueError("choice_probs must sum to a positive value.")

        if peak_specs is not None:
            invalid_peak_spec_labels = set(peak_specs) - set(labels.tolist())
            if invalid_peak_spec_labels:
                raise ValueError(
                    "peak_specs contains labels not present in choices: "
                    f"{sorted(invalid_peak_spec_labels)}"
                )

        return labels, probs / prob_sum

    def _add_kernel_to_trace(self, trace: np.ndarray, kernel: np.ndarray, onset_idx: int) -> None:
        start = int(onset_idx)
        stop = start + kernel.size
        if stop <= 0 or start >= self.N:
            return

        kernel_start = 0
        kernel_stop = kernel.size
        if start < 0:
            kernel_start = -start
            start = 0
        if stop > self.N:
            kernel_stop -= stop - self.N
            stop = self.N

        if kernel_stop > kernel_start:
            trace[start:stop] += kernel[kernel_start:kernel_stop]

    def _render_event_train(self, timestamps: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
        trace = np.zeros(self.N)
        timestamps = np.asarray(timestamps, dtype=float)
        if timestamps.size == 0:
            return trace

        kernel, _ = self._pulse_kernel(
            amplitude=spec["amplitude"],
            event_dur_sec=spec["event_dur_sec"],
            tau_p_sec=spec["tau_p_sec"],
            shape_k=spec["shape_k"],
        )
        onset_idxs = np.rint((timestamps - self.t[0]) * self.fs).astype(int)
        for onset_idx in onset_idxs:
            self._add_kernel_to_trace(trace, kernel, onset_idx)
        return trace

    def _refresh_event_layers(self) -> None:
        neural_true = np.zeros(self.N)
        neural_decaying = np.zeros(self.N)
        iso_event_true = np.zeros(self.N)
        iso_event_decaying = np.zeros(self.N)
        event_signals: dict[str, dict[str, np.ndarray]] = {}

        for label, timestamps in self.events.items():
            spec = self.event_peak_specs.get(label, self._default_peak_spec())
            rendered = self._render_event_train(timestamps=timestamps, spec=spec)

            label_signals = {
                "exp_true": np.zeros(self.N),
                "exp": np.zeros(self.N),
                "iso_true": np.zeros(self.N),
                "iso": np.zeros(self.N),
            }

            if spec["channel"] in {"exp", "both"}:
                label_signals["exp_true"] = rendered.copy()
                label_signals["exp"] = rendered * self.decay_exp if spec["scale_with_bleach"] else rendered.copy()
                neural_true += label_signals["exp_true"]
                neural_decaying += label_signals["exp"]

            if spec["channel"] in {"iso", "both"}:
                label_signals["iso_true"] = rendered.copy()
                label_signals["iso"] = rendered * self.decay_iso if spec["scale_with_bleach"] else rendered.copy()
                iso_event_true += label_signals["iso_true"]
                iso_event_decaying += label_signals["iso"]

            event_signals[label] = label_signals

        self.event_signals = event_signals
        self.neural_true = neural_true
        self.neural_decaying = neural_decaying
        self.iso_event_true = iso_event_true
        self.iso_event_decaying = iso_event_decaying

        std = np.std(self.neural_true)
        self.neural_norm = self.neural_true / std if std > 0 else np.zeros_like(self.neural_true)

        self.F_exp = (self.B_exp + self.neural_decaying + self.noise_exp) * self.M
        self.F_iso = (self.B_iso.copy() + self.iso_event_decaying + self.noise_iso) * self.M

    def add_event(
        self,
        time_range: tuple[float, float],
        overall_prob: float,
        choices: list[str],
        choice_probs: list[float],
        peak_specs: dict[str, dict[str, Any]] | None = None,
        relative_to: str = "event",
        allow_out_of_bounds: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Add stochastic event timestamps relative to an existing event stream and rebuild the
        simulated signal so the new events contribute peaks.

        Args:
            time_range (tuple[float, float]): Relative time bounds around each anchor event.
            overall_prob (float): Probability that an event occurs for each anchor event.
            choices (list[str]): Event labels to assign to generated timestamps.
            choice_probs (list[float]): Relative probabilities for each label in ``choices``.
            peak_specs (dict[str, dict] | None): Optional per-label peak settings. Each label-specific
                dict can include ``amplitude``, ``event_dur_sec``, ``tau_p_sec``, ``shape_k``,
                ``channel`` ('exp', 'iso', or 'both'), and ``scale_with_bleach``.
            relative_to (str): Existing event label used as the anchor event.
            allow_out_of_bounds (bool): If False, drop generated events outside the session time bounds.
        Returns:
            dict[str, np.ndarray]: Updated timestamp arrays for the labels in ``choices``.
        """
        labels, probs = self._validate_add_event_args(
            time_range=time_range,
            overall_prob=overall_prob,
            choices=choices,
            choice_probs=choice_probs,
            relative_to=relative_to,
            peak_specs=peak_specs,
        )

        anchors = np.asarray(self.events[relative_to], dtype=float)
        occurs = self.rng.random(anchors.size) < overall_prob
        new_times = anchors[occurs] + self.rng.uniform(time_range[0], time_range[1], size=occurs.sum())
        new_labels = self.rng.choice(labels, size=occurs.sum(), p=probs)

        if not allow_out_of_bounds:
            in_bounds = (new_times >= self.t[0]) & (new_times <= self.t[-1])
            new_times = new_times[in_bounds]
            new_labels = new_labels[in_bounds]

        peak_specs = {} if peak_specs is None else peak_specs
        updated = {}
        for label in labels:
            label = str(label)
            existing = np.asarray(self.events.get(label, np.array([], dtype=float)), dtype=float)
            added = np.asarray(new_times[new_labels == label], dtype=float)
            merged = existing.copy() if added.size == 0 else np.sort(np.concatenate([existing, added]))
            self.events[label] = merged

            fallback = self.event_peak_specs.get(label, self._default_peak_spec())
            self.event_peak_specs[label] = self._normalize_peak_spec(peak_specs.get(label), fallback=fallback)
            updated[label] = merged.copy()

        self._refresh_event_layers()
        return updated

    def window(self, series: np.ndarray, center: float, bounds: Tuple[float, float]):
        low, high = bounds
        window_idxs = np.searchsorted(self.t, [center + low, center + high])
        return series[window_idxs[0]:window_idxs[1]]

    def plot_layers(self):
        fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True, sharey="row", figsize=(10, 15))
        t = self.t
        axs[0, 0].plot(t, self.neural_true)
        axs[0, 0].set_title("Experimental signal")
        axs[0, 0].set_ylabel("Nueral Signal")

        axs[0, 1].plot(t, self.iso_event_true)
        axs[0, 1].set_title("Isosbestic signal")

        axs[1, 0].set_ylabel("Photobleaching")
        axs[1, 0].plot(t, self.B_exp)
        axs[1, 1].plot(t, self.B_iso)

        axs[2, 0].set_ylabel("Attenuation")
        axs[2, 0].plot(t, self.M)
        axs[2, 1].plot(t, self.M)

        axs[3, 0].set_ylabel("Noise")
        axs[3, 0].plot(t, self.noise_exp)
        axs[3, 1].plot(t, self.noise_iso)

        axs[4, 0].set_ylabel("Full Signal")
        axs[4, 0].plot(t, self.F_exp)
        axs[4, 1].plot(t, self.F_iso)

    def get_single_event(
        self,
        event_idx: int = 0,
        buffer: tuple[float, float] = (0, 0),
        label: str = "event",
        channel: str = "exp",
        decayed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if label not in self.events:
            raise KeyError(f"label '{label}' not found in events: {list(self.events)}")
        if channel not in {"exp", "iso"}:
            raise ValueError("channel must be either 'exp' or 'iso'.")

        event_onset = self.events[label][event_idx]
        event_dur_sec = self.event_peak_specs.get(label, self._default_peak_spec())["event_dur_sec"]

        left = event_onset + buffer[0]
        right = event_onset + event_dur_sec + buffer[1]

        l_idx = np.searchsorted(self.t, left, side="left")
        r_idx = np.searchsorted(self.t, right, side="right")

        signal_key = channel if decayed else f"{channel}_true"
        event_ts = self.event_signals[label][signal_key][l_idx:r_idx]
        ts = self.t[l_idx:r_idx] - event_onset

        return event_ts, ts

    def to_dict(self):
        return {
            "t": self.t,
            "fs": self.fs,
            "F_exp": self.F_exp,
            "F_iso": self.F_iso,
            "neural_true": self.neural_true,
            "neural_decaying": self.neural_decaying,
            "event_times_sec": self.event_times_sec,
            "events": {label: times.copy() for label, times in self.events.items()},
            "event_peak_specs": {label: spec.copy() for label, spec in self.event_peak_specs.items()},
            "B_exp": self.B_exp,
            "B_iso": self.B_iso,
            "M": self.M,
            "pulse_kernel": self.pulse_kernel,
        }

    def to_PhotometryExperiment(self, downsample: int = 10):
        obj = PhotometryExperiment(
            raw_signal=downsample_1d(self.F_exp, factor=downsample),
            raw_isosbestic=downsample_1d(self.F_iso, factor=downsample),
            time=downsample_1d(self.t, factor=downsample),
            frequency=self.fs / downsample,
            events={label: times.copy() for label, times in self.events.items()},
        )
        return obj
