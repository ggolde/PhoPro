import numpy as np
import pandas as pd

from dataclasses import dataclass

@dataclass
class WindowResult:
    signals: np.ndarray
    times: np.ndarray
    time_grid: np.ndarray
    invalid_mask: np.ndarray
    events: dict[str, np.ndarray]

    @property
    def is_empty(self) -> bool: return self.signals is None

    @classmethod
    def empty(cls) -> "WindowResult":
        return WindowResult(None, None, None, None, None)

    def apply_mask(self, mask: np.ndarray) -> None:
        if self.is_empty:
            return
        self.signals = self.signals[mask, :]
        self.times = self.times[mask, :]
        self.invalid_mask = self.invalid_mask[mask]
        self.events = {
            k : arr[mask] for k, arr in self.events.items()
        }

def _build_relative_time_grid(bounds: tuple[float, float], freq: float) -> np.ndarray:
    low, high = bounds
    n = int(round((high - low) * freq))
    return low + np.arange(n, dtype=float) / freq

def _build_nearest_time_grid(bounds: tuple[float, float], freq: float) -> np.ndarray:
    n = int(round((bounds[1] - bounds[0]) * freq))
    start_offset = int(np.rint(bounds[0] * freq))
    return (start_offset + np.arange(n, dtype=int)) / freq

def _snap_to_nearest_timepoint(timestamps: np.ndarray, time: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=float)
    out = np.full_like(timestamps, np.nan, dtype=float)

    valid = ~np.isnan(timestamps)
    if not np.any(valid):
        return out

    vals = timestamps[valid]
    idx = np.searchsorted(time, vals, side='left')
    idx = np.clip(idx, 1, len(time) - 1)

    left = time[idx - 1]
    right = time[idx]
    idx = idx - (vals - left <= right - vals)

    out[valid] = time[idx]
    return out

def create_windows_nearest(
        signal: np.ndarray,
        time: np.ndarray,
        events: dict[str, np.ndarray],
        centers: np.ndarray,
        bounds: tuple[float, float],
        freq: float,
        ) -> WindowResult:
    # construct shared time grid
    common_time_grid = _build_nearest_time_grid(bounds, freq)
    n = common_time_grid.size

    # snap centers to nearest time sample
    snapped_centers = _snap_to_nearest_timepoint(centers, time)

    # slice windows
    center_idxs = np.searchsorted(time, snapped_centers, side='left')
    center_idxs = np.clip(center_idxs, 0, len(time) - 1)

    start_offset = int(np.rint(bounds[0] * freq))
    window_idxs = center_idxs[:, None] + start_offset + np.arange(n)[None, :]
    window_idxs = np.clip(window_idxs, 0, len(time) - 1)

    signal_windows = signal[window_idxs]
    time_windows = time[window_idxs] - snapped_centers[:, None]

    # snap and center all events
    events_centered = {
        k: _snap_to_nearest_timepoint(v, time) - snapped_centers
        for k, v in events.items()
    }

    # check window validity
    too_low = (snapped_centers + common_time_grid[0] < time[0])
    too_high = (snapped_centers + common_time_grid[-1] > time[-1]) 
    invalid_window_mask = too_low | too_high

    result = WindowResult(
        signals=signal_windows,
        times=time_windows,
        time_grid=common_time_grid,
        invalid_mask=invalid_window_mask,
        events=events_centered
    )
    return result

def create_windows_interp(
        signal: np.ndarray,
        time: np.ndarray,
        events: dict[str, np.ndarray],
        centers: np.ndarray,
        bounds: tuple[float, float],
        freq: float,
        ) -> WindowResult:

    common_time_grid = _build_relative_time_grid(bounds, freq)
    sample_t = centers[:, None] + common_time_grid[None, :]

    signal_windows = np.vstack([
        np.interp(t_row, time, signal)
        for t_row in sample_t
    ])

    time_windows = np.broadcast_to(common_time_grid, signal_windows.shape).copy()
    events_centered = {k: v - centers for k, v in events.items()}

    # check window validity
    too_low = (centers + bounds[0] < time[0])
    too_high = (centers + bounds[-1] > time[-1])
    invalid_window_mask = too_low | too_high

    result = WindowResult(
        signals=signal_windows,
        times=time_windows,
        time_grid=common_time_grid,
        invalid_mask=invalid_window_mask,
        events=events_centered
    )
    return result