import numpy as np
import pandas as pd

from dataclasses import dataclass

@dataclass
class WindowResult:
    signals: np.ndarray
    times: np.ndarray
    time_grid: np.ndarray
    centers: np.ndarray
    invalid_mask: np.ndarray
    events: dict[str, np.ndarray] | None
    layers: dict[str, np.ndarray] | None

    @property
    def n_windows(self) -> int: return self.signals.shape[0]
    @property
    def has_events(self) -> bool: return self.events is not None
    @property
    def has_layers(self) -> bool: return self.layers is not None

    def apply_mask(self, mask: np.ndarray) -> None:
        self.signals = self.signals[mask, :]
        self.times = self.times[mask, :]
        self.invalid_mask = self.invalid_mask[mask]
        self.centers = self.centers[mask]

        if self.has_events:
            self.events = {k : arr[mask] for k, arr in self.events.items()}

        if self.has_layers:
            self.layers = {k : arr[mask] for k, arr in self.layers.items()}

    def to_obs(self, add_trial_num: bool = False) -> pd.DataFrame:
        data = {}
        if add_trial_num:
            data['trial_num'] = np.arange(self.n_windows) + 1
        
        data['start_time'] = self.centers + self.time_grid[0]
        data['stop_time'] = self.centers + self.time_grid[-1]

        data.update(self.events)

        return pd.DataFrame(data)


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

def _find_invalid_windows(centers: np.ndarray, new_time: np.ndarray | tuple, old_time: np.ndarray) -> np.ndarray:
    new_time = np.asarray(new_time)
    too_low = (centers + new_time.min() < old_time.min())
    too_high = (centers + new_time.max() > old_time.max()) 
    is_invalid = too_low | too_high
    return is_invalid

def _validate_window_input_1D(
        signal: np.ndarray,
        time: np.ndarray,
        events: dict[str, np.ndarray],
        centers: np.ndarray,
        bounds: tuple[float, float],
        ) -> None:
    
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {signal.shape}")
    
    if time.ndim != 1:
        raise ValueError(f"time must be 1D, got shape {time.shape}")
    
    if not np.all(np.diff(time) > 0):
        raise ValueError("time must be strictly increasing")
    
    if centers.ndim != 1:
        raise ValueError(f"centers must be 1D, got shape {centers.shape}")
    
    if signal.size != time.size:
        raise ValueError(f"signal size ({signal.size}) must match time size ({time.size})")
    
    if len(bounds) != 2:
        raise ValueError(f"Bounds must be of length 2, got {len(bounds)}.")
    
    if bounds[0] > bounds[-1]:
        raise ValueError(f"Left bound ({bounds[0]}) must be less than right bound ({bounds[-1]})")
    
    for label, arr in events.items():
        if arr.ndim != 1:
            raise ValueError(f"Event array for {label} must be 1D, got shape {arr.shape}")
        if arr.size != centers.size:
            raise ValueError(f"Event array for {label} is not the same length as centers.")

def _validate_window_input_2D(
        signal: np.ndarray,
        time: np.ndarray,
        centers: np.ndarray,
        bounds: tuple[float, float],
        ) -> None:
    
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D, got shape {signal.shape}")
    
    if time.ndim != 1:
        raise ValueError(f"time must be 1D, got shape {time.shape}")

    if not np.all(np.diff(time) > 0):
        raise ValueError("time must be strictly increasing")
    
    if centers.ndim != 1:
        raise ValueError(f"centers must be 1D, got shape {centers.shape}")
    
    if signal.shape[0] != centers.size:
        raise ValueError(f"centers size ({centers.size}) must match signal sample axis ({signal.shape[0]})")
    
    if signal.shape[1] != time.size:
        raise ValueError(f"signal time axis ({signal.shape[1]}) must match time size ({time.size})")
    
    if len(bounds) != 2:
        raise ValueError(f"Bounds must be of length 2, got {len(bounds)}.")
    
    if bounds[0] > bounds[-1]:
        raise ValueError(f"Left bound ({bounds[0]}) must be less than right bound ({bounds[-1]})")

def create_windows_nearest_1D(
        signal: np.ndarray,
        time: np.ndarray,
        events: dict[str, np.ndarray],
        centers: np.ndarray,
        bounds: tuple[float, float],
        freq: float,
        ) -> WindowResult:
    """Window a 1D array through nearest sample snapping."""
    _validate_window_input_1D(signal, time, events, centers, bounds)

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
    invalid_window_mask = _find_invalid_windows(snapped_centers, common_time_grid, time)
    signal_windows[invalid_window_mask, :] = np.nan
    time_windows[invalid_window_mask, :] = np.nan

    # package result
    result = WindowResult(
        signals=signal_windows,
        times=time_windows,
        time_grid=common_time_grid,
        centers=snapped_centers,
        invalid_mask=invalid_window_mask,
        events=events_centered,
        layers=None,
    )
    return result

def create_windows_interp_1D(
        signal: np.ndarray,
        time: np.ndarray,
        events: dict[str, np.ndarray],
        centers: np.ndarray,
        bounds: tuple[float, float],
        freq: float,
        ) -> WindowResult:
    """Windows a 1D array through linear intepretation."""
    _validate_window_input_1D(signal, time, events, centers, bounds)

    # construct new time grid
    common_time_grid = _build_relative_time_grid(bounds, freq)
    sample_t = centers[:, None] + common_time_grid[None, :]

    # apply windowing and center events
    signal_windows = np.vstack([
        np.interp(t_row, time, signal)
        for t_row in sample_t
    ])
    time_windows = np.broadcast_to(common_time_grid, signal_windows.shape).copy()
    events_centered = {k: v - centers for k, v in events.items()}

    # check window validity
    invalid_window_mask = _find_invalid_windows(centers, bounds, time)
    signal_windows[invalid_window_mask, :] = np.nan
    time_windows[invalid_window_mask, :] = np.nan

    # package result
    result = WindowResult(
        signals=signal_windows,
        times=time_windows,
        time_grid=common_time_grid,
        centers=centers,
        invalid_mask=invalid_window_mask,
        events=events_centered,
        layers=None,
    )
    return result

def create_windows_nearest_2D(
        signal: np.ndarray,
        time: np.ndarray,
        centers: np.ndarray,
        bounds: tuple[float, float],
        freq: float,
        events: dict[str, np.ndarray] | None,
        layers: dict[str, np.ndarray] | None,
        ) -> WindowResult:
    """Window a 2D array by snapping centers to nearest time sample."""
    # validate
    _validate_window_input_2D(signal, time, centers, bounds)

    # construct new time grid
    common_time_grid = _build_nearest_time_grid(bounds, freq)
    n = common_time_grid.size

    # center snapped center idxs
    snapped_centers = _snap_to_nearest_timepoint(centers, time)
    center_idxs = np.searchsorted(time, snapped_centers, side='left')
    center_idxs = np.clip(center_idxs, 0, len(time) - 1)

    # build window indexes
    start_offset = int(np.rint(bounds[0] * freq))
    window_idxs = center_idxs[:, None] + start_offset + np.arange(n)[None, :]
    window_idxs = np.clip(window_idxs, 0, len(time) - 1)

    # apply windowing
    row_idxs = np.arange(signal.shape[0])[:, None]
    signal_windows = signal[row_idxs, window_idxs]
    time_windows = time[window_idxs] - snapped_centers[:, None]
    layer_windows = None if layers is None else {k : arr[row_idxs, window_idxs] for k, arr in layers.items()}

    # center events
    if events is not None:
        events_centered = {
            k: _snap_to_nearest_timepoint(v, time) - snapped_centers
            for k, v in events.items()
        }
    else:
        events_centered = None

    # check for invalid windows
    invalid_window_mask = _find_invalid_windows(snapped_centers, common_time_grid, time)
    signal_windows[invalid_window_mask, :] = np.nan
    time_windows[invalid_window_mask, :] = np.nan

    return WindowResult(
        signals=signal_windows,
        times=time_windows,
        time_grid=common_time_grid,
        centers=snapped_centers,
        invalid_mask=invalid_window_mask,
        events=events_centered,
        layers=layer_windows,
    )

def create_windows_interp_2D(
        signal: np.ndarray,
        time: np.ndarray,
        centers: np.ndarray,
        bounds: tuple[float, float],
        freq: float,
        events: dict[str, np.ndarray] | None,
        layers: dict[str, np.ndarray] | None,
        ) -> WindowResult:
    """Window a 2D array by interpolating each row onto a common centered time grid."""
    # validate
    _validate_window_input_2D(signal, time, centers, bounds)

    # construct new time grid
    common_time_grid = _build_relative_time_grid(bounds, freq)
    sample_t = centers[:, None] + common_time_grid[None, :]

    # build and apply windows
    signal_windows = np.vstack([
        np.interp(sample_t[i], time, signal[i])
        for i in range(signal.shape[0])
    ])
    time_windows = np.broadcast_to(common_time_grid, signal_windows.shape).copy()
    if layers is not None:
        layer_windows = {
            k : np.vstack([
                np.interp(sample_t[i], time, layer[i])
                for i in range(layer.shape[0])
            ])
            for k, layer in layers.items()
        }
    else:
        layer_windows = None

    # center events
    if events is not None:
        events_centered = {k: v - centers for k, v in events.items()}
    else:
        events_centered = None
        

    # check for invalid windows
    invalid_window_mask = _find_invalid_windows(centers, bounds, time)
    signal_windows[invalid_window_mask, :] = np.nan
    time_windows[invalid_window_mask, :] = np.nan

    return WindowResult(
        signals=signal_windows,
        times=time_windows,
        time_grid=common_time_grid,
        centers=centers,
        invalid_mask=invalid_window_mask,
        events=events_centered,
        layers=layer_windows,
    )
