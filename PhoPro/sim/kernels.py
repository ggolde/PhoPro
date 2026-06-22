"""Kernel and taper functions for simulated photometry events."""

from typing import Literal
import numpy as np

from scipy.signal.windows import hann

##############################
#region --- EDGE SMOOTHING ---
##############################

def half_hann(n: int, side: str = "right") -> np.ndarray:
    """Return one half of a Hann window.

    Parameters
    ----------
    n : int
        Number of points in the returned half-window.
    side : {'left', 'right'}, default='right'
        Which side of the Hann window to return.

    Returns
    -------
    np.ndarray
        Half-Hann window with length ``n``.

    Raises
    ------
    ValueError
        If ``n`` is less than 2 or ``side`` is not recognized.
    """
    if n < 2:
        raise ValueError("n must be at least 2")

    window = hann(2 * n - 1, sym=True)

    if side == "left":
        return window[:n]
    elif side == "right":
        return window[n - 1:]
    else:
        raise ValueError("side must be 'left' or 'right'")

def taper_end_hann(y: np.ndarray, taper_fraction: float, side: Literal['left', 'right'], min_points: int = 5) -> np.ndarray:
    """Taper one end of an array in place with a half-Hann window.

    Parameters
    ----------
    y : np.ndarray
        Array to modify.
    taper_fraction : float
        Fraction of the array to taper.
    side : {'left', 'right'}
        End of ``y`` to taper.
    min_points : int, default=5
        Minimum number of points used for the taper.

    Returns
    -------
    np.ndarray
        Input array after in-place tapering.

    Raises
    ------
    ValueError
        If ``side`` is not recognized.
    """
    smooth_size = max(int(y.size * taper_fraction), min_points)

    if side == 'left':
        y[:smooth_size] *= half_hann(smooth_size, side=side)
    elif side == 'right':
        y[-smooth_size:] *= half_hann(smooth_size, side=side)
    else:
        raise ValueError("side must be 'left' or 'right'")

    return y

def taper_end_linear(y: np.ndarray, taper_fraction: float, side: Literal['left', 'right'], min_points: int = 5) -> np.ndarray:
    """Taper one end of an array in place with a linear ramp.

    Parameters
    ----------
    y : np.ndarray
        Array to modify.
    taper_fraction : float
        Fraction of the array to taper.
    side : {'left', 'right'}
        End of ``y`` to taper.
    min_points : int, default=5
        Minimum number of points used for the taper.

    Returns
    -------
    np.ndarray
        Input array after in-place tapering.

    Raises
    ------
    ValueError
        If ``side`` is not recognized.
    """
    smooth_size = max(int(y.size * taper_fraction), min_points)
    left_val = float(y[smooth_size-1])
    right_val = float(y[-smooth_size])

    if side == 'left':
        y[:smooth_size] = left_val * np.linspace(0, 1, num=smooth_size)
    elif side == 'right':
        y[-smooth_size:] = right_val * np.linspace(1, 0, num=smooth_size)
    else:
        raise ValueError("side must be 'left' or 'right'")

    return y

def smootherstep(x: np.ndarray) -> np.ndarray:
    """
    Quintic smoothstep.

    Maps x in [0, 1] to an S-curve in [0, 1] with zero first
    and second derivatives at both endpoints.
    """
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (x * (x * 6 - 15) + 10)

#endregion

################################
#region --- KERNEL FUNCTIONS ---
################################

def gamma_kernel(
        time: np.ndarray,
        amplitude: float,
        shape_k: float,
        tau_sec: float,
        ) -> np.ndarray:
    """Evaluate a peak-normalized gamma event kernel.

    Parameters
    ----------
    time : np.ndarray
        Time points, in seconds.
    amplitude : float
        Peak amplitude of the kernel.
    shape_k : float
        Gamma shape parameter.
    tau_sec : float
        Decay time constant, in seconds.

    Returns
    -------
    np.ndarray
        Kernel values evaluated at ``time``.
    """
    power = shape_k - 1.0
    scaled_time = time / (power * tau_sec)

    return (
        amplitude
        * np.power(scaled_time, power)
        * np.exp(power - time / tau_sec)
    )

def exp_decay_kernel(
        time: np.ndarray,
        amplitude: float,
        tau_sec: float,
        ) -> np.ndarray:
    """Evaluate an exponential decay kernel.

    Parameters
    ----------
    time : np.ndarray
        Time points, in seconds.
    amplitude : float
        Initial amplitude of the kernel.
    tau_sec : float
        Decay time constant, in seconds.

    Returns
    -------
    np.ndarray
        Kernel values evaluated at ``time``.
    """
    return amplitude * np.exp(-time / tau_sec)

def alpha_kernel(
        time: np.ndarray,
        amplitude: float,
        tau_sec: float,
        ) -> np.ndarray:
    """Evaluate a peak-normalized alpha-function kernel.

    Parameters
    ----------
    time : np.ndarray
        Time points, in seconds.
    amplitude : float
        Peak amplitude of the kernel.
    tau_sec : float
        Time constant, in seconds.

    Returns
    -------
    np.ndarray
        Kernel values evaluated at ``time``.
    """
    scaled_time = time / tau_sec
    return amplitude * scaled_time * np.exp(1.0 - scaled_time)

def diff_of_exp_kernel(
        time: np.ndarray,
        amplitude: float,
        tau_rise_sec: float,
        tau_decay_sec: float,
        ) -> np.ndarray:
    """Evaluate a peak-normalized difference-of-exponentials kernel.

    Parameters
    ----------
    time : np.ndarray
        Time points, in seconds.
    amplitude : float
        Peak amplitude of the kernel.
    tau_rise_sec : float
        Rise time constant, in seconds.
    tau_decay_sec : float
        Decay time constant, in seconds.

    Returns
    -------
    np.ndarray
        Kernel values evaluated at ``time``.
    """
    peak_time = (
        tau_rise_sec
        * tau_decay_sec
        / (tau_decay_sec - tau_rise_sec)
        * np.log(tau_decay_sec / tau_rise_sec)
    )

    peak_value = (
        np.exp(-peak_time / tau_decay_sec)
        - np.exp(-peak_time / tau_rise_sec)
    )

    curve = (
        np.exp(-time / tau_decay_sec)
        - np.exp(-time / tau_rise_sec)
    )

    return amplitude * curve / peak_value

def sum_of_exp_kernel(
        time: np.ndarray,
        amplitude: float,
        tau_fast_sec: float,
        tau_slow_sec: float,
        fast_weight: float,
        ) -> np.ndarray:
    """Evaluate a weighted sum of two exponential decays.

    Parameters
    ----------
    time : np.ndarray
        Time points, in seconds.
    amplitude : float
        Overall amplitude multiplier.
    tau_fast_sec : float
        Fast decay time constant, in seconds.
    tau_slow_sec : float
        Slow decay time constant, in seconds.
    fast_weight : float
        Weight applied to the fast exponential.

    Returns
    -------
    np.ndarray
        Kernel values evaluated at ``time``.
    """
    return amplitude * (
        fast_weight * np.exp(-time / tau_fast_sec)
        + (1.0 - fast_weight) * np.exp(-time / tau_slow_sec)
    )

def gaussian_kernel(
        time: np.ndarray,
        amplitude: float,
        center_sec: float,
        sigma_sec: float,
        ) -> np.ndarray:
    """Evaluate a Gaussian event kernel.

    Parameters
    ----------
    time : np.ndarray
        Time points, in seconds.
    amplitude : float
        Peak amplitude of the kernel.
    center_sec : float
        Center of the Gaussian, in seconds.
    sigma_sec : float
        Standard deviation of the Gaussian, in seconds.

    Returns
    -------
    np.ndarray
        Kernel values evaluated at ``time``.
    """
    return amplitude * np.exp(
        -0.5 * ((time - center_sec) / sigma_sec) ** 2
    )

def smooth_trapezoid_kernel(
        time: np.ndarray,
        amplitude: float,
        rise_duration: float,
        plateau_duration: float,
        decay_duration: float,
        ) -> np.ndarray:
    """
    Broad kernel with S-shaped rise, plateau, and S-shaped decay.

    Parameters
    ----------
    time : np.ndarray
        Time points, in seconds.
    amplitude : float
        Peak height of plateau.
    rise_duration : float
        Duration of the S-shaped increase, in seconds.
    plateau_duration : float
        Duration of the broad middle region, in seconds.
    decay_duration : float
        Duration of the S-shaped decrease, in seconds.

    Returns
    -------
    y : np.ndarray
        Kernel values evaluated at ``time``.
    """
    if rise_duration <= 0:
        raise ValueError("rise_duration must be positive.")
    if plateau_duration < 0:
        raise ValueError("plateau_duration must be non-negative.")
    if decay_duration <= 0:
        raise ValueError("decay_duration must be positive.")

    time = np.asarray(time)
    y = np.zeros_like(time, dtype=float)

    t0 = 0.0
    t1 = rise_duration
    t2 = rise_duration + plateau_duration
    t3 = rise_duration + plateau_duration + decay_duration

    # rise
    rise_mask = (time >= t0) & (time < t1)
    x_rise = (time[rise_mask] - t0) / rise_duration
    y[rise_mask] = amplitude * smootherstep(x_rise)

    # plateau
    plateau_mask = (time >= t1) & (time < t2)
    y[plateau_mask] = amplitude

    # decay
    decay_mask = (time >= t2) & (time <= t3)
    x_decay = (time[decay_mask] - t2) / decay_duration
    y[decay_mask] = amplitude * (1 - smootherstep(x_decay))

    return y

def domed_trapezoid_kernel(
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

    y = smooth_trapezoid_kernel(
        time=time,
        rise_duration=rise_duration,
        plateau_duration=plateau_duration,
        decay_duration=decay_duration,
        amplitude=1.0,
    )

    total_duration = rise_duration + plateau_duration + decay_duration
    x = np.clip(time / total_duration, 0.0, 1.0)

    # broad dome: 0 at edges, 1 at center
    dome = 4 * x * (1 - x)

    y = y * (1 + dome_strength * dome)

    # re-normalize to requested amplitude
    max_y = np.nanmax(y)
    if max_y > 0:
        y = amplitude * y / max_y

    return y

#endregion
