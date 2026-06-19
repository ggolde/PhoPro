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

#endregion
