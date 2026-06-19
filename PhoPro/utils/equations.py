import numpy as np

########################
#region --- MODELING ---
########################

def neg_exponential_3(x, a, b, c):
    """
    Negative single exponential for photobleaching curve fitting.
    Args:
        x (array-like): Independent variable (e.g., time).
        a (float): Amplitude of the exponential component.
        b (float): Decay rate of the exponential component.
        c (float): Constant offset term.
    Returns:
        np.ndarray: Evaluated exponential curve with the same shape as ``x``.
    """
    return a * np.exp(-b * x) + c

def neg_bi_exponential_5(x, a1, b1, a2, b2, c):
    """
    Negative bi-exponential for photobleaching curve fitting.
    Args:
        x (array-like): Independent variable (e.g., time).
        a1 (float): Amplitude of the fast exponential component.
        b1 (float): Decay rate of the fast component.
        a2 (float): Amplitude of the slow exponential component.
        b2 (float): Decay rate of the slow component.
        c (float): Constant offset term.
    Returns:
        np.ndarray: Evaluated bi-exponential curve with the same shape as ``x``.
    """
    return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * x) + c

#endregion

##########################
#region --- STAT FUNCS ---
##########################

def sem(arr, axis=0):
    """
    Compute the standard error of the mean along a given axis.
    Args:
        arr (array-like): Input data.
        axis (int, optional): Axis along which to compute the SEM. Defaults to 0.
    Returns:
        np.ndarray or float: Standard error of the mean along the specified axis.
    """
    n = arr.shape[axis]
    std = np.std(arr, axis=axis)
    return std / np.sqrt(n)

#endregion