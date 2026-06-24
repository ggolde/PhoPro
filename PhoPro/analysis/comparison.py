"""Statistical comparison utilities for trial-wise photometry data."""

from __future__ import annotations

from typing import Literal, Any, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from plotnine import ggplot
from scipy.stats import ttest_ind, false_discovery_control, PermutationMethod
from scipy.stats import t as t_dist, ttest_ind

from typing import TYPE_CHECKING

from ..utils import operations, graphing

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

if TYPE_CHECKING:
    from ..core.PhotometryData import PhotometryData

############################
#region --- RESULT CLASS ---
############################
@dataclass
class ClusterTestResult:
    """Result object for cluster-based group comparisons."""

    A: np.ndarray
    B: np.ndarray
    time: np.ndarray
    pvals: np.ndarray
    clusters: list[np.ndarray]
    cluster_pvals: list[np.ndarray]
    stat_observed: np.ndarray
    stat_treshold: float

    def __post_init__(self) -> None:
        """Calculate summary traces for plotting."""
        self.A_avg = np.nanmean(self.A, axis=0)
        self.A_std = np.nanstd(self.A, axis=0)

        self.B_avg = np.nanmean(self.B, axis=0)
        self.B_std = np.nanstd(self.B, axis=0)

    # --- query ---
    def any_significant(self, cutoff: float = 0.05) -> bool:
        """Check whether any point is significant.

        Parameters
        ----------
        cutoff : float, default=0.05
            P-value cutoff.

        Returns
        -------
        bool
            ``True`` if any p-value is below ``cutoff``.
        """
        return bool(np.any(self.pvals < cutoff))

    def significant_at(self, cutoff: float = 0.05) -> np.ndarray:
        """Return time points with significant p-values.

        Parameters
        ----------
        cutoff : float, default=0.05
            P-value cutoff.

        Returns
        -------
        np.ndarray
            Time values where ``pvals <= cutoff``.
        """
        sig_idx = np.nonzero(self.pvals <= cutoff)
        return self.time[sig_idx]

    # --- plotting ---
    def plot_comparison(
            self,
            p_cutoff: float = 0.05,
            A_label: str = 'A',
            B_label: str = 'B',
            title: str = '',
            y_lab: str = '',
            unsig_color: str = '#7a7a7a',
            cmap: str = 'viridis_r',
            line_kwargs: dict = {},
            point_kwargs: dict = {},
            ribbon_kwargs: dict = {},
            theme_kwargs: dict = {},
            ) -> ggplot:
        """Plot group means and cluster-test p-values.

        Parameters
        ----------
        p_cutoff : float, default=0.05
            P-value cutoff used for highlighting significance.
        A_label : str, default='A'
            Label for group A.
        B_label : str, default='B'
            Label for group B.
        title : str, default=''
            Plot title.
        y_lab : str, default=''
            Y-axis label.
        unsig_color : str, default='#7a7a7a'
            Color used for non-significant points.
        cmap : str, default='viridis_r'
            Colormap used for p-value coloring.
        line_kwargs : dict, default={}
            Keyword arguments forwarded to line geoms.
        point_kwargs : dict, default={}
            Keyword arguments forwarded to point geoms.
        ribbon_kwargs : dict, default={}
            Keyword arguments forwarded to ribbon geoms.
        theme_kwargs : dict, default={}
            Keyword arguments forwarded to the plot theme helper.

        Returns
        -------
        ggplot
            Plot object.
        """
        def _df_builder_helper(source: str, arr: np.ndarray, error: np.ndarray, pvalue: np.ndarray, time: np.ndarray) -> pd.DataFrame:
            """Build one long dataframe for a group summary."""
            index = np.arange(time.size)
            return pd.DataFrame(
                {
                    'time':time, 'source':source, 'value':arr, 'pvalue':pvalue,
                    'value_min': arr - error, 'value_max': arr + error
                },
                index=index
            )

        long_df = pd.concat(
            [
                _df_builder_helper(A_label, self.A_avg, self.A_std, self.pvals, self.time),
                _df_builder_helper(B_label, self.B_avg, self.B_std, self.pvals, self.time),
            ],
            axis=0,
            ignore_index=True,
        )

        p = graphing.plot_cluster_test(
            df=long_df,
            title=title,
            y_lab=y_lab,
            unsig_color=unsig_color,
            cmap=cmap,
            p_thr=p_cutoff,
            line_kwargs=line_kwargs,
            point_kwargs=point_kwargs,
            ribbon_kwargs=ribbon_kwargs,
            theme_kwargs=theme_kwargs
        )

        return p

#######################
#region --- UTILITY ---
#######################
def t_test(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Run pointwise Welch t-tests.

    Parameters
    ----------
    X : np.ndarray
        Group X observations with shape ``(n_observations, n_times)``.
    Y : np.ndarray
        Group Y observations with shape ``(n_observations, n_times)``.

    Returns
    -------
    stat : np.ndarray
        Pointwise t statistics.
    pval : np.ndarray
        Pointwise p-values.
    df_mean : float
        Mean degrees of freedom across time points.
    """
    res = ttest_ind(X, Y, axis=0, equal_var=False, nan_policy="omit")
    stat = res.statistic
    pval = res.pvalue
    df_mean = res.df.mean()
    return stat, pval, df_mean

def t_thresh(p_threshold, df) -> float:
    """Calculate a two-tailed t-statistic threshold.

    Parameters
    ----------
    p_threshold : float
        Pointwise p-value threshold.
    df : float
        Degrees of freedom.

    Returns
    -------
    float
        Absolute t-statistic threshold.
    """
    t_thres: float = t_dist.isf(p_threshold / 2.0, df=df)
    return t_thres

def pointwise_ttest(
        X: np.ndarray,
        Y: np.ndarray,
        permutations: int | None = None,
        multiple_test_correction: Literal['bh', 'by'] | None = None,
        random_state: int = 42,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run pointwise t-tests with optional permutations and FDR correction.

    Parameters
    ----------
    X : np.ndarray
        Group X observations with shape ``(n_observations, n_times)``.
    Y : np.ndarray
        Group Y observations with shape ``(n_observations, n_times)``.
    permutations : int or None, default=None
        Number of permutation resamples. If ``None``, analytic Welch t-tests
        are used.
    multiple_test_correction : {'bh', 'by'} or None, default=None
        Optional false-discovery-rate correction method.
    random_state : int, default=42
        Random seed used for permutation tests.

    Returns
    -------
    t_stat : np.ndarray
        Pointwise t statistics.
    pval : np.ndarray
        Pointwise p-values, optionally corrected.
    df : np.ndarray or None
        Degrees of freedom for analytic tests, or ``None`` for permutation
        tests.
    """
    if permutations is not None:
        method = PermutationMethod(n_resamples=permutations, random_state=random_state)
        res = ttest_ind(X, Y, axis=0, equal_var=False, method=method)
        df = None
    else:
        res = ttest_ind(X, Y, axis=0, equal_var=False)
        df = res.df

    t_stat = res.statistic
    pval = res.pvalue

    if multiple_test_correction is not None:
        pval = false_discovery_control(pval, method=multiple_test_correction)

    return t_stat, pval, df

#endregion

############################
#region --- CLUSTER TEST ---
############################
def _find_clusters(mask: np.ndarray) -> list[np.ndarray]:
    """Find contiguous true clusters in a one-dimensional mask."""
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, breaks)

def _calc_clusters_masses(obs: np.ndarray, thresh: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Calculate cluster indexes and absolute cluster masses."""
    clusters = _find_clusters(obs > thresh) + _find_clusters(obs < -thresh)
    masses = np.abs(np.asarray([obs[idx].sum() for idx in clusters], float))
    return clusters, masses

def cluster_permutation_test(
        A: PhotometryData | np.ndarray,
        B: PhotometryData | np.ndarray,
        stat_func: Callable = t_test,
        thres_func: Callable = t_thresh,
        n_perm: int = 2000,
        p_threshold: int = 0.05,
        n_jobs: int = 1,
        random_state: int | None = 42,
        ) -> ClusterTestResult:
    """Run a cluster-mass permutation test between two groups.

    Parameters
    ----------
    A : PhotometryData or np.ndarray
        Group A data. Arrays must have shape ``(n_observations, n_times)``.
    B : PhotometryData or np.ndarray
        Group B data. Arrays must have shape ``(n_observations, n_times)``.
    stat_func : Callable, default=t_test
        Function returning ``(statistic, pvalue, df)`` for two groups.
    thres_func : Callable, default=t_thresh
        Function returning the pointwise statistic threshold.
    n_perm : int, default=2000
        Number of permutations.
    p_threshold : int, default=0.05
        Pointwise p-value threshold used to form clusters.
    n_jobs : int, default=1
        Number of parallel jobs. Falls back to ``1`` if ``joblib`` is absent.
    random_state : int or None, default=42
        Random seed for permutations.

    Returns
    -------
    ClusterTestResult
        Cluster test result with p-values assigned to clustered time points.

    Raises
    ------
    ValueError
        If the two groups have different numbers of time points.
    """
    from ..core.PhotometryData import PhotometryData

    # validate inputs
    if isinstance(A, PhotometryData):
        time = A.ts
        A = A.X
    else:
        time = np.arange(A.shape[1], dtype=float)
    if isinstance(B, PhotometryData):
        B = B.X

    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.shape[1] != B.shape[1]:
        raise ValueError("X and Y must have the same number of timepoints (axis=1).")

    if (n_jobs != 1) and not _HAS_JOBLIB:
        print("joblib not found, falling back to n_jobs=1.")
        n_jobs = 1

    # observed clusters and test statistic
    rng = np.random.default_rng(random_state)
    stat_obs, p_obs, df_approx = stat_func(A, B)
    thresh = thres_func(p_threshold=p_threshold, df=df_approx)
    clusters_obs, masses_obs = _calc_clusters_masses(obs=stat_obs, thresh=thresh)

    # permutation setup
    Z = np.vstack([A, B])
    n_total = Z.shape[0]
    perm_indices = np.vstack([
        rng.permutation(n_total) for _ in range(n_perm)
    ])

    def _perm_worker(b):
        """Calculate the maximum cluster mass for one permutation."""
        idx = perm_indices[b]
        Xp = Z[idx[:A.shape[0]], :]
        Yp = Z[idx[A.shape[0]:], :]
        stat_perm, _, _ = stat_func(Xp, Yp)
        cluster_perm, masses_perm = _calc_clusters_masses(stat_perm, thresh)
        if len(cluster_perm) > 0:
            return np.max(masses_perm)
        else:
            return 0.0

    # run permutation
    if n_jobs == 1:
        null_max_masses = np.array([_perm_worker(b) for b in range(n_perm)], float)
    else:
        null_max_masses = np.array(
            Parallel(n_jobs=n_jobs)(delayed(_perm_worker)(b) for b in range(n_perm)), # type: ignore
            dtype=float,
        )

    # calculate p-values from null-dist
    if len(masses_obs) > 0:
        reshaped_null_masses = np.tile(np.abs(null_max_masses), (len(masses_obs), 1))
        cluster_pvals = (reshaped_null_masses >= np.abs(masses_obs)[:, np.newaxis]).sum(axis=1).astype(float) / n_perm
    else:
        cluster_pvals = np.array([])

    # package result
    pvals = np.full_like(stat_obs, np.nan, dtype=float)
    for i, clust_idx in enumerate(clusters_obs):
        pvals[clust_idx] = cluster_pvals[i]

    result = ClusterTestResult(
        A=A,
        B=B,
        time=time,
        pvals=pvals,
        clusters=clusters_obs,
        cluster_pvals=cluster_pvals,
        stat_observed=stat_obs,
        stat_treshold=float(thresh),
    )
    return result

#endregion

##########################
#region --- DEPTH TEST ---
##########################

def _calc_cluster_depth_stats(stats: np.ndarray, thresh: float) -> tuple[list[np.ndarray], np.ndarray]:
    """Calculate per-depth cluster statistics."""
    clusters = _find_clusters(stats > thresh) + _find_clusters(stats < -thresh)
    depth_stats = np.zeros(shape=(len(clusters), len(stats)))
    for i, idxs in enumerate(clusters):
        depth_stats[i, :len(idxs)] = np.abs(stats[idxs])
    return clusters, depth_stats

def _head_to_tail_stats(depth_stats: np.ndarray) -> np.ndarray:
    """Reverse nonzero cluster-depth statistics from head to tail."""
    for i in range(depth_stats.shape[0]):
        valid_depths = depth_stats[i, :] != 0
        depth_stats[i, valid_depths] = depth_stats[i, valid_depths][::-1]
    return depth_stats

def _troendle_step_down(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Apply Troendle step-down adjustment for cluster-depth statistics."""
    _obs = obs.copy()
    _null = null.copy()

    # find active and non-degenerate depths
    valid_depths = _obs != 0.0
    is_degenerate = (_null == 0).all(axis=0)
    is_valid = valid_depths & ~is_degenerate
    _padj = np.ones_like(_obs, dtype=float)

    # trim data for troendle algorithm
    obs = _obs[is_valid]
    null = _null[:, is_valid]
    padj = _padj[is_valid]
    is_active = np.ones_like(obs, dtype=bool)

    n_perm, n_points = null.shape

    order = np.argsort(-obs)
    for i in order:
        max_null = null[:, is_active].max(axis=1)
        padj[i] = np.sum(max_null >= obs[i]) / n_perm
        is_active[i] = False

    _padj[is_valid] = padj
    _padj[is_degenerate] = np.min(padj)
    # return padj's with valid depth, length equal to tested cluster's
    return _padj[valid_depths]

def cluster_depth_test(
        A: PhotometryData | np.ndarray,
        B: PhotometryData | np.ndarray,
        stat_func: Callable = t_test,
        thres_func: Callable = t_thresh,
        n_perm: int = 2000,
        p_threshold: int = 0.05,
        n_jobs: int = 1,
        random_state: int | None = 42,
        ) -> ClusterTestResult:
    """Run a cluster-depth permutation test between two groups.

    Parameters
    ----------
    A : PhotometryData or np.ndarray
        Group A data. Arrays must have shape ``(n_observations, n_times)``.
    B : PhotometryData or np.ndarray
        Group B data. Arrays must have shape ``(n_observations, n_times)``.
    stat_func : Callable, default=t_test
        Function returning ``(statistic, pvalue, df)`` for two groups.
    thres_func : Callable, default=t_thresh
        Function returning the pointwise statistic threshold.
    n_perm : int, default=2000
        Number of permutations.
    p_threshold : int, default=0.05
        Pointwise p-value threshold used to form clusters.
    n_jobs : int, default=1
        Number of parallel jobs. Falls back to ``1`` if ``joblib`` is absent.
    random_state : int or None, default=42
        Random seed for permutations.

    Returns
    -------
    ClusterTestResult
        Cluster-depth test result with adjusted p-values assigned to clustered
        time points.

    Raises
    ------
    ValueError
        If the two groups have different numbers of time points.
    """
    from ..core.PhotometryData import PhotometryData

    # validate inputs
    if isinstance(A, PhotometryData):
        time = A.ts
        A = A.X
    else:
        time = np.arange(A.shape[1], dtype=float)
    if isinstance(B, PhotometryData):
        B = B.X

    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.shape[1] != B.shape[1]:
        raise ValueError("X and Y must have the same number of timepoints (axis=1).")
    if (n_jobs != 1) and not _HAS_JOBLIB:
        print("joblib not found, falling back to n_jobs=1.")
        n_jobs = 1

    # observed clusters and test statistic
    rng = np.random.default_rng(random_state)
    stat_obs, p_obs, df_approx = stat_func(A, B)
    thresh = thres_func(p_threshold=p_threshold, df=df_approx)
    clusters_obs, head_obs = _calc_cluster_depth_stats(stats=stat_obs, thresh=thresh)
    tail_obs = _head_to_tail_stats(head_obs)

    # set up permutation
    Z = np.vstack([A, B])
    n_total = Z.shape[0]
    perm_indices = np.vstack([
        rng.permutation(n_total) for _ in range(n_perm)
    ])

    def _perm_worker(b):
        """Calculate maximum depth statistics for one permutation."""
        idx = perm_indices[b]
        Xp = Z[idx[:A.shape[0]], :]
        Yp = Z[idx[A.shape[0]:], :]
        stat_perm, _, _ = stat_func(Xp, Yp)
        cluster_perm, head_perm = _calc_cluster_depth_stats(stat_perm, thresh)
        if len(cluster_perm) > 0:
            return np.max(head_perm, axis=0)
        else:
            return np.zeros(head_perm.shape[1], dtype=float)

    # generate null distributions
    if n_jobs == 1:
        null_head = np.array([_perm_worker(b) for b in range(n_perm)], float)
    else:
        null_head = np.array(
            Parallel(n_jobs=n_jobs)(delayed(_perm_worker)(b) for b in range(n_perm)), # type: ignore
            dtype=float,
        )
    null_tail = _head_to_tail_stats(null_head)

    # calculate padj from null distribution
    cluster_pvals = []
    pvals = np.full_like(stat_obs, np.nan, dtype=float)
    for i, clust_idx in enumerate(clusters_obs):
        head_padj = _troendle_step_down(head_obs[i, :], null=null_head)
        tail_padj = _troendle_step_down(tail_obs[i, :], null=null_tail)
        padj_i = np.vstack([head_padj, tail_padj]).max(axis=0)
        cluster_pvals.append(padj_i)
        pvals[clust_idx] = padj_i

    # package results
    result = ClusterTestResult(
        A=A,
        B=B,
        time=time,
        pvals=pvals,
        clusters=clusters_obs,
        cluster_pvals=cluster_pvals,
        stat_observed=stat_obs,
        stat_treshold=float(thresh),
    )
    return result

#endregion
