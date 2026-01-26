from scipy.stats import ttest_ind, false_discovery_control, PermutationMethod
from scipy.stats import t as t_dist, ttest_ind
from typing import Literal, Tuple, List, Dict, Any

import numpy as np

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

def t_test(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    res = ttest_ind(X, Y, axis=0, equal_var=False, nan_policy="omit")
    stat = res.statistic
    pval = res.pvalue
    df_mean = res.df.mean()
    return stat, pval, df_mean

def t_thresh(p_threshold, df) -> float:
    t_thres = t_dist.isf(p_threshold / 2.0, df=df)
    return t_thres

def pointwise_ttest(
        X: np.ndarray, 
        Y: np.ndarray,
        permutations: int | None = None,
        multiple_test_correction: Literal['bh', 'by'] | None = None,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if permutations is not None:
        method = PermutationMethod(n_resamples=permutations, random_state=random_state)
        t_stat, pval = ttest_ind(X, Y, axis=0, equal_var=False, method=method)
        df = None
    else:
        res = ttest_ind(X, Y, axis=0, equal_var=False)
        t_stat = res.statistic
        pval = res.pvalue
        df = res.df
    if multiple_test_correction is not None:
        pval = false_discovery_control(pval, method=multiple_test_correction)
    return t_stat, pval, df

def _find_clusters(mask: np.ndarray) -> List[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, breaks)

def _calc_clusters_masses(obs: np.ndarray, thresh: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:    
    clusters = _find_clusters(obs > thresh) + _find_clusters(obs < -thresh)
    masses = np.abs(np.asarray([obs[idx].sum() for idx in clusters], float))
    return clusters, masses

def cluster_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    stat_func: callable = t_test,
    thres_func: callable = t_thresh,
    n_perm: int = 2000,
    p_threshold: int = 0.05,
    n_jobs: int = 1,
    random_state: int | None = 42,
    ) -> Dict[str, Any]:
    # validate inputs
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of timepoints (axis=1).")

    if (n_jobs != 1) and not _HAS_JOBLIB:
        print("joblib not found, falling back to n_jobs=1.")
        n_jobs = 1

    # observed clusters and test statistic
    rng = np.random.default_rng(random_state)
    stat_obs, p_obs, df_approx = stat_func(X, Y)
    thresh = thres_func(p_threshold=p_threshold, df=df_approx)
    clusters_obs, masses_obs = _calc_clusters_masses(obs=stat_obs, thresh=thresh)

    # permutation setup
    Z = np.vstack([X, Y])
    n_total = Z.shape[0]
    perm_indices = np.vstack([
        rng.permutation(n_total) for _ in range(n_perm)
    ])

    def _perm_worker(b):
        idx = perm_indices[b]
        Xp = Z[idx[:X.shape[0]], :]
        Yp = Z[idx[X.shape[0]:], :]
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
            Parallel(n_jobs=n_jobs)(delayed(_perm_worker)(b) for b in range(n_perm)),
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
    for i in range(clusters_obs):
        pvals[clusters_obs[i]] = cluster_pvals[i]
    
    result = {
        "pvals": pvals,
        "clusters": clusters_obs,
        "cluster_pvals": cluster_pvals,
        "stat_obs": stat_obs,
        "stat_threshold": float(thresh),
    }
    return result

def _calc_cluster_depth_stats(stats: np.ndarray, thresh: float) -> Tuple[List[np.ndarray], np.ndarray]:
    clusters = _find_clusters(stats > thresh) + _find_clusters(stats < -thresh)
    depth_stats = np.zeros(shape=(len(clusters), len(stats)))
    for i, idxs in enumerate(clusters):
        depth_stats[i, :len(idxs)] = np.abs(stats[idxs])
    return clusters, depth_stats

def _head_to_tail_stats(depth_stats: np.ndarray) -> np.ndarray:
    for i in range(depth_stats.shape[0]):
        valid_depths = depth_stats[i, :] != 0
        depth_stats[i, valid_depths] = depth_stats[i, valid_depths][::-1]
    return depth_stats

def _troendle_step_down(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
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
    X: np.ndarray,
    Y: np.ndarray,
    stat_func: callable = t_test,
    thres_func: callable = t_thresh,
    n_perm: int = 2000,
    p_threshold: int = 0.05,
    n_jobs: int = 1,
    random_state: int | None = 42,
    ) -> Dict[str, Any]:
    # validate inputs
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of timepoints (axis=1).")
    if (n_jobs != 1) and not _HAS_JOBLIB:
        print("joblib not found, falling back to n_jobs=1.")
        n_jobs = 1

    # observed clusters and test statistic
    rng = np.random.default_rng(random_state)
    stat_obs, p_obs, df_approx = stat_func(X, Y)
    thresh = thres_func(p_threshold=p_threshold, df=df_approx)
    clusters_obs, head_obs = _calc_cluster_depth_stats(stats=stat_obs, thresh=thresh)
    tail_obs = _head_to_tail_stats(head_obs)

    # set up permutation
    Z = np.vstack([X, Y])
    n_total = Z.shape[0]
    perm_indices = np.vstack([
        rng.permutation(n_total) for _ in range(n_perm)
    ])

    def _perm_worker(b):
        idx = perm_indices[b]
        Xp = Z[idx[:X.shape[0]], :]
        Yp = Z[idx[X.shape[0]:], :]
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
            Parallel(n_jobs=n_jobs)(delayed(_perm_worker)(b) for b in range(n_perm)),
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
    result = {
        "pvals": pvals,
        "clusters": clusters_obs,
        "cluster_pvals": cluster_pvals,
        "stat_obs": stat_obs,
        "t_threshold": float(thresh),
    }
    return result