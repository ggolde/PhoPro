import numpy as np
import pandas as pd
import pytest

from PhoPro.analysis import (
    ArtifactResult,
    ClusterTestResult,
    FMMResult,
    ODS_Detector,
    PeakResult,
    RollingThresholdDetector,
    Spline_Corrector,
    StaticThresholdDetector,
    cluster_depth_test,
    cluster_permutation_test,
)
from PhoPro.analysis import artifact, comparison, peaks


# --- artifacts ---

def test_artifact_interval_helpers_roundtrip_boolean_mask():
    mask = np.asarray([False, True, True, False, True, False])

    intervals = artifact.boolean_mask_to_intervals(mask)
    flat = artifact.intervals_to_flat_idx(intervals)
    rebuilt = artifact.intervals_to_bool_mask(mask, intervals)

    assert np.array_equal(intervals, np.asarray([[1, 3], [4, 5]]))
    assert np.array_equal(flat, np.asarray([1, 2, 4]))
    assert np.array_equal(rebuilt, mask)
    assert np.isclose(artifact.median_abs_deviation(np.asarray([1.0, 1.0, 3.0])), 0.0)


def test_artifact_result_defaults_type_and_validates_required_columns():
    unlabeled = ArtifactResult(pd.DataFrame({"start_idx": [2], "stop_idx": [4]}))

    assert unlabeled.types == ["unlabeled"]
    assert np.array_equal(unlabeled.intervals, np.asarray([[2, 4]]))

    with pytest.raises(ValueError, match="Missing required"):
        ArtifactResult(pd.DataFrame({"start_idx": [2]}))


def test_artifact_result_groups_intervals_metrics_and_score():
    result = ArtifactResult(pd.DataFrame({
        "type": ["spike", "jump"],
        "start_idx": [4, 8],
        "stop_idx": [6, 10],
    }))
    signal = np.asarray([1.0, 1.0, 1.0, 1.0, 5.0, 4.0, 1.0, 1.0, 3.0, 4.0, 2.0, 2.0])
    time = np.arange(signal.size, dtype=float)

    result.calculate_metrics(signal=signal, time=time, buffer=2)

    assert result.types == ["spike", "jump"]
    assert set(result.groups) == {"spike", "jump"}
    assert np.array_equal(result.group_intervals["spike"], np.asarray([[4, 6]]))
    assert {"amplitude", "duration", "baseline_change", "start_time", "stop_time"} <= set(result.df.columns)
    assert np.isfinite(result.score)


def test_ods_detector_detects_outlying_derivative_and_reference_filter():
    signal = np.zeros(40, dtype=float)
    signal[20] = 10.0
    time = np.arange(signal.size, dtype=float)
    detector = ODS_Detector(
        score_func=lambda delta: delta,
        score_threshold=5.0,
        jump_score_threshold=100.0,
        reference_cor_cutoff=-1.0,
        expand_sec=(0.0, 1.0),
        buffer_sec=2.0,
        n_chunks=4,
    )

    result = detector.detect(signal=signal, reference=signal.copy(), time=time)

    assert isinstance(result, ArtifactResult)
    assert result.df.shape[0] >= 1
    assert {"type", "jump_score", "reference_cor"} <= set(result.df.columns)
    assert set(result.df["type"]) <= {"spike", "jump"}


def test_spline_corrector_interpolates_spike_artifact():
    time = np.arange(12, dtype=float)
    baseline = time.copy()
    signal = baseline.copy()
    signal[5:7] = 100.0
    artifacts = ArtifactResult(pd.DataFrame({
        "type": ["spike"],
        "start_idx": [5],
        "stop_idx": [7],
    }))
    corrector = Spline_Corrector(
        anchor_sec=(2.0, 2.0),
        correct_spikes=True,
        correct_jumps=False,
    )

    corrected = corrector.correct(signal=signal, time=time, artifacts=artifacts)

    assert corrected.shape == signal.shape
    assert np.allclose(corrected[:5], baseline[:5])
    assert np.allclose(corrected[7:], baseline[7:])
    assert np.allclose(corrected[5:7], baseline[5:7], atol=1e-6)


# --- comparison ---

def _cluster_stat(A, B):
    return A.mean(axis=0) - B.mean(axis=0), np.ones(A.shape[1]), 10.0


def _cluster_thresh(p_threshold, df):
    return 0.5


def _cluster_arrays():
    A = np.zeros((4, 6), dtype=float)
    B = np.zeros((4, 6), dtype=float)
    A[:, 2:4] = 2.0
    return A, B


def test_cluster_test_result_summaries_queries_and_plot():
    A = np.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    B = np.asarray([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
    result = ClusterTestResult(
        A=A,
        B=B,
        time=np.asarray([0.0, 1.0, 2.0]),
        pvals=np.asarray([0.1, 0.01, np.nan]),
        clusters=[np.asarray([1])],
        cluster_pvals=[np.asarray([0.01])],
        stat_observed=np.asarray([0.0, 3.0, 0.0]),
        stat_treshold=2.0,
    )

    assert np.allclose(result.A_avg, [2.0, 3.0, 4.0])
    assert np.allclose(result.B_avg, [3.0, 3.0, 3.0])
    assert result.any_significant(cutoff=0.05)
    assert np.array_equal(result.significant_at(cutoff=0.05), np.asarray([1.0]))
    assert hasattr(result.plot_comparison(), "draw")


def test_cluster_permutation_test_returns_cluster_result_and_validates_shapes():
    A, B = _cluster_arrays()

    result = cluster_permutation_test(
        A,
        B,
        stat_func=_cluster_stat,
        thres_func=_cluster_thresh,
        n_perm=20,
        random_state=1,
    )

    assert isinstance(result, ClusterTestResult)
    assert [cluster.tolist() for cluster in result.clusters] == [[2, 3]]
    assert result.pvals.shape == (A.shape[1],)
    assert np.isfinite(result.pvals[2:4]).all()

    with pytest.raises(ValueError, match="same number of timepoints"):
        cluster_permutation_test(A, B[:, :-1], stat_func=_cluster_stat, thres_func=_cluster_thresh)


def test_cluster_depth_test_returns_pointwise_cluster_pvalues():
    A, B = _cluster_arrays()

    result = cluster_depth_test(
        A,
        B,
        stat_func=_cluster_stat,
        thres_func=_cluster_thresh,
        n_perm=20,
        random_state=1,
    )

    assert isinstance(result, ClusterTestResult)
    assert [cluster.tolist() for cluster in result.clusters] == [[2, 3]]
    assert len(result.cluster_pvals) == 1
    assert result.cluster_pvals[0].shape == (2,)
    assert np.isfinite(result.pvals[2:4]).all()


def test_pointwise_ttest_and_cluster_helpers_smoke():
    X = np.asarray([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]])
    Y = np.asarray([[1.5, 2.5, 3.5], [1.6, 2.6, 3.6]])

    stat, pval, df = comparison.pointwise_ttest(X, Y, multiple_test_correction="bh")
    clusters = comparison._find_clusters(np.asarray([False, True, True, False, True]))

    assert stat.shape == pval.shape == df.shape == (3,)
    assert [cluster.tolist() for cluster in clusters] == [[1, 2], [4]]


# --- peaks ---

def _peak_result() -> PeakResult:
    return PeakResult(pd.DataFrame({
        "trial_idx": [1, 0, 0, 1],
        "start_idx": [5, 4, 1, 2],
        "stop_idx": [6, 4, 2, 3],
        "peak_idx": [5, 4, 1, 2],
        "peak_time": [5.0, 4.0, 1.0, 2.0],
        "height": [1.0, 4.0, 2.0, 3.0],
        "direction": ["positive", "positive", "negative", "negative"],
    }))


def test_peak_result_properties_filtering_and_csv_roundtrip(tmp_path):
    result = _peak_result()
    path = tmp_path / "peaks.csv"

    assert result.n_peaks == 4
    assert not result.empty
    assert not result.one_peak_per_trial
    assert result.df[["trial_idx", "start_idx"]].to_numpy().tolist() == [[0, 1], [0, 4], [1, 2], [1, 5]]
    assert result.filter("height", "lowest", None).df["height"].tolist() == [2.0, 1.0]
    assert result.filter("height", "highest", None).df["height"].tolist() == [4.0, 3.0]
    assert result.filter("peak_time", "closest", 3.0).df["peak_time"].tolist() == [4.0, 2.0]
    assert result.filter("height", "between", (2.0, 3.0)).df["height"].tolist() == [2.0, 3.0]
    assert result.filter("direction", "equals", "positive").df["direction"].tolist() == ["positive", "positive"]

    result.to_csv(path)
    loaded = PeakResult.from_csv(path)
    assert loaded.n_peaks == result.n_peaks


def test_peak_result_validates_filter_inputs():
    result = _peak_result()

    with pytest.raises(ValueError, match="trial_idx"):
        PeakResult(pd.DataFrame({"height": [1.0]}))
    with pytest.raises(KeyError, match="missing"):
        result.filter("missing", "highest", None)
    with pytest.raises(ValueError, match="tuple"):
        result.filter("height", "between", 2.0)
    with pytest.raises(ValueError, match="<="):
        result.filter("height", "between", (3.0, 2.0))
    with pytest.raises(ValueError, match="scalar"):
        result.filter("height", "closest", None)
    with pytest.raises(ValueError, match="not recognized"):
        result.filter("height", "unknown", None)


def test_static_threshold_detector_detects_directions_and_applies_size_filters():
    signals = np.asarray([
        [0.0, 2.0, 2.0, 0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, -2.0, -2.0, 0.0, 0.0, 0.0],
    ])
    detector = StaticThresholdDetector(
        center_method="zeros",
        scale_method="ones",
        test_magnitude=0.5,
    )

    both = detector.detect(
        signals=signals,
        time=np.arange(signals.shape[1], dtype=float),
        frequency=1.0,
        direction="both",
        detailed=True,
    )
    merged = detector.detect(
        signals=signals[:1],
        time=np.arange(signals.shape[1], dtype=float),
        frequency=1.0,
        min_distance_sec=2.0,
        min_duration_sec=4.0,
        max_duration_sec=5.0,
        direction="positive",
    )

    assert both.n_peaks == 3
    assert set(both.df["direction"]) == {"positive", "negative"}
    assert {"left_prominence", "right_prominence"} <= set(both.df.columns)
    assert merged.n_peaks == 1
    assert merged.df.iloc[0]["start_idx"] == 1
    assert merged.df.iloc[0]["stop_idx"] == 5


def test_rolling_threshold_detector_detects_with_rowwise_windows():
    signals = np.zeros((2, 12), dtype=float)
    signals[0, 5:7] = 3.0
    signals[1, 8:10] = -3.0
    detector = RollingThresholdDetector(
        window_width_sec=3.0,
        center_method="zeros",
        scale_method="ones",
        test_magnitude=0.5,
    )

    result = detector.detect(
        signals=signals,
        time=np.arange(signals.shape[1], dtype=float),
        frequency=1.0,
        direction="both",
    )

    assert result.n_peaks == 2
    assert set(result.df["direction"]) == {"positive", "negative"}


def test_threshold_detectors_validate_methods_and_direction():
    with pytest.raises(ValueError, match="Center method"):
        StaticThresholdDetector(center_method="bad")
    with pytest.raises(ValueError, match="Scale method"):
        StaticThresholdDetector(scale_method="bad")

    detector = StaticThresholdDetector(center_method="zeros", scale_method="ones")
    with pytest.raises(ValueError, match="Peak direction"):
        detector.detect(
            signals=np.zeros((1, 5)),
            time=np.arange(5, dtype=float),
            frequency=1.0,
            direction="bad",
        )


# --- FMM result wrapper ---

def _fmm_result(include_aic: bool = True, include_time: bool = True) -> FMMResult:
    columns = []
    rows = []
    if include_time:
        columns.append(("time", "time"))
    columns.extend([
        ("Intercept", "beta"),
        ("Intercept", "lower"),
        ("Intercept", "upper"),
        ("condition", "beta"),
        ("condition", "lower"),
        ("condition", "upper"),
    ])
    if include_aic:
        columns.append(("AIC", "AIC"))

    for i in range(2):
        row = []
        if include_time:
            row.append(float(i))
        row.extend([1.0 + i, 0.5 + i, 1.5 + i, -0.5 - i, -1.0 - i, 0.0 - i])
        if include_aic:
            row.append(10.0 + i)
        rows.append(row)

    df = pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns))
    return FMMResult(df=df, formula="photometry ~ condition")


def test_fmm_result_accessors_long_export_plot_and_csv_roundtrip(tmp_path):
    result = _fmm_result()
    path = tmp_path / "fmm.csv"

    assert result.column_groups == ["time", "Intercept", "condition", "AIC"]
    assert result.terms == ["Intercept", "condition"]
    assert result.coefficients.shape[1] == 6
    assert list(result.term("Intercept").columns) == ["time", "beta", "lower", "upper"]
    assert list(result.term("condition", include_time=False).columns) == ["beta", "lower", "upper"]
    assert list(result.stat("beta").columns) == ["Intercept", "condition"]
    assert "AIC" in result.aic.columns

    long_df = result.to_long(only_terms=["condition"], stat_map={"value": "beta", "lower": "lower"})
    assert set(long_df["term"]) == {"condition"}
    assert {"time_idx", "time", "term", "value", "lower"} <= set(long_df.columns)
    assert hasattr(result.plot(only_terms=["Intercept"]), "draw")

    result.to_csv(path)
    loaded = FMMResult.from_csv(path, formula=result.formula)
    assert loaded.formula == result.formula
    assert loaded.df.shape == result.df.shape


def test_fmm_result_validates_columns_and_missing_groups():
    with pytest.raises(ValueError, match="MultiIndex"):
        FMMResult(pd.DataFrame({"time": [0.0]}))

    result = _fmm_result(include_aic=False)
    with pytest.raises(KeyError, match="AIC"):
        _ = result.aic
    with pytest.raises(KeyError, match="Unknown"):
        result.term("missing")
    with pytest.raises(KeyError, match="time"):
        _ = _fmm_result(include_time=False).time
