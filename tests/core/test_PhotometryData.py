import numpy as np
import pytest

from pyFiberPhotometry.core.PhotometeryData import PhotometryData


# --- constructors and basic shape contracts ---

def test_from_arrays_fixture_shapes_and_metadata(photometry_data):
    assert isinstance(photometry_data, PhotometryData)
    assert photometry_data.n_trials == 3
    assert photometry_data.n_times == 20
    assert photometry_data.X.shape == (3, 20)
    assert photometry_data.obs.index.tolist() == ["0", "1", "2"]
    assert photometry_data.get_layer("layer").shape == (3, 20)
    assert photometry_data.uns["frequency"] == 1.0
    assert np.isclose(photometry_data.freq, 1.0)


def test_from_arrays_rejects_mismatched_layer_shape(photometry_data):
    with pytest.raises(AssertionError):
        PhotometryData.from_arrays(
            obs=photometry_data.obs,
            data=photometry_data.X,
            time_points=photometry_data.ts,
            layers={"bad": np.ones((photometry_data.n_trials, photometry_data.n_times - 1))},
        )


# --- row, metadata, and object operations ---

def test_downsample_reduces_time_dimension_layers_and_frequency(photometry_data):
    out = photometry_data.downsample(2)

    assert out.n_trials == photometry_data.n_trials
    assert out.n_times == 10
    assert out.X.shape == (3, 10)
    assert out.get_layer("layer").shape == (3, 10)
    assert out.obs["animal"].tolist() == photometry_data.obs["animal"].tolist()
    assert np.isclose(out.uns["frequency"], 0.5)
    assert np.isclose(out.freq, 0.5, atol=0.1)


def test_combine_obj_returns_combined_object(photometry_data):
    out = photometry_data.combine_obj(photometry_data, inplace=False)

    assert isinstance(out, PhotometryData)
    assert out.n_trials == 6
    assert out.n_times == photometry_data.n_times
    assert out.obs["animal"].tolist() == ["a", "a", "b", "a", "a", "b"]


def test_combine_obj_inplace_updates_object(photometry_data):
    other = photometry_data.copy()

    result = photometry_data.combine_obj(other, inplace=True)

    assert result is None
    assert photometry_data.n_trials == 6
    assert photometry_data.n_times == other.n_times


def test_filter_rows_returns_subset_with_layers_and_metadata(photometry_data):
    out = photometry_data.filter_rows(np.array([True, False, True]), inplace=False)

    assert isinstance(out, PhotometryData)
    assert out.n_trials == 2
    assert out.n_times == photometry_data.n_times
    assert out.get_layer("layer").shape == (2, photometry_data.n_times)
    assert out.uns["frequency"] == photometry_data.uns["frequency"]


def test_filter_rows_inplace_updates_object(photometry_data):
    result = photometry_data.filter_rows(np.array([True, False, True]), inplace=True)

    assert result is None
    assert photometry_data.n_trials == 2
    assert photometry_data.obs["animal"].tolist() == ["a", "b"]


def test_add_obs_columns_and_drop_obs_columns(photometry_data):
    photometry_data.add_obs_columns({"session": ["s1", "s1", "s2"]})

    assert "session" in photometry_data.obs.columns
    assert photometry_data.obs["session"].tolist() == ["s1", "s1", "s2"]

    photometry_data.drop_obs_columns(["session"])
    assert "session" not in photometry_data.obs.columns


def test_add_metadata_updates_uns(photometry_data):
    photometry_data.add_metadata({"lab": "test_lab", "batch": 2}, keys=["lab"])

    assert photometry_data.uns["lab"] == "test_lab"
    assert "batch" not in photometry_data.uns


def test_get_text_value_counts_summarizes_column(photometry_data):
    text = photometry_data.get_text_value_counts("animal")

    assert "a: 2" in text
    assert "b: 1" in text


# --- aggregation ---

def test_collapse_groups_trials_and_adds_metric_layers(photometry_data):
    out = photometry_data.collapse(
        group_on=["animal"],
        data_cols=["score"],
        collapse_cols=["condition"],
    )

    assert isinstance(out, PhotometryData)
    assert out.n_trials == 2
    assert "n" in out.obs.columns
    assert "score" in out.obs.columns
    assert "score_std" in out.obs.columns
    assert "condition" in out.obs.columns
    assert "std" in out.adata.layers
    assert sorted(out.obs["animal"].tolist()) == ["a", "b"]


def test_collapse_without_groups_collapses_all_trials(photometry_data):
    out = photometry_data.collapse(
        group_on=None,
        data_cols=["score"],
        count_col="n",
    )

    assert out.n_trials == 1
    assert out.obs.loc["0", "n"] == photometry_data.n_trials
    assert np.isclose(out.obs.loc["0", "score"], np.mean([1.0, 3.0, 5.0]))


# --- windowing ---

def test_window_returns_centered_timebase(photometry_data):
    out = photometry_data.window(
        centers=10.0,
        bounds=(-3.0, 4.0),
    )

    assert isinstance(out, PhotometryData)
    assert out.n_trials == photometry_data.n_trials
    assert np.isclose(out.ts[0], -3.0)
    assert np.isclose(out.ts[-1], 3.0)
    assert out.X.shape[1] == 7


def test_window_accepts_centers_from_obs_column_and_recenters_events(photometry_data):
    photometry_data.add_obs_columns({
        "center": [5.0, 10.0, 15.0],
        "event_time": [6.0, 11.0, 16.0],
    })

    out = photometry_data.window(
        centers="center",
        bounds=(-2.0, 3.0),
        event_cols=["event_time"],
    )

    assert out.n_trials == photometry_data.n_trials
    assert out.n_times == 5
    assert np.allclose(out.obs["event_time"].to_numpy(dtype=float), 1.0)
    assert out.get_layer("layer").shape == out.X.shape


def test_window_drop_invalid_masks_obs_data_and_layers(photometry_data):
    photometry_data.add_obs_columns({
        "center": [1.0, 10.0, 18.0],
        "event_time": [1.0, 10.0, 18.0],
    })

    out = photometry_data.window(
        centers="center",
        bounds=(-2.0, 2.0),
        event_cols=["event_time"],
        invalid_window_policy="drop",
    )

    assert out.n_trials == 2
    assert out.obs["animal"].tolist() == ["a", "b"]
    assert np.allclose(out.obs["event_time"].to_numpy(dtype=float), 0.0)
    assert out.X.shape == (2, 4)
    assert out.get_layer("layer").shape == (2, 4)


def test_window_invalid_policy_error_raises(photometry_data):
    with pytest.raises(ValueError, match="Invalid trial windows"):
        photometry_data.window(
            centers=np.asarray([1.0, 10.0, 18.0]),
            bounds=(-2.0, 2.0),
            invalid_window_policy="error",
        )


# --- numerical summaries and dataframe exports ---

def test_area_under_curve_matches_numpy_trapezoid(photometry_data):
    auc = photometry_data.area_under_curve()
    expected = np.trapezoid(y=photometry_data.X, x=photometry_data.ts, axis=1)

    assert np.allclose(auc, expected)


def test_area_under_curve_applies_transformation(photometry_data):
    auc = photometry_data.area_under_curve(transformation=lambda x: x * 2)
    expected = np.trapezoid(y=photometry_data.X * 2, x=photometry_data.ts, axis=1)

    assert np.allclose(auc, expected)


def test_trials_to_long_df_has_expected_columns_and_size(photometry_data):
    df = photometry_data.trials_to_long_df(
        err_layer="layer",
        obs_cols=["animal", "condition"],
        downsample=None,
    )

    assert {"trial_id", "time_idx", "time", "signal", "animal", "condition", "layer"} <= set(df.columns)
    assert len(df) == photometry_data.n_trials * photometry_data.n_times
    assert df["time_idx"].min() == 0
    assert df["time_idx"].max() == photometry_data.n_times - 1


def test_trials_to_long_df_default_downsamples(photometry_data):
    df = photometry_data.trials_to_long_df(obs_cols=["animal"])

    assert len(df) == photometry_data.n_trials * 2
    assert df["time_idx"].max() == 1


def test_trials_to_wide_df_has_obs_and_signal_columns(photometry_data):
    df = photometry_data.trials_to_wide_df(
        obs_cols=["animal"],
        signal_prefix="sig",
        downsample=2,
    )

    signal_cols = [col for col in df.columns if col.startswith("sig.")]
    assert df["animal"].tolist() == photometry_data.obs["animal"].tolist()
    assert len(signal_cols) == 10
    assert len(df) == photometry_data.n_trials
