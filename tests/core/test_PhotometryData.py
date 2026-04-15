import numpy as np

from pyFiberPhotometry.core.PhotometeryData import PhotometryData


def test_from_arrays_fixture_shapes_and_metadata(photometry_data):
    assert isinstance(photometry_data, PhotometryData)
    assert photometry_data.n_trials == 3
    assert photometry_data.n_times == 20
    assert photometry_data.X.shape == (3, 20)
    assert photometry_data.get_layer("layer").shape == (3, 20)
    assert np.isclose(photometry_data.freq, 1.0)


def test_downsample_reduces_time_dimension_and_frequency(photometry_data):
    out = photometry_data.downsample(2)

    assert out.n_trials == photometry_data.n_trials
    assert out.n_times == 10
    assert out.X.shape == (3, 10)
    assert out.get_layer("layer").shape == (3, 10)
    assert np.isclose(out.freq, 0.5, atol=0.1)


def test_combine_obj_returns_combined_object(photometry_data):
    out = photometry_data.combine_obj(photometry_data, inplace=False)

    assert isinstance(out, PhotometryData)
    assert out.n_trials == 6
    assert out.n_times == photometry_data.n_times


def test_filter_rows_returns_subset(photometry_data):
    out = photometry_data.filter_rows(np.array([True, False, True]), inplace=False)

    assert isinstance(out, PhotometryData)
    assert out.n_trials == 2
    assert out.n_times == photometry_data.n_times


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


def test_window_returns_centered_timebase(photometry_data):
    out = photometry_data.window(
        centers=10.0,
        bounds=(-3.0, 4.0),
        series=photometry_data.ts,
    )

    assert isinstance(out, PhotometryData)
    assert out.n_trials == photometry_data.n_trials
    assert np.isclose(out.ts[0], -3.0)
    assert np.isclose(out.ts[-1], 3.0)
    assert out.X.shape[1] == 7


def test_trials_to_long_df_has_expected_columns_and_size(photometry_data):
    df = photometry_data.trials_to_long_df(
        err_layer="layer",
        obs_cols=["animal", "condition"],
    )

    assert {"trial_id", "time_idx", "time", "signal", "animal", "condition", "layer"} <= set(df.columns)
    assert len(df) == photometry_data.n_trials * photometry_data.n_times
    assert df["time_idx"].min() == 0
    assert df["time_idx"].max() == photometry_data.n_times - 1
