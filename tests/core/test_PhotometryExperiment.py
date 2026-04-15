import numpy as np
import pytest

from pyFiberPhotometry.core.PhotometeryData import PhotometryData


def test_preprocess_signal_outputs(experiment):
    experiment.preprocess_signal()

    assert experiment.signal.dtype == np.float32
    assert experiment.signal.shape == experiment.time.shape
    assert np.isfinite(experiment.signal).all()
    assert np.isfinite(experiment.fitted_isosbestic).all()
    assert experiment.processed_sig.shape == experiment.raw_signal.shape
    assert experiment.processed_iso.shape == experiment.raw_isosbestic.shape
    assert "isosbestic_fit" in experiment.metadata
    assert "r2_val" in experiment.metadata["isosbestic_fit"]
    assert np.isfinite(experiment.metadata["isosbestic_fit"]["r2_val"])


def test_preprocess_recovers_signal_reasonably(experiment, sim):
    experiment.preprocess_signal(
        iso_correction_method="dF/F",
        signal_normalization="nullZ",
        fit_using="IRLS",
    )

    recovered = experiment.signal
    truth = sim.neural_true[:recovered.size]

    corr = np.corrcoef(recovered, truth)[0, 1]
    assert np.isfinite(corr)
    assert corr > 0.4


def test_extract_trial_data_creates_trial_and_baseline_objects(experiment, sim):
    experiment.run_pipeline()

    assert isinstance(experiment.trial_data, PhotometryData)
    assert isinstance(experiment.baseline_data, PhotometryData)
    assert experiment.trial_data.n_trials == sim.events["event"].size
    assert experiment.baseline_data.n_trials == sim.events["event"].size
    assert experiment.trial_data.n_times > 0
    assert experiment.baseline_data.n_times > 0
    assert "event" in experiment.trial_data.obs.columns
    assert np.isfinite(experiment.trial_data.X).all()
    assert np.isfinite(experiment.baseline_data.X).all()


def test_extract_trial_data_overlap_check_raises(experiment):
    experiment.preprocess_signal()

    experiment.events["choice_left"] = experiment.events["event"] + 0.3
    experiment.events["choice_right"] = experiment.events["event"] + 0.4

    with pytest.raises(ValueError, match="over lap|overlap"):
        experiment.extract_trial_data(
            align_to="event",
            center_on=["choice_left", "choice_right"],
            trial_bounds=(-1.0, 2.0),
            baseline_bounds=(-1.0, 0.0),
            event_tolerences={
                "choice_left": (0.0, 1.0),
                "choice_right": (0.0, 1.0),
            },
            check_overlap=True,
        )


def test_extract_trial_data_requires_baseline_for_zero_norm(experiment):
    experiment.preprocess_signal()

    with pytest.raises(ValueError, match="Baseline bounds"):
        experiment.extract_trial_data(
            align_to="event",
            center_on=["event"],
            trial_bounds=(-1.0, 2.0),
            baseline_bounds=None,
            trial_normalization="zero",
        )
