import numpy as np
import pytest

from pyFiberPhotometry.core.PhotometryExperiment import PhotometryExperiment
from pyFiberPhotometry.core.PhotometeryData import PhotometryData
from pyFiberPhotometry.utils.ops import neg_bi_exponential_5


# --- preprocessing ---

def test_preprocess_signal_outputs(experiment):
    experiment.preprocess_signal()

    assert np.issubdtype(experiment.signal.dtype, np.floating)
    assert experiment.signal.shape == experiment.time.shape
    assert np.isfinite(experiment.signal).all()
    assert np.isfinite(experiment.fitted_ref).all()
    assert experiment.filt_sig.shape == experiment.raw_signal.shape
    assert experiment.fitted_ref.shape == experiment.raw_signal.shape
    assert "reference_fit" in experiment.metadata
    assert experiment.metadata["reference_fit"]["type"] == "isosbestic"
    assert "r2_val" in experiment.metadata["reference_fit"]
    assert np.isfinite(experiment.metadata["reference_fit"]["r2_val"])


def test_preprocess_recovers_signal_reasonably(experiment, sim):
    experiment.preprocess_signal(
        correction_method="dF/F",
        signal_normalization="nullZ",
        fit_using="IRLS",
    )

    recovered = experiment.signal
    truth = sim.neural_true[:recovered.size]

    corr = np.corrcoef(recovered, truth)[0, 1]
    assert np.isfinite(corr)
    assert corr > 0.4


def test_preprocess_rejects_channel_incompatible_correction_methods(
    experiment,
    single_channel_experiment,
):
    with pytest.raises(ValueError, match="single channel"):
        experiment.preprocess_signal(correction_method="dB/B")

    with pytest.raises(ValueError, match="dual channel"):
        single_channel_experiment.preprocess_signal(correction_method="dF/F")


def test_fit_photobleaching_curve_recovers_smooth_bleaching_trend(experiment):
    params = (2.0, 0.08, 1.0, 0.01, 5.0)
    expected = neg_bi_exponential_5(experiment.time, *params)
    signal = expected + 1e-3 * np.sin(0.7 * experiment.time)

    fitted_curve, r2_val, fitted_params = experiment.fit_photobleaching_curve(
        signal=signal,
        window_dur=2.0,
    )

    assert fitted_curve.shape == signal.shape
    assert len(fitted_params) == 5
    assert np.isfinite(fitted_curve).all()
    assert np.isfinite(fitted_params).all()
    assert r2_val > 0.99
    assert np.nanmedian(np.abs(fitted_curve - expected)) < 0.05


def test_preprocess_single_channel_processing(single_channel_experiment):
    exp = single_channel_experiment

    exp.preprocess_signal(
        correction_method="dB/B",
        signal_normalization="none",
        cutoff_frequency=3.0,
    )

    assert exp.channel_mode == "single"
    assert exp.metadata["reference_fit"]["type"] == "photobleaching"
    assert exp.metadata["reference_fit"]["r2_val"] > 0.5
    assert exp.metadata["correction_method"] == "dB/B"
    assert exp.filt_sig.shape == exp.raw_signal.shape
    assert exp.fitted_ref.shape == exp.raw_signal.shape
    assert exp.signal.shape == exp.raw_signal.shape
    assert np.isfinite(exp.filt_sig).all()
    assert np.isfinite(exp.fitted_ref).all()
    assert np.isfinite(exp.signal).all()


def test_preprocess_requires_detector_when_corrector_is_given(experiment):
    class Corrector:
        def correct(self, signal, time, artifacts):
            return signal

    with pytest.raises(ValueError, match="artifact_detector"):
        experiment.preprocess_signal(artifact_corrector=Corrector())


def test_preprocess_accepts_callable_correction_and_normalization(experiment):
    def correction(signal, fitted_ref):
        return signal - fitted_ref

    def normalize(signal):
        return signal / np.nanstd(signal)

    experiment.preprocess_signal(
        correction_method=correction,
        signal_normalization=normalize,
    )

    assert experiment.metadata["correction_method"] == "correction"
    assert np.isfinite(experiment.signal).all()


# --- trial extraction: basic outputs ---

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


def test_extract_trial_data_center_on_none_uses_alignment_event(experiment):
    experiment.preprocess_signal()

    experiment.extract_trial_data(
        align_to="event",
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=None,
        trial_normalization="none",
    )

    assert experiment.trial_data.n_trials == experiment.events["event"].size
    assert "event" in experiment.trial_data.obs.columns
    assert np.allclose(experiment.trial_data.obs["event"].to_numpy(dtype=float), 0.0)


def test_extract_trial_data_single_center_on_does_not_overlap(experiment):
    experiment.preprocess_signal()

    experiment.extract_trial_data(
        align_to="event",
        center_on="choice_left",
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=None,
        trial_normalization="none",
        check_overlap=True,
    )

    assert "choice_left" in experiment.trial_data.obs.columns
    assert experiment.trial_data.obs["choice_left"].notna().any()


# --- trial extraction: alignment inputs ---

def test_extract_trial_data_accepts_eventless_alignment_timestamps(experiment):
    experiment.preprocess_signal()

    centers = [10.0, 20.0, 30.0]
    experiment.extract_trial_data(
        align_to=centers,
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=None,
        trial_normalization="none",
    )

    assert isinstance(experiment.trial_data, PhotometryData)
    assert experiment.trial_data.n_trials == len(centers)
    assert "ALIGNMENTS" in experiment.trial_data.obs.columns
    assert "align_event" not in experiment.trial_data.obs.columns
    assert np.allclose(experiment.trial_data.obs["ALIGNMENTS"].to_numpy(dtype=float), 0.0)
    assert np.isfinite(experiment.trial_data.X).all()


def test_extract_trial_data_accepts_scalar_integer_alignment_timestamp(experiment):
    experiment.preprocess_signal()

    experiment.extract_trial_data(
        align_to=20,
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=None,
        trial_normalization="none",
    )

    assert experiment.trial_data.n_trials == 1
    assert "ALIGNMENTS" in experiment.trial_data.obs.columns
    assert np.allclose(experiment.trial_data.obs["ALIGNMENTS"].to_numpy(dtype=float), 0.0)


def test_extract_trial_data_accepts_multiple_alignment_event_labels(experiment):
    experiment.preprocess_signal()

    experiment.events["align_a"] = np.asarray([10.0, 20.0])
    experiment.events["align_b"] = np.asarray([10.0, 15.0])

    experiment.extract_trial_data(
        align_to=["align_b", "align_a"],
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=None,
        trial_normalization="none",
    )

    assert isinstance(experiment.trial_data, PhotometryData)
    assert experiment.trial_data.n_trials == 4
    assert "ALIGNMENTS" in experiment.trial_data.obs.columns
    assert "align_event" in experiment.trial_data.obs.columns
    assert experiment.trial_data.obs["align_event"].tolist() == [
        "align_b",
        "align_a",
        "align_b",
        "align_a",
    ]
    assert np.allclose(experiment.trial_data.obs["ALIGNMENTS"].to_numpy(dtype=float), 0.0)


# --- trial extraction: validation and errors ---

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
            check_overlap=True,
        )


def test_extract_trial_data_reports_missing_alignment_labels(experiment):
    experiment.preprocess_signal()

    with pytest.raises(KeyError, match="missing"):
        experiment.extract_trial_data(
            align_to=["event", "missing"],
            trial_bounds=(-1.0, 2.0),
        )


def test_extract_trial_data_reports_missing_center_on_labels(experiment):
    experiment.preprocess_signal()

    with pytest.raises(KeyError, match="missing_center"):
        experiment.extract_trial_data(
            align_to="event",
            center_on=["choice_left", "missing_center"],
            trial_bounds=(-1.0, 2.0),
        )


@pytest.mark.parametrize("align_to", [[], "empty_event"])
def test_extract_trial_data_rejects_empty_alignments(experiment, align_to):
    experiment.preprocess_signal()
    experiment.events["empty_event"] = np.asarray([], dtype=float)

    with pytest.raises(ValueError, match="align_to|timestamp|event label"):
        experiment.extract_trial_data(
            align_to=align_to,
            trial_bounds=(-1.0, 2.0),
        )


@pytest.mark.parametrize("trial_normalization", ["zscore", "zero", "mad", "amp"])
def test_extract_trial_data_requires_baseline_for_baseline_normalizations(
    experiment,
    trial_normalization,
):
    experiment.preprocess_signal()

    with pytest.raises(ValueError, match="Baseline bounds"):
        experiment.extract_trial_data(
            align_to="event",
            trial_bounds=(-1.0, 2.0),
            baseline_bounds=None,
            trial_normalization=trial_normalization,
        )


def test_extract_trial_data_none_tolerances_default_to_trial_bounds(experiment):
    experiment.preprocess_signal()

    experiment.extract_trial_data(
        align_to="event",
        center_on="choice_left",
        trial_bounds=(-1.0, 2.0),
        event_tolerences={"choice_left": None},
        baseline_bounds=None,
        trial_normalization="none",
    )

    assert "choice_left" in experiment.trial_data.obs.columns
    assert experiment.trial_data.obs["choice_left"].notna().any()


# --- trial extraction: invalid windows ---

def test_extract_trial_data_drop_invalid_windows_keeps_align_event_aligned(experiment):
    experiment.preprocess_signal()

    experiment.events["align_a"] = np.asarray([0.5, 30.0])
    experiment.events["align_b"] = np.asarray([40.0, experiment.time[-1] - 0.25])

    experiment.extract_trial_data(
        align_to=["align_a", "align_b"],
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=None,
        trial_normalization="none",
        invalid_window_policy="drop",
    )

    assert experiment.metadata["invalid_windows"] == [0, 3]
    assert experiment.trial_data.n_trials == 2
    assert experiment.trial_data.obs["align_event"].tolist() == ["align_a", "align_b"]


def test_extract_trial_data_invalid_window_policy_error_raises(experiment):
    experiment.preprocess_signal()

    with pytest.raises(ValueError, match="Invalid trial windows"):
        experiment.extract_trial_data(
            align_to=[0.5],
            trial_bounds=(-1.0, 2.0),
            invalid_window_policy="error",
        )


def test_extract_trial_data_all_invalid_windows_raises(experiment):
    experiment.preprocess_signal()

    with pytest.raises(ValueError, match="No trials remain"):
        experiment.extract_trial_data(
            align_to=[0.5],
            trial_bounds=(-1.0, 2.0),
            invalid_window_policy="drop",
        )


# --- trial extraction: normalization ---

def test_extract_trial_data_custom_normalization_receives_baseline(experiment):
    experiment.preprocess_signal()
    calls = {}

    def normalize(trial_signals, baseline_signals):
        calls["trial_shape"] = trial_signals.shape
        calls["baseline_shape"] = baseline_signals.shape
        return trial_signals - np.nanmean(baseline_signals, axis=1, keepdims=True)

    experiment.extract_trial_data(
        align_to="event",
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=(-1.0, 0.0),
        trial_normalization=normalize,
    )

    assert calls["trial_shape"] == experiment.trial_data.X.shape
    assert calls["baseline_shape"] == experiment.baseline_data.X.shape
    assert np.isfinite(experiment.trial_data.X).all()


def test_extract_trial_data_interp_alignment_has_shared_timebase(experiment):
    experiment.preprocess_signal()

    experiment.extract_trial_data(
        align_to=[10.25, 20.5, 30.75],
        trial_bounds=(-1.0, 2.0),
        window_alignment="interp",
        trial_normalization="none",
    )

    assert np.isclose(experiment.trial_data.ts[0], -1.0)
    assert np.isclose(experiment.trial_data.ts[-1], 1.95)
    assert np.allclose(experiment.trial_data.obs["ALIGNMENTS"].to_numpy(dtype=float), 0.0)
