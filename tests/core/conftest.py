import numpy as np
import pandas as pd
import pytest

from pyFiberPhotometry.core.PhotometryExperiment import PhotometryExperiment
from pyFiberPhotometry.core.PhotometeryData import PhotometryData
from pyFiberPhotometry.utils.sim import SimulatedPhotometryGenerator

# set up dummy experiment tester subclass
class DummyExperiment(PhotometryExperiment):
    def run_pipeline(self) -> None:
        self.preprocess_signal(
            cutoff_frequency=3.0,
            order=4,
            iso_correction_method="dF/F",
            signal_normalization="none",
            fit_using="IRLS",
            maxiter=200,
            c=3,
            detrend_bleaching=False,
        )
        self.extract_trial_data(
            align_to="event",
            center_on=["choice_left", "choice_right"],
            trial_bounds=(-2.0, 4.0),
            baseline_bounds=(-2.0, 0.0),
            event_tolerences={
                "choice_left": (0.0, 1.0),
                "choice_right": (0.0, 1.0),
            },
            trial_normalization="none",
            check_overlap=True,
            time_error_threshold=0.2,
            event_conflict_logic="first",
        )


@pytest.fixture
def sim():
    sim = SimulatedPhotometryGenerator(
        T_sec=120,
        fs=20.0,
        n_events=12,
        event_dur_sec=2.0,
        n_artifacts=4,
        seed=7,
        bleach_iso_scale=1.0,
    )
    sim.add_event(
        time_range=(0.2, 0.8),
        overall_prob=1.0,
        choices=["choice_left", "choice_right"],
        choice_probs=[0.5, 0.5],
        relative_to="event",
    )
    return sim


@pytest.fixture
def experiment(sim):
    return DummyExperiment(
        raw_signal=sim.F_exp,
        raw_isosbestic=sim.F_iso,
        time=sim.t,
        frequency=sim.fs,
        events=sim.events,
        metadata={'source':'simulated_test'}
    )


@pytest.fixture
def photometry_data():
    trace_size = (3, 20)
    rng = np.random.default_rng(42)

    obs = pd.DataFrame({
        "animal": ["a", "a", "b"],
        "condition": ["x", "x", "y"],
        "score": [1.0, 3.0, 5.0],
    })

    data = rng.random(size=trace_size, dtype=float)
    time = np.arange(0, 20)
    layers = {'layer' : rng.random(size=trace_size, dtype=float)}
    metadata = {'frequency' : 1.0}

    return PhotometryData.from_arrays(
        obs=obs,
        data=data,
        time_points=time,
        layers=layers,
        metadata=metadata,
    )
