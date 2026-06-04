import numpy as np
import pandas as pd
import pathlib
import pytest

from pyFiberPhotometry.core.PhotometryExperiment import PhotometryExperiment
from pyFiberPhotometry.core.PhotometeryData import PhotometryData
from pyFiberPhotometry.core.PhotometryLoader import PhotometryLoader
from pyFiberPhotometry.core.PhotometryPipeline import PhotometryPipeline
from pyFiberPhotometry.utils.sim import SimulatedPhotometryGenerator

# simulated data
def simulate_data() -> SimulatedPhotometryGenerator:
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

# set up dummy experiment tester subclass
class DummyExperiment(PhotometryExperiment):
    def run_pipeline(self) -> None:
        self.preprocess_signal(
            cutoff_frequency=3.0,
            order=4,
            correction_method="dF/F",
            signal_normalization="none",
            fit_using="IRLS",
            maxiter=200,
            c=3,
        )
        self.extract_trial_data(
            align_to="event",
            center_on=["choice_left", "choice_right"],
            trial_bounds=(-2.0, 4.0),
            baseline_bounds=(-2.0, 0.0),
            trial_normalization="none",
            check_overlap=True,
            time_error_threshold=0.2,
            event_conflict_logic="first",
        )

# dummy loader class for pipeline tests
class DummyLoader(PhotometryLoader):
    def __init__(
            self,
            file: str,
            ) -> None:
        self.file = file
        self.metadata = {'key1': 'value1'}
        return
    
    def extract_data(self) -> dict:
        exp = simulate_data()
        self.metadata['source'] = self.file
        data = dict(
            raw_signal = exp.F_exp,
            raw_isosbestic = exp.F_iso,
            time = exp.t,
            frequency = None,
            events = exp.events,
            metadata = self.metadata,
        )
        return data

@pytest.fixture
def sim():
    return simulate_data()


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
def single_channel_experiment(sim):
    return PhotometryExperiment(
        raw_signal=sim.F_exp,
        raw_isosbestic=None,
        time=sim.t,
        frequency=sim.fs,
        events=sim.events.copy(),
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

@pytest.fixture
def dummy_pipeline():
    test_dir = pathlib.Path(__file__).parent.parent.resolve()
    target_dir = test_dir / 'test_data/dummy_pipeline'
    pipeline = PhotometryPipeline(
        data_directory=target_dir,
        target_type='file',
        recursive=False,
        pattern='dummy_file*.csv',
        loader_cls=DummyLoader,
    )

    return pipeline