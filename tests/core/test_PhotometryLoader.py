import numpy as np
import pytest
import pathlib

from PhoPro.core.PhotometryExperiment import PhotometryExperiment
from PhoPro.core.PhotometryData import PhotometryData
from PhoPro.core.PhotometryLoader import PhotometryLoader, TDTLoader, CSVLoader

# --- correct loading ---

def test_csv_loader_proper_loading():
    test_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_file = test_dir / 'test_data/dummy_experiment.csv'
    event_cols = ['event1', 'event2', 'event3']

    loader = CSVLoader(
        target_file,
        time_col='time',
        signal_col='experimental',
        isosbestic_col='isosbestic',
        event_cols=event_cols,
        downsample=None,
    )
    exp = loader.load()

    assert np.all(exp.raw_signal == np.asarray([10.4, 3.2, 2.4, 5.7, 6.5, 5.3]))
    assert np.all(exp.raw_isosbestic == np.asarray([3.1, 2.8, 2.6, 2.5, 2.4, 2.3]))
    assert all(label in exp.events for label in event_cols)
    assert np.all(exp.events['event1'] == np.asarray([4.0]))
    assert np.all(exp.events['event2'] == np.asarray([1.0]))
    assert np.all(exp.events['event3'] == np.asarray([3.0]))
    assert (exp.frequency == 1.0)

def test_annotation_handler_json():
    test_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_file = test_dir / 'test_data/dummy_experiment.csv'
    target_annotation = test_dir / 'test_data/dummy_annotation.json'
    event_cols = ['event1', 'event2', 'event3']

    loader = CSVLoader(
        target_file,
        time_col='time',
        signal_col='experimental',
        isosbestic_col='isosbestic',
        event_cols=event_cols,
        downsample=None,
        annotation_file=target_annotation,
        annotation_handler='json'
    )
    exp = loader.load()

    assert 'key1' in exp.metadata
    assert 'key2' in exp.metadata
    assert exp.metadata['key1'] == 'value1'
    assert exp.metadata['key2'] == 50

def test_annotation_handler_yaml():
    test_dir = pathlib.Path(__file__).parent.parent.resolve()

    target_file = test_dir / 'test_data/dummy_experiment.csv'
    target_annotation = test_dir / 'test_data/dummy_annotation.yml'
    event_cols = ['event1', 'event2', 'event3']

    loader = CSVLoader(
        target_file,
        time_col='time',
        signal_col='experimental',
        isosbestic_col='isosbestic',
        event_cols=event_cols,
        downsample=None,
        annotation_file=target_annotation,
        annotation_handler='yaml'
    )
    exp = loader.load()

    assert 'key1' in exp.metadata
    assert 'key2' in exp.metadata
    assert exp.metadata['key1'] == 'value1'
    assert exp.metadata['key2'] == 50