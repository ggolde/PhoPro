"""Analysis tools for artifacts, peaks, group comparisons, and FMM models."""

from .artifact import ArtifactResult, ODS_Detector, Spline_Corrector
from .comparison import ClusterTestResult, cluster_depth_test, cluster_permutation_test
from .FMM import FMMResult, run_fastFMM
from .peaks import PeakResult, StaticThresholdDetector, RollingThresholdDetector

__all__ = [
    'ArtifactResult',
    'ODS_Detector',
    'Spline_Corrector',
    'ClusterTestResult',
    'cluster_depth_test',
    'cluster_permutation_test',
    'FMMResult',
    'run_fastFMM',
    'PeakResult',
    'StaticThresholdDetector',
    'RollingThresholdDetector'
]
