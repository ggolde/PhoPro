"""Simulation tools for synthetic fiber photometry recordings."""

from .SimulatedPhotometry import SimulatedPhotometry

from .layers import (
    TimeBase,
    PhotobleachingLayer,
    EventLayer, EventSpec,
    NoiseShotLayer, NoiseGaussianLayer,
    MovementAttenuationLayer,
    ArtifactSpikeLayer, ArtifactJumpLayer,
)

__all__ = [
    "SimulatedPhotometry",
    "TimeBase",
    "PhotobleachingLayer",
    "EventLayer",
    "EventSpec",
    "NoiseShotLayer",
    "NoiseGaussianLayer",
    "MovementAttenuationLayer",
    "ArtifactSpikeLayer",
    "ArtifactJumpLayer",
]
