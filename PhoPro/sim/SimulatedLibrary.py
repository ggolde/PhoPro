"""Utilities for generating and loading simulated photometry libraries."""

from __future__ import annotations
from typing import Any, Callable

import itertools
import inspect
import logging
import json

import numpy as np

from pathlib import Path

from . import SimulatedPhotometry
from ..core.PhotometryLoader import PhotometryLoader

#########################
#region --- GENERATOR ---
#########################
class SimulatedLibraryGenerator:
    """Generate simulated photometry libraries from parameter permutations.

    Parameters
    ----------
    contanst_kwargs : dict[str, Any]
        Keyword arguments passed unchanged to
        ``SimulatedPhotometry.from_parameters()`` for every simulation.
    to_permute_kwargs : dict[str, list[Any]]
        Keyword arguments whose values are fully crossed to create parameter
        permutations. Each dictionary value must be a list of candidate values.
    replicates : int, optional
        Number of replicates generated for each parameter permutation, by
        default 1.
    seed : int | None, optional
        Seed used to shuffle the generated per-simulation seeds, by default
        None.

    Attributes
    ----------
    params : dict[int, dict[str, Any]]
        Parameter dictionary for each generated library member.
    n_permutations : int
        Number of unique parameter permutations.
    n_samples : int
        Total number of generated samples, including replicates.
    """

    params: dict[int, dict[str, Any]]

    def __init__(
            self,
            contanst_kwargs: dict[str, Any],
            to_permute_kwargs: dict[str, list[Any]],
            replicates: int = 1,
            seed: int | None = None,
            ) -> None:
        """Initialize the generator and build the parameter table."""
        # pre-validate kwargs
        self._validate_kwargs(SimulatedPhotometry.from_parameters, contanst_kwargs)
        self._validate_kwargs(SimulatedPhotometry.from_parameters, to_permute_kwargs)

        # assign attrs
        self.constants = contanst_kwargs
        self.to_permute = to_permute_kwargs
        self.replicates = replicates
        self.seed = seed

        # build param permutations
        self._build_params()

    # --- VALIDATION ---
    def _validate_kwargs(
            self,
            func: Callable,
            kwargs: dict[str, Any],
            ) -> None:
        """Validate keyword arguments against a callable signature."""
        sig = inspect.signature(func)
        params = sig.parameters
        skip = {'self', 'cls'}

        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )

        accepted = {
            name for name, p in params.items()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and name not in skip
        }

        required = {
            name for name, p in params.items()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and p.default is inspect.Parameter.empty
            and name not in skip
        }

        missing = required - kwargs.keys()
        unexpected = set() if accepts_var_kw else (kwargs.keys() - accepted)

        if missing:
            raise TypeError(f"Missing required kwargs for {func.__qualname__}(): {sorted(missing)}")
        if unexpected:
            raise TypeError(f"Unexpected kwargs for {func.__qualname__}(): {sorted(unexpected)}")

    # --- PERMUTAION ---
    def _permutation_generator(self):
        """Yield dictionaries for each crossed parameter permutation."""
        for inp in itertools.product(*self.to_permute.values()):
            yield dict(zip(self.to_permute.keys(), inp))

    def _build_params(self) -> None:
        """Build parameter dictionaries and assign unique simulation seeds."""
        # generate permutations
        permutations = [p for p in self._permutation_generator()]
        self.n_permutations = len(permutations)
        self.n_samples = self.n_permutations * self.replicates

        # set up unique seeds
        rng = np.random.default_rng(self.seed)
        seed_bank = np.arange(1, self.n_samples + 1).astype(int)
        rng.shuffle(seed_bank)

        # set up iteration
        i = 0
        self.params = {}

        for perm_id, perm in enumerate(permutations):
            for rep_id in range(self.replicates):
                self.params[i] = (
                    self.constants | perm | {'seed' : int(seed_bank[i])}
                )
                i = i + 1

    # --- OPERATIONS ---
    def update_params(self, to_update: dict[str, Any]) -> None:
        """Update every built parameter dictionary.

        Parameters
        ----------
        to_update : dict[str, Any]
            Keyword-value pairs to merge into each built parameter dictionary.
        """
        for i, args in self.params.items():
            self.params[i] = args | to_update

    # --- generation ---
    def generate_library(
            self,
            output_dir: Path,
            to_trials_kwargs: dict[str, Any],
            prefix: str = 'simlib',
            param_file: str = '_params.json',
            true_file: str = '_true_trials.h5ad',
            exp_folder: str | None = None,
            log_file: str | None = None,
            operation: Callable[[SimulatedPhotometry], None] | None = None,
            ) -> None:
        """Generate simulated experiments and combined true-signal trials.

        Parameters
        ----------
        output_dir : Path
            Directory where generated files are written.
        to_trials_kwargs : dict[str, Any]
            Keyword arguments passed to
            ``SimulatedPhotometry.to_PhotometryData()`` when building the true
            signal ``PhotometryData`` object.
        prefix : str, optional
            Prefix for generated per-experiment CSV files, by default
            ``"simlib"``.
        param_file : str, optional
            File name for the saved parameter table, by default
            ``"_params.json"``.
        true_file : str, optional
            File name for the combined true-signal ``PhotometryData`` object,
            by default ``"_true_trials.h5ad"``.
        exp_folder : str | None, optional
            The subfolder within ``output_dir`` in which per-experiment CSVs
            are saved. If ``None``, experiment CSVs are not saved, by default
            None.
        log_file : str | None, optional
            Optional log file path relative to ``output_dir``, by default None.
        operation : Callable[[SimulatedPhotometry], None] | None, optional
            Optional function applied to each generated
            ``SimulatedPhotometry`` object before export and trial extraction.
            This can be used to add events or otherwise modify generated data,
            by default None.
        """

        # coerce inputs
        output_dir = Path(output_dir)
        if not output_dir.exists():
            Path(output_dir).mkdir(exist_ok=True)
        if (exp_folder is not None) and not (output_dir / exp_folder).exists():
            Path(output_dir / exp_folder).mkdir(exist_ok=True)

        # --- set up logger ---
        logger = logging.getLogger(__name__)
        if log_file is not None:
            logging.basicConfig(filename=output_dir / log_file, filemode='w', level=logging.INFO, force=True)
        logger.info(
            f'Beginning generation of {self.n_samples} simulated datasets '
            f'from {self.n_permutations} permutations of parameters with '
            f'{self.replicates} replicates each.\n'
        )

        # save param file
        param_file_path = output_dir / param_file
        logger.info(f'Saving parameters to {str(param_file_path)}...\n')
        with open(param_file_path, 'w') as f:
            json.dump(self.params, f, indent=4)

        # iterate over all params
        i = 0
        logger.info(f'Iterating over {self.n_samples} parameters...\n')

        for lib_id, params in self.params.items():
            logger.info(f'Generating experiment {lib_id} ({i+1}/{self.n_samples})...')
            sim = SimulatedPhotometry.from_parameters(**params)

            if operation is not None:
                logger.info(f'Executing custom operation...')
                operation(sim)

            if exp_folder is not None:
                save_path = output_dir / exp_folder / f'{prefix}_{lib_id}.csv'
                logger.info(f'Saving generated data to {save_path}...')
                sim.to_experiment_csv(save_path).to_csv(save_path, index=False)

            logger.info(f'Extracting trial data...')
            if i == 0:
               logger.info(f'Creating first trial data instance...')
               data = (
                   sim
                   .to_PhotometryData(**to_trials_kwargs)
                   .mutate_obs(lib_id = lib_id)
               )
               trials = data.copy()
            else:
                logger.info(f'Accumulating result...')
                data = (
                    sim
                    .to_PhotometryData(**to_trials_kwargs)
                    .mutate_obs(lib_id = lib_id)
                )
                trials.combine_obj(data, inplace=True) #type: ignore

            logger.info('Completed job.\n')

            # iterate
            i += 1

        # save true signal trials
        trials.write_h5ad(output_dir / true_file) #type: ignore

        # report success
        logger.info(f'Library generation complete!')

    # --- UTILITY ---
    def create_loader_args(
            self,
            operation: Callable[[SimulatedPhotometry], None] | None = None,
            as_single_channel: bool = False,
            ) -> list[dict[str, Any]]:
        """Create loader argument dictionaries for all generated parameters.

        Parameters
        ----------
        operation : Callable[[SimulatedPhotometry], None] | None, optional
            Optional function applied after simulation in
            ``SimulatedParamLoader``, by default None.
        as_single_channel : bool, optional
            Whether loaders should omit the isosbestic channel, by default
            False.

        Returns
        -------
        list[dict[str, Any]]
            Loader keyword dictionaries covering all generated parameters.
        """
        return [
            dict(key=str(key), operation=operation, as_single_channel=as_single_channel)
            for key in self.params.keys()
        ]

#endregion

######################
#region --- LOADER ---
######################
class SimulatedParamLoader(PhotometryLoader):
    """Load one simulated experiment from a saved parameter table.

    Parameters
    ----------
    json : str
        Path to the JSON parameter table written by
        ``SimulatedLibraryGenerator.generate_library()``.
    key : str
        Identifier of the parameter set to load.
    operation : Callable[[SimulatedPhotometry], None] | None, optional
        Optional function applied to the generated ``SimulatedPhotometry``
        object before extracting loader data, by default None.
    as_single_channel : bool, optional
        Whether to omit the isosbestic channel from extracted data, by default
        False.
    """

    def __init__(
            self,
            json: str,
            key: str,
            operation: Callable[[SimulatedPhotometry], None] | None = None,
            as_single_channel: bool = False,
            ) -> None:
        """Initialize the simulated-parameter loader."""
        self.json = json
        self.key = key
        self.operation = operation
        self.as_single_channel = as_single_channel

    def extract_data(self) -> dict[str, Any]:
        """Generate and return raw photometry data for the selected parameter set.

        Returns
        -------
        dict[str, Any]
            Dictionary containing raw signal, optional raw isosbestic signal,
            time values, event onsets, and metadata.
        """
        with open(self.json, 'r') as f:
            params = json.load(f)
        params = params[self.key]

        sim = SimulatedPhotometry.from_parameters(**params)

        metadata: dict = params | {
            'source' : str(self.json),
            'lib_id' : str(self.key),
        }

        for k, v in metadata.items():
            if isinstance(v, dict):
                metadata[k] = '; '.join([f'{nk}={nv}' for nk, nv in v.items()])

        if self.operation is not None:
            self.operation(sim)

        data = dict(
            raw_signal = sim.F_exp,
            raw_isosbestic = None if self.as_single_channel else sim.F_iso,
            time = sim.time,
            events = sim.event_layer.onsets_to_dict(),
            metadata = metadata,
        )

        return data

#endregion
