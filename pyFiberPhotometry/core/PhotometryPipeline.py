from __future__ import annotations

import os
import inspect
import logging

from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from .PhotometryExperiment import PhotometryExperiment
from .PhotometeryData import PhotometryData
from .PhotometryLoader import PhotometryLoader

class PhotometryPipeline:
    def __init__(
            self,
            data_directory: str | Path,
            target_type: Literal['file', 'folder'],
            loader_cls: type[PhotometryLoader],
            experiment_cls: type[PhotometryExperiment] = PhotometryExperiment,
            data_cls: type[PhotometryData] = PhotometryData,
            recursive: bool = False,
            pattern: str | None = None,
            ) -> None:
        """Initialize a generic directory-level photometry pipeline.

        Args:
            data_directory (str | Path): Root directory containing candidate
                input files or folders to process.
            target_type (Literal['file', 'folder']): File type to
                treat as a pipeline input. Must be either ``'file'`` or
                ``'folder'``.
            loader_cls (type[PhotometryLoader]): Loader class to use.
            experiment_cls (type[PhotometryExperiment]): Experiment class to use.
            data_cls (type[PhotometryData]): Trial-data class to use.
            recursive (bool): Whether to recursively search under
                ``data_directory`` when discovering inputs.
            pattern (str | None): Optional glob pattern used to filter
                discovered inputs. If ``None``, all children under
                ``data_directory`` are considered.

        Raises:
            ValueError: If ``data_directory`` does not exist or is not a
                directory.
        """
        # validate inputs
        data_directory = Path(data_directory).expanduser()
        if not data_directory.exists():
            raise ValueError(f"Data directory {data_directory} does not exist.")
        if not data_directory.is_dir():
            raise ValueError(f"Data directory {data_directory} is not a directory.")    
    
        # save attributes
        self.data_directory = data_directory
        self.target_type = target_type
        self.loader_cls = loader_cls
        self.experiment_cls = experiment_cls
        self.data_cls = data_cls
        self.recursive = recursive
        self.pattern = pattern

    # --- job building helpers ---
    def _matches_input_kind(self, path: Path) -> bool:
        match self.target_type:
            case 'file':
                return path.is_file()
            case 'folder':
                return path.is_dir()
            case _:
                raise ValueError(f'Target type {self.target_type} not recognized.')
    
    def _build_jobs(
            self,
            inputs: list[Path],
            loader_kwargs: dict[str, Any] | list[dict[str, Any]],
            preprocess_kwargs: dict[str, Any],
            trial_extraction_kwargs: dict[str, Any],
            ) -> list[dict[str, Any]]:
        norm_loader_kwargs = [loader_kwargs] if isinstance(loader_kwargs, dict) else list(loader_kwargs)
        jobs = [
            {
                'input': fpath, 
                'loader_kwargs': loader_args, 
                'preprocess_kwargs': preprocess_kwargs,
                'trial_extraction_kwargs': trial_extraction_kwargs,
            }
            for fpath in inputs for loader_args in norm_loader_kwargs
        ]
        return jobs
    
    # --- validation helpers ---
    def _validate_kwargs(
            self,
            func: Callable,
            kwargs: dict[str, Any],
            supplied_positionally: Iterable[str] = (),
            ) -> None:
        sig = inspect.signature(func)
        params = sig.parameters
        skip = {'self', 'cls', *supplied_positionally}

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
        
    def _validate_all_kwargs(
            self,
            loader_kwargs: dict[str, Any] | list[dict[str, Any]],
            preprocess_kwargs: dict[str, Any],
            trial_extraction_kwargs: dict[str, Any],
            ) -> None:
        loader_sig = inspect.signature(self.loader_cls.__init__)
        loader_positional = [
            name for name, param in loader_sig.parameters.items()
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and name != 'self'
        ]
        loader_path_arg = loader_positional[0] if len(loader_positional) > 0 else None

        # unwrap loaded if wrapped
        norm_loader_kwargs = [loader_kwargs] if isinstance(loader_kwargs, dict) else list(loader_kwargs)
        for loader_kwargs_i in norm_loader_kwargs: 
            self._validate_kwargs(
                self.loader_cls.__init__,
                loader_kwargs_i,
                supplied_positionally=() if loader_path_arg is None else (loader_path_arg,),
            )

        self._validate_kwargs(self.experiment_cls.preprocess_signal, preprocess_kwargs)
        self._validate_kwargs(self.experiment_cls.extract_trial_data, trial_extraction_kwargs)

    def _validate_output_dir(self, output_dir: Path | None) -> None:
        if output_dir is None:
            return
        path = Path(output_dir)
        if not path.exists():
            if not path.parent.exists():
                raise ValueError(f"Parent directory does not exist: {path.parent}")
            path.mkdir(exist_ok=True)

    def _validate_low_memory_mode(
            self, 
            low_memory_mode: bool, 
            trial_output_path: str | Path | None,
            ) -> None:
        if low_memory_mode:
            if trial_output_path is None:
                raise ValueError(f'Using low memory mode without an output file.')
            elif os.path.exists(trial_output_path):
                raise ValueError(f'Using low memory mode with preexsisting output file: {trial_output_path}')
        
    # --- pipeline helpers ---
    def _load_experiment(self, path: Path, loader_kwargs: dict[str, Any]) -> type[PhotometryExperiment]:
        loader = self.loader_cls(path, **loader_kwargs)
        exp: type[PhotometryExperiment] = loader.load()
        return exp
    
    def _run_preprocessing(self, exp: type[PhotometryExperiment], preprocess_kwargs: dict[str, Any]) -> None:
        exp.preprocess_signal(**preprocess_kwargs)

    def _run_trial_extraction(self, exp: type[PhotometryExperiment], trial_extraction_kwargs: dict[str, Any]) -> None:
        exp.extract_trial_data(**trial_extraction_kwargs)

    def _accumulate_result(
            self,
            exp: type[PhotometryExperiment],
            trial_output_path: Path | None = None,
            low_memory_mode: bool = False,
            ) -> None:
        # four cases, low memory mode * first path
        if low_memory_mode:
            if os.path.exists(trial_output_path):
                exp.trial_data.append_on_disk_h5ad(trial_output_path)
            else:
                exp.trial_data.write_h5ad(trial_output_path)

        else:
            if self.trial_data is None:
                self.trial_data = exp.trial_data.copy()
            else:
                self.trial_data.combine_obj(exp.trial_data, inplace=True)

    def _finalize_result(
            self,
            trial_output_path: Path | None = None,
            low_memory_mode: bool = False,
            ) -> type[PhotometryData]:
        if low_memory_mode:
            trial_data: type[PhotometryData] = self.data_cls.read_h5ad(trial_output_path)

        else:
            if self.trial_data is None:
                raise ValueError('Trial data is missing. Perhaps all jobs errored.')
            
            trial_data = self.trial_data
            trial_data.obs.reset_index(drop=True, inplace=True)
            trial_data.obs.index = trial_data.obs.index.astype(str)

        if trial_output_path is not None:
            trial_data.write_h5ad(trial_output_path)

        return trial_data

    # --- public pipeline API ---
    def discover_inputs(self) -> list[Path]:
        """Discover candidate inputs under ``self.data_directory``."""
        if self.pattern is None:
            candidates = (
                self.data_directory.rglob('*')
                if self.recursive
                else self.data_directory.iterdir()
            )
        else:
            candidates = (
                self.data_directory.rglob(self.pattern)
                if self.recursive
                else self.data_directory.glob(self.pattern)
            )

        inputs = [path for path in candidates if self._matches_input_kind(path)]
        return inputs
    
    def run(
            self,

            # args
            loader_kwargs: dict[str, Any] | list[dict[str, Any]],
            preprocess_kwargs: dict[str, Any],
            trial_extraction_kwargs: dict[str, Any],

            # I/O
            output_dir: str | Path | None = None,
            log_file: str | None = None,
            trial_output_file: str | None = 'trials.h5ad',

            # pipeline params
            low_memory_mode: bool = False,

            # misc
            passdown_metadata: list[str] | None = ['source'],
            id_builder: Callable[[type[PhotometryExperiment]], str] | None = None,
            post_load_operation: Callable[[type[PhotometryExperiment]], None] | None = None,
            post_preprocess_operation: Callable[[type[PhotometryExperiment]], None] | None = None,
            post_trial_extraction_operation: Callable[[type[PhotometryExperiment]], None] | None = None,
            ) -> type[PhotometryData]:
        """Run the batch processing pipeline over all discovered inputs.

        For each discovered input, the pipeline constructs a loader, loads an
        experiment, preprocesses the continuous signal, extracts trial-wise
        data, optionally applies custom hook operations, passes selected
        metadata down into ``trial_data.obs``, and accumulates the resulting
        trial-data objects in memory or on disk.

        Args:
            output_dir (str | Path | None): Directory where outputs should be
                written. If ``None``, nothing is saved.
            loader_kwargs (dict[str, Any] | list[dict[str, Any]]): Keyword
                arguments passed to ``loader_cls`` for each job, excluding the
                discovered input path that the pipeline supplies positionally.
                Use list of dictionaries creates to create multiple jobs per
                single input, useful for file formats that contain multiple
                experiment's data within a single folder/file.
            preprocess_kwargs (dict[str, Any]): Keyword arguments passed to
                ``PhotometryExperiment.preprocess_signal()``.
            trial_extraction_kwargs (dict[str, Any]): Keyword arguments passed
                to ``PhotometryExperiment.extract_trial_data()``.
            log_file (str | None): Optional path to a log file. If provided,
                logging is configured to write to that file. Default ``None``.
            trial_output_file (str | None): Name of the output ``.h5ad`` file
                written under ``output_dir`` for accumulated trial data.
            low_memory_mode (bool): If ``True``, accumulate trial data
                directly on disk instead of keeping the full combined object in
                memory.
            passdown_metadata (list[str] | None): Metadata keys from
                ``exp.metadata`` to copy into ``exp.trial_data.obs`` for each
                processed experiment. If ``None``, no metadata columns are
                added.
            id_builder (Callable[[type[PhotometryExperiment]], str] | None):
                Optional callable used to construct and assign an experiment ID
                after loading. Automatically passes the experiment ID to ``trial_data.obs``.
            post_load_operation (Callable[[type[PhotometryExperiment]], None] | None):
                Optional callable run immediately after an experiment is loaded.
            post_preprocess_operation (Callable[[type[PhotometryExperiment]], None] | None):
                Optional callable run immediately after preprocessing completes.
            post_trial_extraction_operation (Callable[[type[PhotometryExperiment]], None] | None):
                Optional callable run immediately after trial extraction completes.

        Returns:
            type[PhotometryData]: The trial-data object containing all extracted trials.

        Raises:
            TypeError: If provided kwargs do not match the accepted signatures
                of the loader, preprocessing, or trial-extraction methods.
            ValueError: If the output configuration is invalid or if no inputs
                are discovered.
        """
        
        # --- set up logger ---
        logger = logging.getLogger(__name__)
        if log_file is not None:
            logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, force=True)
        logger.info('Beginning pipeline')

        # --- coerce inputs ---
        if output_dir is not None:
            output_dir = Path(output_dir)
            trial_output_path = os.path.join(output_dir, trial_output_file)
        else:
            trial_output_path = None

        # --- validate inputs ---
        logger.info('Validating inputs...')
        self._validate_output_dir(output_dir)
        self._validate_low_memory_mode(low_memory_mode, trial_output_path)
        self._validate_all_kwargs(loader_kwargs, preprocess_kwargs, trial_extraction_kwargs,)

        # --- construct jobs ---
        logger.info('Discovering inputs...')
        inputs = self.discover_inputs()
        if len(inputs) == 0:
            raise ValueError(f'No inputs discovered!')

        logger.info('Building jobs...')
        jobs = self._build_jobs(
            inputs=inputs, 
            loader_kwargs=loader_kwargs,
            preprocess_kwargs=preprocess_kwargs,
            trial_extraction_kwargs=trial_extraction_kwargs
        )

        # --- set up job iteration ---
        n_jobs = len(jobs)
        self.trial_data: type[PhotometryData] = None
        n_errors = 0
        n_processed = 0

        # --- iterate over jobs ---
        logger.info(f'Iterating over {n_jobs} jobs...\n')
        for i, job in enumerate(jobs):
            try:
                logger.info(f'Processing job {job["input"]} ({i+1}/{n_jobs})')

                # 1. load experiment
                logger.info(f'Loading expriment...')
                exp = self._load_experiment(job['input'], job['loader_kwargs'])

                if post_load_operation is not None:
                    logger.info(f'Running custom post loading operation...')
                    post_load_operation(exp)

                # 2. run preprocess
                logger.info(f'Preprocessing signal...')
                self._run_preprocessing(exp, job['preprocess_kwargs'])

                if post_preprocess_operation is not None:
                    logger.info(f'Running custom post preprocessing operation...')
                    post_preprocess_operation(exp)
                
                # 3. run extract trial data
                logger.info(f'Extracting trial data...')
                self._run_trial_extraction(exp, job['trial_extraction_kwargs'])

                if post_trial_extraction_operation is not None:
                    logger.info(f'Running custom post trial extraction operation...')
                    post_trial_extraction_operation(exp)

                # warn if any trial windows were invalid
                if exp.metadata['invalid_windows'] is not None:
                    logger.warning(
                        f'Invalid trial windows at indexes {exp.metadata["invalid_windows"]} '
                        f'have been dropped.'
                    )

                # assign uid if function provided
                if id_builder is not None:
                    logger.info(f'Assiging experiment ID...')
                    exp.id = id_builder(exp)
                    exp.trial_data.obs['experiment_id'] = exp.id
                    logger.info(f'Expriment ID = {str(exp.id)}')

                # pass down specified metadata
                if passdown_metadata is not None:
                    logger.info(f'Passing down experiment metadata as columns...')
                    exp.trial_data.add_obs_columns(add_from=exp.metadata, keys=passdown_metadata)

                # accumulate object
                logger.info(f'Accumulating result...')
                self._accumulate_result(exp, trial_output_path, low_memory_mode)

                # log expriment info
                logger.info(
                    f'Finished processing experiment with '
                    f'{exp.trial_data.n_trials} trials x {exp.trial_data.n_times} timepoints.'
                )

                # clean up before next iteration
                logger.info(f'Job {i+1} complete.\n')
                del exp
                n_processed += 1

            # handle errors
            except Exception as e:
                logger.error(f'Error processing {job["input"]}: \n\t {e}\n', exc_info=True)
                n_errors += 1

        # --- finalize ---
        logger.info('Finalizng results...')
        trial_data = self._finalize_result(trial_output_path, low_memory_mode)

        logger.info(
            f'Processing pipeline complete with {n_errors} errors '
            f'and {n_processed} successes out of {n_jobs} jobs.'
        )

        return trial_data
        
