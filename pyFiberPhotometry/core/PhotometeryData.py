from __future__ import annotations
from typing import Any, Callable, Literal, cast
from anndata.experimental import concat_on_disk

import matplotlib.pyplot as plt
import matplotlib.axes
import anndata as ad
import numpy as np
import pandas as pd
import os

from ..utils.ops import downsample_ndarray, downsample_1d, reconstruct_time_points

ad.settings.allow_write_nullable_strings = True

class PhotometryData:
    """
    Wrap an AnnData object for trial-wise photometry time-series data.
    """
    adata: ad.AnnData

    # --- constructors ---
    def __init__(self, adata: ad.AnnData):
        """
        Initialize a PhotometryData wrapper from an AnnData object.
        Args:
            adata (anndata.AnnData): AnnData object containing time-series data.
        Returns:
            None
        """
        assert isinstance(adata, ad.AnnData)
        self.adata = adata
    
    @classmethod
    def from_arrays(
        cls,
        obs: pd.DataFrame, 
        data: np.ndarray, 
        time_points: np.ndarray,
        layers: dict[str, np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
        ) -> "PhotometryData":
        """
        Construct PhotometryData from arrays and observation metadata.
        Args:
            obs (pd.DataFrame): Per-trial observation metadata.
            data (np.ndarray): Time-series data of shape (n_trials, n_time).
            time_points (np.ndarray): Time axis of shape (n_time,).
            layers (dict[str, np.ndarray] | None): Optional named layers matching shape of data.
            metadata (dict[str, Any] | None): Optional unstructured metadata stored in .uns.
        Returns:
            PhotometryData
        """
        obs = obs.copy()
        obs.reset_index(drop=True, inplace=True)
        obs.index = obs.index.astype(str)
        var = pd.DataFrame({"t": time_points})
        var.index = var.index.astype(str)

        A = ad.AnnData(X=data, obs=obs, var=var, uns=metadata)
        for k, v in (layers or {}).items():
            assert v.shape == data.shape
            A.layers[k] = v
        return cls(A)
    
    # --- I/O ---
    @classmethod
    def read_h5ad(cls, path: str) -> "PhotometryData":
        """
        Read a PhotometryData object from an .h5ad file.
        Args:
            path (str): Path to the .h5ad file.
        Returns:
            PhotometryData: Loaded PhotometryData instance.
        """        
        return cls(ad.read_h5ad(path))
    @classmethod
    def read_zarr(cls, path: str) -> "PhotometryData":
        """
        Read a PhotometryData object from zarr storage.
        Args:
            path (str): Path to the zarr storage.
        Returns:
            PhotometryData: Loaded PhotometryData instance.
        """      
        return cls(ad.read_zarr(path))
    
    def write_h5ad(self, path: str) -> None:
        """
        Write the underlying AnnData to an .h5ad file.
        Args:
            path (str): Path to the output .h5ad file.
        Returns:
            None
        """        
        self.adata.write_h5ad(path)
    def write_zarr(self, path: str) -> None:
        """
        Write the underlying AnnData to zarr storage.
        Args:
            path (str): Path to the output zarr storage.
        Returns:
            None
        """     
        self.adata.write_zarr(path)

    def append_on_disk_h5ad(self, path: str, join: str = 'inner', merge: str = 'same', uns_merge: str = 'first') -> None:
        """
        Append this object's data to an existing .h5ad file on disk or create it if it does not exist.
        Args:
            path (str): Path to the target .h5ad file.
        Returns:
            None
        """        
        if not os.path.exists(path):
            self.write_h5ad(path)
            return
        
        # create tmp files and rename 
        base, ext = os.path.splitext(path)
        tmp_path_new = base + '_new_tmp' + ext
        self.write_h5ad(tmp_path_new)

        tmp_path_old = base + '_base_tmp' + ext
        os.rename(path, tmp_path_old)

        try:
            concat_on_disk(
                in_files=[tmp_path_old, tmp_path_new],
                out_file=path,
                axis='obs',
                join=join,
                merge=merge,
                uns_merge=uns_merge,
            )
        except Exception as e:
            os.rename(tmp_path_old, path)
            os.remove(tmp_path_new)
            raise Exception(f'In core.PhotometryData.append_on_disk_h5ad() concat_on_disk: {e}')
        
        os.remove(tmp_path_new)
        os.remove(tmp_path_old)
        return

    # --- convenience views ---
    @property
    def X(self) -> np.ndarray: return cast(np.ndarray, self.adata.X)
    @property
    def ts(self) -> np.ndarray: return self.adata.var["t"].to_numpy()
    @property
    def obs(self) -> pd.DataFrame: return cast(pd.DataFrame, self.adata.obs)
    @property
    def var(self) -> pd.DataFrame: return cast(pd.DataFrame, self.adata.var)
    @property
    def uns(self) -> dict: return dict(self.adata.uns)
    @property
    def n_trials(self) -> int: return self.adata.n_obs
    @property
    def n_times(self) -> int: return self.adata.n_vars

    # --- hidden functions ---
    def _agg(
            self, 
            method: Callable[..., np.ndarray], 
            group_on: list[str] | None, 
            data_cols: list[str], 
            count_col: str | None = None
            ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Aggregate X and selected obs columns over groups.
        Args:
            method (callable): Aggregation function applied along axis 0.
            group_on (list[str]): Columns in obs used to define groups.
            data_cols (list[str]): Observation columns to aggregate.
            count_col (str | None): Optional column name to store group counts.
        Returns:
            tuple[pd.DataFrame, np.ndarray]: Aggregated obs and X arrays.
        """
        X = np.asarray(self.adata.X)
        obs: pd.DataFrame = self.adata.obs

        if group_on is None or len(group_on) == 0:
            new_cols = data_cols if count_col is None else [count_col] + data_cols
            
            X_agg = method(X, axis=0)[np.newaxis, :]
            obs_agg = pd.DataFrame(columns=new_cols, index=[0])
            obs_agg.loc[0, data_cols] = method(obs.loc[:, data_cols], axis=0)
            if count_col is not None: obs_agg.loc[0, count_col] = obs.index.size

        else:
            groups = obs.groupby(group_on, sort=False, observed=False).indices
            n_groups = len(groups)
            new_cols = group_on + data_cols if count_col is None else group_on + [count_col] + data_cols
            
            X_agg = np.empty((n_groups, X.shape[1]), dtype=X.dtype)
            obs_agg = pd.DataFrame(columns=new_cols, index=range(n_groups))

            for i, (gkey, idxs) in enumerate(groups.items()):
                X_agg[i] = method(X[idxs], axis=0)
                obs_agg.loc[i, group_on] = gkey
                obs_agg.loc[i, data_cols] = method(obs.iloc[idxs][data_cols], axis=0)
                if count_col is not None: obs_agg.loc[i, count_col] = len(idxs)

        # clean dtypes
        obs_agg = obs_agg.infer_objects()

        return obs_agg, X_agg

    # --- operations ---
    def downsample(self, factor: int) -> "PhotometryData":
        X_new = downsample_ndarray(self.adata.X, factor=factor, axis=1)
        t_new = downsample_1d(self.var['t'].to_numpy(), factor=factor)
        layers_new = {k : downsample_ndarray(v, factor=factor, axis=1) for k, v in self.adata.layers.items()}
        metadata = self.uns.copy()
        freq_cur = metadata.get('frequency', None)
        if freq_cur is not None: 
            freq_new = freq_cur / factor
            metadata.update({'frequency': freq_new})

        return PhotometryData.from_arrays(
            obs=self.obs,
            data=X_new,
            time_points=t_new,
            layers=layers_new,
            metadata=metadata,
        )

    def combine_obj(
            self, 
            to_append: "PhotometryData" | list["PhotometryData"], 
            inplace: bool = False, 
            join: str = 'inner', 
            merge: str ='same', 
            uns_merge: str = 'same'
            ) -> None | PhotometryData:
        """
        Concatenate the rows of object with one or more PhotometryData objects.
        Args:
            to_append (PhotometryData | list[PhotometryData]): Object(s) to append.
            inplace (bool) : Whether to modify the original object or return new merged one.
            join (str) : How to align values when concatenating. If "outer", the union of the other axis is taken. If "inner", the intersection.
            merge (str) : How elements not aligned to the axis being concatenated along are selected. Currently implemented strategies include:
                None: No elements are kept.
                'same': Elements that are the same in each of the objects.
                'unique': Elements for which there is only one possible value.
                'first': The first element seen at each from each position.
                'only': Elements that show up in only one of the objects.
            uns_merge (str) : How the elements of `.uns` are selected. Uses the same set of strategies as the merge argument, except applied recursively.
        Returns:
            Combined PhotometryData or None depending on ``inplace``.
        """        
        if not isinstance(to_append, list):
            to_append = [to_append]

        adatas = [getattr(self, 'adata')] + [getattr(obj, 'adata') for obj in to_append]
        for i in range(len(adatas)):
            adatas[i].obs = adatas[i].obs.infer_objects()

        merged_adata = ad.concat(
            adatas=adatas,
            axis='obs',
            join=join,
            merge=merge,
            uns_merge=uns_merge,
            index_unique=''
        )
        merged_adata.obs.reset_index(drop=True, inplace=True) # type: ignore
        if inplace:
            self.adata = merged_adata
            return
        else:
            return PhotometryData(merged_adata)

    def collapse(
            self,
            group_on: list[str] | None,
            method: Callable = np.nanmean,
            metrics: dict[str, Callable] = {},
            data_cols: list[str] = [],
            count_col: str | None = 'n'
            ) -> "PhotometryData":
        """
        Collapse trials by grouping obs and aggregating X and selected obs columns.
        Args:
            group_on (list[str]): Columns in obs used to define groups, if None or [] it will fully collapse the data.
            method (callable): Aggregation function for the main X matrix.
            metrics (dict[str, callable]): Additional named aggregation functions stored as layers.
            data_cols (list[str]): Observation columns to aggregate with each method.
            count_col (str | None): Optional column name to store group counts.
        Returns:
            PhotometryData.
        """        
        obs_agg, X_agg = self._agg(method=method, group_on=group_on, data_cols=data_cols, count_col=count_col)

        layers = {}
        for key, func in metrics.items():
            obs_lay, X_lay = self._agg(method=func, group_on=group_on, data_cols=data_cols, count_col=count_col)
            layers[key] = X_lay
            obs_agg = obs_agg.join(obs_lay[data_cols], rsuffix='_' + str(key))
        
        new_obj = PhotometryData.from_arrays(
            obs=obs_agg,
            data=X_agg,
            time_points=self.adata.var['t'],
            layers=layers,
            metadata=self.adata.uns
        )
        return new_obj
    
    def get_window_idxs(
            self, 
            series: np.ndarray, 
            centers: np.ndarray, 
            bounds: tuple[int, int], 
            freq: float
            ) -> np.ndarray:
        """
        Get indexs for windows centered around ``centers``.
        Args:
            series (np.ndarray): Time-like sorted series that windows will be calculated in.
            centers (np.ndarray): The specified centers of the windows in same units as ``series``.
            bounds (np.ndarray): Specify the lower and upper bounds of the windows.
            freq (float): The frequency of ``series``.
        Returns:
            2D np.ndarray of window indexes.
        """
        low, high = bounds
        left_idxs = np.searchsorted(series, centers + low, side='left')

        # calculate minimum window size to ensure stable sizes
        target_len = np.floor((bounds[1] - bounds[0]) * freq).astype(int)
        window_idxs = left_idxs[:, None] + np.arange(target_len)[None, :]
        return window_idxs

    def window(
        self,
        series: np.ndarray,
        centers: np.ndarray | float | int,
        bounds: tuple[float, float],
        freq: float,
        event_cols: list[str] | None = None,
        ) -> "PhotometryData":
        """
        Return new PhotometryData object whose time series has been windowed.
        Args:
            series (np.ndarray): Time-like sorted series that windows will be calculated in.
            centers (np.ndarray): The specified centers of the windows in same units as ``series``.
            bounds (np.ndarray): Specify the lower and upper bounds of the windows.
            freq (float): The frequency of ``series``.
            event_cols (list[str]): Columns of self.adata.obs to be centered to ``centers``.
        Returns:
            2D np.ndarray of window indexes.
        """
        if isinstance(centers, (float, int)): 
            centers = np.full(shape=self.X.shape[0], fill_value=centers, dtype=float)
        series = np.asarray(series)
        centers = np.asarray(centers)

        window_idxs = self.get_window_idxs(series=series, centers=centers, bounds=bounds, freq=freq)
        row_slice = np.arange(self.X.shape[0])[:, None]

        data = self.X[row_slice, window_idxs]
        layers = {}
        for k, layer in self.adata.layers.items():
            layers[k] = layer[row_slice, window_idxs]

        time_points = reconstruct_time_points(bounds=bounds, freq=freq)
        obs = self.adata.obs.copy()
        if event_cols is not None:
            obs[event_cols] = obs[event_cols] - centers[:, None]

        

        out: "PhotometryData" = PhotometryData.from_arrays(
            obs=obs,
            data=data,
            time_points=time_points,
            layers=layers,
            metadata=self.adata.uns,
        )
        return out
    
    # --- plotting ---
    def plot_line(
            self, i: int, 
            ax: matplotlib.axes.Axes | None = None, 
            label_with: list[str] | None = None, 
            err_layer: str | None = None,
            plt_kwargs: dict = {},
        ) -> None:
        """
        Plot a single trial time-series with optional error band.
        Args:
            i (int): Trial index to plot.
            ax (matplotlib.axes.Axes | None): Axis to plot on. Creates a new one if None.
            label_with (list[str] | None): Obs columns used to build the legend label.
            err_layer (str | None): Optional layer name providing per-timepoint error.
            **kwargs: Additional keyword arguments passed to plt.plot.
        Returns:
            None
        """        
        if ax is None: fig, ax = plt.subplots()
        
        x = self.var['t'].to_numpy()
        y = self.X[i, :]
        row = self.obs.iloc[i]

        if label_with is not None:
            vals = row[label_with].astype(str).to_list()
            name_val_pairs = [k + ': ' + v for k, v in zip(label_with, vals)]
            label = ', '.join(name_val_pairs)
        else:
            label = None

        ax.plot(x, y, label=label, **plt_kwargs)
        if err_layer is not None:
            yerr = self.adata.layers[err_layer][i]
            ax.fill_between(x, y + yerr, y - yerr, alpha=0.3)

    def plot_set(
            self, 
            subset: list[bool | int], 
            ax: matplotlib.axes.Axes = None,
            title: str = None, 
            label_with: list[str] = None, 
            err_layer: str = None, 
            plt_kwargs: dict = {},
            ) -> None:
        """
        Plot a set of trials given a boolean or index subset.
        Args:
            subset (list[bool | int]): Boolean mask or indices selecting trials.
            ax (matplotlib.axes.Axes | None): Axis to plot on. Creates a new one if None.
            title (str | None): Optional title for the axis.
            label_with (list[str] | None): Obs columns used to build legend labels.
            err_layer (str | None): Optional layer name providing per-timepoint error.
            **kwargs: Additional keyword arguments passed to plot_line.
        Returns:
            None
        """        
        idxs = np.arange(self.n_trials)[subset]
        if ax is None: fig, ax = plt.subplots()
        for i in idxs:
            self.plot_line(i, ax=ax, label_with=label_with, err_layer=err_layer, plt_kwargs=plt_kwargs)
        if label_with is not None: plt.legend()
        if title is not None: ax.set_title(title)

    def plot_groups(
            self, 
            group_on: list[str], 
            label_with: list[str] | None = None, 
            err_layer: str | None = None, 
            save_dir: str | None = None,
            save_ext: str = '.png',
            save_dpi: int = 140, 
            plt_kwargs: dict = {},
            ax = None,
            ):
        """
        Plot trials grouped by observation columns.
        Args:
            group_on (list[str]): Obs columns used to define groups.
            label_with (list[str] | None): Obs columns used to build legend labels.
            err_layer (str | None): Optional layer name providing per-timepoint error.
            save_dir (str | None): Optional output directory to save figures to.
            save_ext (str): What format to save images in.
            save_dpi (int): What dpi to save images at.
            **kwargs: Additional keyword arguments passed to plot_set.
        Returns:
            None
        """        
        groups = self.obs.groupby(group_on).indices
        for gkey, idxs in groups.items():
            if not isinstance(gkey, tuple): gkey=[gkey]
            title = ', '.join([f'{name}: {val}' for name, val in zip(group_on, gkey)])
            self.plot_set(subset=idxs, label_with=label_with, title=title, err_layer=err_layer, plt_kwargs=plt_kwargs, ax=ax)
            if save_dir is not None:
                file_name = '_'.join([f'{name}-{val}' for name, val in zip(group_on, gkey)]) + save_ext
                plt.savefig(os.path.join(save_dir, file_name), dpi=save_dpi)
            plt.show()

    # --- convienience ---
    def filter_rows(self, mask: np.ndarray, inplace: bool = False) -> "None | PhotometryData":
        """
        Filter rows (trials) using a boolean mask.
        Args:
            mask (np.ndarray): Boolean array of length n_trials.
            inplace (bool): If True, modify in place. If False, return a new object.
        Returns:
            None | PhotometryData: New filtered object if inplace is False, else None.
        """        
        if inplace:
            self.adata = self.adata[mask, :].copy()
        else:
            return PhotometryData(self.adata[mask, :].copy())

    def add_obs_columns(self, add_from: dict[str, Any], keys: list[str] | None = None) -> None:
        """
        Add columns to obs from a dictionary.
        Args:
            add_from (dict[str, Any]): Mapping from column names to values.
            keys (list[str] | None): Keys from add_from to add. Defaults to all keys.
        Returns:
            None
        """        
        keys = list(add_from.keys()) if keys is None else keys
        for k in keys:
            self.adata.obs[k] = add_from[k]

    def add_metadata(self, add_from: dict[str, Any], keys: list[str] | None = None) -> None:
        """
        Add entries to the .uns metadata dictionary.
        Args:
            add_from (dict[str, Any]): Mapping from keys to metadata values.
            keys (list[str] | None): Keys from add_from to add. Defaults to all keys.
        Returns:
            None
        """        
        keys = list(add_from.keys()) if keys is None else keys
        for k in keys:
            self.adata.uns[k] = add_from[k]

    def drop_obs_columns(self, to_drop: list[str]) -> None:
        """
        Drop observation columns from obs.
        Args:
            to_drop (list[str]): Column names to drop.
        Returns:
            None
        """        
        self.adata.obs = self.obs.drop(to_drop, errors='ignore', axis=1)

    def get_text_value_counts(self, col: str) -> str:
        """
        Get a string summary of value counts for a column in obs.
        Args:
            col (str): Column name in obs.
        Returns:
            str: Comma-separated summary of value counts.
        """
        vc = self.adata.obs[col].value_counts(dropna=False)
        return ", ".join(f"{k}: {v}" for k, v in vc.items())
    
    def trials_to_long_df(
            self, 
            layer: str | None = None,
            err_layer: str | None = None,
            obs_cols: list[str] | None = None, 
            downsample: int | None = None,

            ) -> pd.DataFrame:
        '''
        Translate trial data to long dataframe format. Mostly for graphing.

        Parameters
        ----------
        layer : str | None, optional
            Which layer to export, if none `self.X` used, by default None.
        err_layer : str | None, optional
            Which layer, if any, to add as an error col.
        obs_cols : list[str] | None, optional
            Which columns from `obs` to include, by default None.
        downsample : int | None, optional
            How much to downsample the time series data if at all, by default None.

        Returns
        -------
        pd.DataFrame
            DataFrame containing columns, trial_id, obs_cols specified, time_idx, signal, time, err_layer in long format.
        '''
        obj = self if downsample is None else self.downsample(downsample)
        X = obj.adata.layers[layer] if layer is not None else obj.adata.X

        def _make_long_fast(arr: np.ndarray, obj: PhotometryData, obs_cols: list[str] | None = None, value_name: str = 'signal') -> pd.DataFrame:
                n_trials, n_time = arr.shape

                trial_ids = obj.adata.obs_names.to_numpy()

                out = {
                    "trial_id": np.repeat(trial_ids, n_time),
                    "time_idx": np.tile(np.arange(n_time), n_trials),
                    value_name: np.asarray(arr).reshape(-1),
                }

                if obs_cols is not None:
                    obs_df = obj.adata.obs[obs_cols]
                    for col in obs_cols:
                        out[col] = np.repeat(obs_df[col].to_numpy(), n_time)

                return pd.DataFrame(out)    

        long_signal = _make_long_fast(X, obj=obj, obs_cols=obs_cols)

        long_signal["time_idx"] = long_signal["time_idx"].astype(int)
        long_signal["time"] = obj.ts[long_signal["time_idx"]]

        if err_layer is not None:
            E = obj.adata.layers[err_layer]
            long_err = _make_long_fast(E, obj=obj, obs_cols=None, value_name=err_layer)
            long_signal = long_signal.join(long_err.filter([err_layer]), how='left')

        return long_signal