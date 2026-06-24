"""Functional mixed-model helpers backed by fastFMM."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import re

from plotnine import ggplot
from ..utils import graphing

if TYPE_CHECKING:
    from ..core.PhotometryData import PhotometryData

############################
#region --- RESULT CLASS ---
############################
@dataclass(slots=True)
class FMMResult:
    """Wrapper around fastFMM's multi-indexed result table."""

    df: pd.DataFrame
    formula: str | None = None

    # --- dunders ---
    def __post_init__(self) -> None:
        """Validate the fastFMM result table."""
        if not isinstance(self.df.columns, pd.MultiIndex):
            raise ValueError("FMMResult.df must have MultiIndex columns.")

    def __str__(self) -> str:
        """Return the string representation of the result table."""
        return self.df.__str__()

    def __repr__(self) -> str:
        """Return the interactive representation of the result table."""
        return self.df.__repr__()

    # --- properties ---
    @property
    def column_groups(self) -> list[str]:
        """Top-level column groups in the result table."""
        return self.df.columns.get_level_values(0).unique().to_list()

    @property
    def terms(self) -> list[str]:
        """Model coefficient terms, excluding metadata groups."""
        return [
            col for col in self.column_groups
            if col not in {"time", "AIC"}
        ]

    @property
    def time(self) -> pd.Series:
        """Time values indexed like the result table."""
        if ("time", "time") in self.df.columns:
            return self.df[("time", "time")]

        if "time" in self.column_groups:
            time_df = self.df["time"]
            if time_df.shape[1] == 1:
                out = time_df.iloc[:, 0]
                out.name = "time"
                return out

        raise KeyError("FMMResult does not contain a time column.")

    @property
    def coefficients(self) -> pd.DataFrame:
        """All coefficient-term columns."""
        if not self.terms:
            return self.df.iloc[:, 0:0]

        return self.df.loc[:, pd.IndexSlice[self.terms, :]]

    @property
    def aic(self) -> pd.DataFrame:
        """AIC columns returned by fastFMM."""
        if "AIC" not in self.column_groups:
            raise KeyError("FMMResult does not contain AIC columns.")
        return pd.DataFrame(self.df["AIC"])

    # --- I/O ---
    def to_csv(
            self,
            path: str | PathLike[str],
            **kwargs,
            ) -> None:
        """Write the underlying multi-indexed result table to CSV.

        Parameters
        ----------
        path : str or PathLike[str]
            Output CSV path.
        **kwargs
            Additional keyword arguments passed to ``DataFrame.to_csv``.
        """
        self.df.to_csv(path, **kwargs)

    @classmethod
    def from_csv(
            cls,
            path: str | PathLike[str],
            formula: str | None = None,
            **kwargs,
            ) -> "FMMResult":
        """Read an ``FMMResult`` from a CSV written by ``to_csv``.

        Parameters
        ----------
        path : str or PathLike[str]
            Input CSV path.
        formula : str or None, default=None
            Formula associated with the result.
        **kwargs
            Additional keyword arguments passed to ``pandas.read_csv``.

        Returns
        -------
        FMMResult
            Loaded result object.
        """
        read_kwargs = {"header": [0, 1], "index_col": 0} | kwargs
        df = pd.read_csv(path, **read_kwargs)
        return cls(df=df, formula=formula)

    # --- access ---
    def term(self, name: str, include_time: bool = True) -> pd.DataFrame:
        """Return all statistics for one model term.

        Parameters
        ----------
        name : str
            Model term name.
        include_time : bool, default=True
            If ``True``, prepend a ``time`` column.

        Returns
        -------
        pd.DataFrame
            Statistics for the selected term.

        Raises
        ------
        KeyError
            If ``name`` is not a model term.
        """
        if name not in self.terms:
            raise KeyError(f"Unknown FMM term: {name}")

        out: pd.DataFrame = self.df[name].copy()
        if include_time:
            out.insert(0, "time", self.time.to_numpy())
        return out

    def stat(self, name: str) -> pd.DataFrame:
        """Return one statistic across all model terms.

        Parameters
        ----------
        name : str
            Statistic name in the second column-index level.

        Returns
        -------
        pd.DataFrame
            Statistic values across all coefficient terms.
        """
        return pd.DataFrame(self.coefficients.xs(name, level=1, axis=1).copy())

    # --- export ---
    def to_long(
            self,
            only_terms: list[str] | None = None,
            stat_map: dict[str, str] | None = None,
            ) -> pd.DataFrame:
        """Convert the FMM result table to long format.

        Parameters
        ----------
        only_terms : list[str] or None, default=None
            Model terms to include. If ``None``, all coefficient terms are
            included.
        stat_map : dict[str, str] or None, default=None
            Mapping from output column names to source statistic names in the
            fastFMM table.

        Returns
        -------
        pd.DataFrame
            Long dataframe with time, term, and selected statistics.
        """
        if stat_map is None:
            stat_map = {
                "value": "beta",
                "lower": "lower",
                "upper": "upper",
                "lower_joint": "lower_joint",
                "upper_joint": "upper_joint",
            }

        coefficients = self.coefficients.rename_axis(
            index="time_idx",
            columns=["term", "stat"],
        )

        try:
            long_df = coefficients.stack(level="term", future_stack=True)
        except TypeError:
            long_df = coefficients.stack(level="term")

        long_df = long_df.reset_index()
        long_df.columns.name = None

        rename_map = {
            source: target
            for target, source in stat_map.items()
            if source in long_df.columns
        }
        long_df = long_df.rename(columns=rename_map)

        time_df = self.time.rename("time").rename_axis("time_idx").reset_index()
        long_df = long_df.merge(time_df, on="time_idx", how="left")

        keep_cols = ["time_idx", "time", "term"] + [
            col for col in stat_map
            if col in long_df.columns
        ]

        only_terms = self.terms if only_terms is None else only_terms
        keep_terms = long_df['term'].isin(only_terms)

        return long_df.loc[keep_terms, keep_cols]

    # --- plotting ---
    def plot(
            self,
            only_terms: list[str] | None = None,
            line_kwargs: dict = {},
            hline_kwargs: dict = {},
            ribbon_inner_kwargs: dict = {},
            ribbon_outer_kwargs: dict = {},
            theme_kwargs: dict = {},
            ) -> ggplot:
        """Plot model terms and confidence bands.

        Parameters
        ----------
        only_terms : list[str] or None, default=None
            Model terms to plot. If ``None``, all coefficient terms are plotted.
        line_kwargs : dict, default={}
            Keyword arguments forwarded to line geoms.
        hline_kwargs : dict, default={}
            Keyword arguments forwarded to horizontal reference-line geoms.
        ribbon_inner_kwargs : dict, default={}
            Keyword arguments forwarded to inner confidence-band ribbon geoms.
        ribbon_outer_kwargs : dict, default={}
            Keyword arguments forwarded to outer confidence-band ribbon geoms.
        theme_kwargs : dict, default={}
            Keyword arguments forwarded to the plot theme helper.

        Returns
        -------
        ggplot
            Plot object.
        """

        long_df = self.to_long(only_terms=only_terms)

        p = graphing.plot_FMM_result(
            long_df,
            line_kwargs=line_kwargs,
            hline_kwargs=hline_kwargs,
            ribbon_inner_kwargs=ribbon_inner_kwargs,
            ribbon_outer_kwargs=ribbon_outer_kwargs,
            theme_kwargs=theme_kwargs,
        )

        return p

#endregion

#######################
#region --- FMM API ---
#######################
def run_fastFMM(
        data: PhotometryData,
        formula: str,
        factor_cols: dict[str, str | None] = {},
        parallel: bool = True,
        family: str = "gaussian",
        analytic: bool = True,
        var: bool = True,
        silent: bool = False,
        argvals: list[int] | None = None,
        nknots_min: int | None = None,
        nknots_min_cov: int | None = 35,
        smooth_method: str = "GCV.Cp",
        splines: str = "tp",
        design_mat: bool = False,
        residuals: bool = False,
        n_boots: int = 500,
        seed: int = 1,
        subj_id: str | None = None,
        n_cores: int | None = None,
        caic: bool = False,
        randeffs: bool = False,
        non_neg: int = 0,
        MoM: int = 1,
        concurrent: bool = False,
        impute_outcome: bool = False,
        override_zero_var: bool = False,
        unsmooth: bool = False,
        ) -> FMMResult:
    """Run a fastFMM model on trial-wise photometry data.

    This is a wrapper around ``fast-fmm-rpy2`` that converts
    `PhotometryData` into the wide table expected by fastFMM and wraps the
    multi-indexed output table in `FMMResult`.

    Parameters
    ----------
    data : PhotometryData
        Photometry data used to fit the functional mixed model.
    formula : str
        Formula passed to fastFMM. It must begin with a signal column prefix
        present in the exported wide dataframe.
    factor_cols : dict[str, str or None], default={}
        Metadata columns to convert to R factors. Values specify optional
        reference levels.
    parallel : bool, default=True
        Whether fastFMM should run in parallel.
    family : str, default='gaussian'
        Model family passed to fastFMM.
    analytic : bool, default=True
        Whether to use analytic inference instead of bootstrap inference.
    var : bool, default=True
        Whether to include within-timepoint variance in the model.
    silent : bool, default=False
        Whether to suppress model output.
    argvals : list[int] or None, default=None
        Functional-domain indexes used in the model. ``None`` uses all points.
    nknots_min : int or None, default=None
        Minimum knots for coefficient smoothing.
    nknots_min_cov : int or None, default=35
        Minimum knots for covariance smoothing.
    smooth_method : str, default='GCV.Cp'
        Smoothing-parameter selection method.
    splines : str, default='tp'
        Spline type used by fastFMM.
    design_mat : bool, default=False
        Whether to return the design matrix.
    residuals : bool, default=False
        Whether to save residuals from the unsmoothed LME.
    n_boots : int, default=500
        Number of bootstrap samples.
    seed : int, default=1
        Seed used by fastFMM bootstrap routines.
    subj_id : str or None, default=None
        Column containing subject IDs.
    n_cores : int or None, default=None
        Number of cores used for parallelization.
    caic : bool, default=False
        Whether to calculate CAIC.
    randeffs : bool, default=False
        Whether to return random-effect estimates.
    non_neg : int, default=0
        Non-negativity constraint mode passed to fastFMM.
    MoM : int, default=1
        Method-of-moments estimator setting.
    concurrent : bool, default=False
        Whether to fit a concurrent model.
    impute_outcome : bool, default=False
        Whether to impute missing outcome values with FPCA.
    override_zero_var : bool, default=False
        Whether to proceed when exported columns have zero variance.
    unsmooth : bool, default=False
        Whether to return raw unsmoothed coefficient and variance estimates.

    Returns
    -------
    FMMResult
        Wrapped fastFMM result table.

    Raises
    ------
    ValueError
        If ``formula`` does not begin with a signal prefix present in the
        exported wide dataframe.
    """
    # lazy import fast-fmm-rpy2
    from rpy2.robjects import r
    from rpy2.rinterface import NULL  # type: ignore
    from fast_fmm_rpy2.ingest import pass_pandas_to_r
    from fast_fmm_rpy2.fmm_run import fui
    from fast_fmm_rpy2.plot_fui import plot_fui

    # set up function
    def _factor_r_var(r_var: str, col: str, ref_lvl: str | None) -> None:
        """Factor one R dataframe column and optionally set a reference level."""
        r(f'{r_var}[,"{col}"] = factor({r_var}[,"{col}"], ordered = "FALSE")')
        if ref_lvl is not None:
            r(f'{r_var}[,"{col}"] = relevel({r_var}[,"{col}"], ref = "{ref_lvl}")')

    # step 0: convert Photometry data to dataframe
    # get necessary cols from formula
    keep_cols = (
        pd.Series(re.split(r"[\+\*\|:~]", formula))
        .str.replace(r'[\(\)]', '', regex=True)
        .str.strip()
        .unique()
        .tolist()
    )
    obs_cols = [col for col in keep_cols if col in data.obs]

    df = data.trials_to_wide_df(
        layer=None,
        obs_cols=obs_cols,
        signal_prefix='photometry',
        downsample=None,
    )

    # step 1: validate inputs
    # convert some None types to R NULL type
    def _none_to_null(val):
        """Convert Python None to R NULL."""
        if val is None: return NULL
        else: return val

    argvals = _none_to_null(argvals)
    nknots_min = _none_to_null(nknots_min)
    subj_id = _none_to_null(subj_id)
    n_cores = _none_to_null(n_cores)

    # ensure formula starts with signal prefix
    potential_prefixes = (
        df.columns[
            df.columns.str.contains(".", regex=False)
        ]
        .str.split(".")
        .str[0].unique()
        .to_list()
    )
    formula_valid = np.asarray([formula.startswith(prefix) for prefix in potential_prefixes]).any()
    if not formula_valid:
        raise ValueError(f'Formula ({formula}) does not begin with any potential signal prefixes ({potential_prefixes})')

    # step 2: pass dataframe to R and factor vars
    pass_pandas_to_r(df, r_var_name='dat')
    for col, ref_lvl in factor_cols.items():
        _factor_r_var('dat', col, ref_lvl)

    # step 3: run fastFMM
    model = fui(
        csv_filepath=None,
        r_var_name='dat',
        formula=formula,
        parallel=parallel,
        family=family,
        analytic=analytic,
        var=var,
        silent=silent,
        argvals=argvals,
        nknots_min=nknots_min,
        nknots_min_cov=nknots_min_cov,
        smooth_method=smooth_method,
        splines=splines,
        design_mat=design_mat,
        residuals=residuals,
        n_boots=n_boots,
        seed=seed,
        subj_id=subj_id,
        n_cores=n_cores,
        caic=caic,
        randeffs=randeffs,
        non_neg=non_neg,
        MoM=MoM,
        concurrent=concurrent,
        impute_outcome=impute_outcome,
        override_zero_var=override_zero_var,
        unsmooth=unsmooth,
    )

    # step 4: plot results and retrieve data
    coeff_figs, model_res = plot_fui(model, return_data=True) # type: ignore

    # step 5: coerce results into multi indexed dataframe
    first_df = next(iter(model_res.values()))
    shared_idx = first_df.index.astype(int)

    combined: dict[str, pd.DataFrame] = {
        coeff : data.set_index(shared_idx).drop(columns='s') for coeff, data in model_res.items()
    }
    combined['time'] = pd.DataFrame({'time':data.ts}).set_index(shared_idx)
    combined['AIC'] = model.getbyname('aic').set_index(shared_idx)

    out = pd.concat(combined, axis=1)
    return FMMResult(out, formula=formula)

#endregion
