from typing import Any

import numpy as np
import pandas as pd


def run_fastFMM(
        df: pd.DataFrame,
        formula: str,
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
        ) -> pd.DataFrame:
    """Run the fastFMM model using the specified formula and data. 

    An additional wrapper for the ``fast-fmm-rpy2`` package made to have
    simplified input and output structures

    Args:
        formula (str): Formula to use in the fastFMM model.
        family (str, optional): Family to use in the fastFMM model.
            Defaults to `"gaussian"`.
        analytic (bool, optional): Whether to use analytic inference instead of
            bootstrap. Defaults to True.
        var (bool, optional): Whether to include the within-timepoint variance in
            the model. Defaults to True.
        parallel (bool, optional): Whether to run the model in parallel.
            Defaults to True.
        silent (bool, optional): Whether to suppress model output.
            Defaults to False.
        argvals (list[int] | None, optional): Indices of the functional domain to
            use in the model. Defaults to None, meaning all points are used.
        nknots_min (int | None, optional): Minimum number of knots in the penalized
            smoothing for the regression coefficients. Defaults to None, which uses
            `L / 2` where `L` is the dimension of the functional domain.
        nknots_min_cov (int | None, optional): Minimum number of knots in the
            penalized smoothing for the covariance matrices. Defaults to 35.
        smooth_method (str, optional): Method used to select the smoothing
            parameter in step 2. Defaults to `"GCV.Cp"`.
        splines (str, optional): Type of spline to use in the model.
            Defaults to `"tp"`.
        design_mat (bool, optional): Whether to return the design matrix.
            Defaults to False.
        residuals (bool, optional): Whether to save residuals from the unsmoothed
            LME. Defaults to False.
        n_boots (int, optional): Number of samples to use for bootstrap inference.
            Defaults to 500.
        seed (int, optional): Numeric seed used to ensure bootstrap replicates are
            correlated across functional domains for certain bootstrap approaches.
            Defaults to 1.
        subj_id (str | None, optional): Name of the variable containing subject
            IDs. Defaults to None.
        n_cores (int | None, optional): Number of cores to use for parallelization.
            If not specified, defaults to three-fourths of the detected cores.
            Defaults to None.
        caic (bool, optional): Whether to calculate CAIC. Defaults to False.
        randeffs (bool, optional): Whether to return random effect estimates.
            Defaults to False.
        non_neg (int, optional): Non-negativity constraint mode.
            `0` means no non-negativity constraints,
            `1` means non-negativity constraints on every coefficient for variance,
            `2` means non-negativity on the average of coefficients for one
            variance term. Defaults to 0.
        MoM (int, optional): Method-of-moments estimator setting. Defaults to 1.
        concurrent (bool, optional): Whether to fit a concurrent model.
            Defaults to False.
        impute_outcome (bool, optional): Whether to impute missing outcome values
            with FPCA. Defaults to False.
        override_zero_var (bool, optional): Whether to proceed with model fitting
            if some columns have zero variance. This can be useful when individual
            columns have zero variance but interactions have non-zero variance.
            Defaults to False.
        unsmooth (bool, optional): Whether to return raw coefficient and variance
            estimates without smoothing. Defaults to False.

    Returns:
        object: The fitted fastFMM model.

    Notes:
        
    """
    # lazy import fast-fmm-rpy2
    from rpy2.rinterface import NULL  # type: ignore
    from fast_fmm_rpy2.ingest import pass_pandas_to_r
    from fast_fmm_rpy2.fmm_run import fui
    from fast_fmm_rpy2.plot_fui import plot_fui

    # step 0: validate inputs
    # convert some None types to R NULL type
    def _none_to_null(val):
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
    
    # step 1: pass dataframe to R
    pass_pandas_to_r(df, r_var_name='py_dat')

    # step 2: run fastFMM
    model = fui(
        csv_filepath=None,
        r_var_name='py_dat',
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

    # step 3: plot results and retrieve data
    coeff_figs, model_res = plot_fui(model, return_data=True) # type: ignore

    # step 4: coerce results into multi indexed dataframe
    first_df = next(iter(model_res.values()))
    shared_idx = first_df.index.astype(int)

    combined: dict[str, pd.DataFrame] = {
        coeff : data.set_index(shared_idx).drop(columns='s') for coeff, data in model_res.items()
    }
    combined['AIC'] = model.getbyname('aic').set_index(shared_idx)

    out = pd.concat(combined, axis=1)
    return out




