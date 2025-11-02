"""Data processing tool-kit A module containing functions for analysis and manipulation
of data-sets, mainly using pandas. Includes:

- decorators
- audit
- transforms cleaning
- descriptive stats
- transforms analysis
- inout
"""

import functools
import itertools
import logging
import os
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy
import shap
from dstk import dviztk, slibtk
from gensim.models import Word2Vec
from sklearn import (
    cluster,
    decomposition,
    inspection,
    metrics,
    model_selection,
    neighbors,
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# DATA ANALYSIS ########################################################################


def value_counts_pct(
    ser: pd.Series, dropna=False, fmt=True, ascending=False
) -> pd.DataFrame:
    """Take in a series and return the value counts, pct breakdown and cumulative
    breakdown as a pct formatted DataFrame."""
    ser_vc = ser.value_counts(dropna=dropna)
    ser_vc.index.name = ser.name
    return cum_pcts(ser_vc, fmt, ascending=ascending)


def cum_pcts(
    ser: pd.Series, fmt=True, total=False, ascending=False, precision: int = 0
) -> pd.DataFrame:
    """Adds a pct breakdown and cumulative pct breakdown to a series."""
    ser = ser.sort_values(ascending=ascending)
    df = pd.concat([ser, ser / ser.sum(), ser.cumsum() / ser.sum()], axis=1)
    cols = ["total", "pct", "cumulative"]
    df.columns = cols
    df.index = df.index.astype(str)
    if total:
        df.loc["Totals"] = df.sum()
    if fmt:
        df = pd.concat(
            [
                df.iloc[:, 0].apply(f"{{:,.{precision}f}}".format),
                df[cols[1:]].applymap("{:.0%}".format),
            ],
            axis=1,
        )
    if total:
        df.iloc[-1, -1] = np.NaN
    return df


def flatten_columns(df):
    return ["_".join(filter(bool, c)) for c in df.columns]


def row_pcts(df: pd.DataFrame, flatten: bool = False) -> pd.DataFrame:
    """Append row wise percentage version of the DataFrame under a multi- index."""
    pcts = df.div(df.sum(1), 0)
    totals = df.sum() / df.sum().sum()
    index = pcts / totals
    df = pd.concat([df, pcts, index], axis=1, keys=["point", "pct", "index"])
    if flatten:
        df = flatten_columns(df)
    return df


def column_pcts(df: pd.DataFrame, flatten: bool = False) -> pd.DataFrame:
    """Append col wise percentage version of the DataFrame under a multi- index."""
    pcts = df.div(df.sum(), 1)
    totals = df.sum(1) / df.sum(1).sum()
    index = pd.DataFrame(
        pcts.values / totals.values.reshape(-1, 1),
        columns=pcts.columns,
        index=pcts.index,
    )
    df = pd.concat([df, pcts, index], axis=1, keys=["point", "pct", "index"])
    if flatten:
        df = flatten_columns(df)
    return df.fillna(0)


def make_index(
    df: pd.DataFrame,
    grp_var: str,
    cat_var: str,
    cont_var: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    take tidy DataFrame and make within group pct breakdown column and index for the
    groups in grp_var
    Args:
        df: tidy DataFrame
        grp_var: col name of groups
        cat_var: col name of categories in breakdown
        cont_var: continuious variable

    Returns:
        df: tidy DataFrame
    """
    cols = ["pct", "pct_all", "index", "variable"]
    pct, pct_all, index, variable = [f"{prefix}_{c}" for c in cols] if prefix else cols
    df[pct] = df.groupby(grp_var)[cont_var].transform(lambda x: x / float(x.sum()))
    totals = (
        (df.groupby(cat_var)[cont_var].sum() / df[cont_var].sum())
        .to_frame(pct_all)
        .reset_index()
    )
    df = df.merge(totals, on=cat_var, how="left")
    df[index] = df[pct] / df[pct_all]
    df[variable] = cat_var
    return df


def make_indexes(
    df: pd.DataFrame, grp_var: str, cat_var: str, cont_var: str, split_var: str
) -> pd.DataFrame:
    return pd.concat(
        [
            make_index(df.query(f"{split_var} == {i}"), grp_var, cat_var, cont_var)
            for i in tqdm(df[split_var].unique())
        ]
    )


def sum_make_lag_lfl_indexes(
    df: pd.DataFrame,
    grp_var: str,
    cat_var: str,
    cont_var: str,
    date: str,
    lag_func: str = "year_period",
) -> pd.DataFrame:
    on = [grp_var, cat_var]
    df = sum_make_lag_lfl(df, cont_var, on, date, lag_func=lag_func)
    return make_indexes(
        df, grp_var=grp_var, cat_var=cat_var, cont_var=cont_var, split_var=date
    )


def sum_make_index(
    df: pd.DataFrame,
    grp_var: str,
    cat_var: str,
    cont_var: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    gb = df.groupby([grp_var, cat_var])[cont_var].sum().reset_index()
    return gb.pipe(make_index, grp_var, cat_var, cont_var, prefix)


def quantile_bins(ser: pd.Series, precision=0) -> pd.Series:
    """Bin a series into quintiles returning the bins as a series of type str."""
    q = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    decimal_places = functools.partial(slibtk.re_n_decimal_places, n=precision)
    bins = (
        pd.qcut(ser, q=q, labels=np.arange(len(q) - 1) + 1)
        .astype(str)
        .apply(slibtk.re_no_decimal_places)
        + " "
        + pd.qcut(ser, q=q).astype(str).apply(decimal_places)
    )
    return bins.replace("nan nan", np.NAN)


def category_contribution_to_group_lfl_if_no_chg(
    df: pd.DataFrame,
    cat_var: str,
    cat_val: str,
    value: str,
    grp_var: str,
    grp_val: str,
    date_var: str,
    date_val: str,
    lag_func: str = "year",
) -> pd.DataFrame:
    cond = (
        (df[cat_var] == cat_val) & (df[grp_var] == grp_val) & (df[date_var] == date_val)
    )
    df[f"{value}_sen"] = np.where(cond, df[f"{value}_l1"], df[value])
    lfl = sum_make_lag_lfl(
        df, [f"{value}", f"{value}_sen"], grp_var, date_var, lag_func=lag_func
    )
    lfl = lfl[[grp_var, f"{value}_lfl", f"{value}_sen_lfl"]].dropna()
    lfl["gap"] = lfl[f"{value}_sen_lfl"] - lfl[f"{value}_lfl"]
    lfl["feature"] = cat_val
    return lfl


def category_contribution_to_group_lfl_vs_control(
    df: pd.DataFrame,
    cont_var: str,
    cat_var: str,
    cat_val: str,
    grp_var: str,
    grp_val: str,
    date_var: str,
    date_val: str,
    lag_func: str = "year",
) -> pd.DataFrame:
    """Impact of lfl if group had experience the same growth as control group in one
    category of a variable must already have lfl value in DataFrame."""
    # merge the spend_lfl_ctrl of control
    cond_grp = ~(df[grp_var] == grp_val)
    lfl_ctrl = df[cond_grp][[date_var, cat_var, f"{cont_var}_lfl"]].rename(
        {f"{cont_var}_lfl": f"{cont_var}_lfl_sen"}, axis=1
    )
    df = df.merge(lfl_ctrl, on=[date_var, cat_var], how="left")
    # make spend_sen as (spend_l1 * spend_lfl) except for group use spend_lfl_ctrl
    spend_if_group = df[f"{cont_var}_l1"] * (df[f"{cont_var}_lfl_sen"] + 1)
    cond = (df[cat_var] == cat_val) & (df[date_var] == date_val)
    df[f"{cont_var}_sen"] = np.where(cond, spend_if_group, df[f"{cont_var}"])
    # suming up and recalculating the lfl
    lfl = sum_make_lag_lfl(
        df, [f"{cont_var}", f"{cont_var}_sen"], grp_var, date_var, lag_func=lag_func
    )
    lfl = lfl[[grp_var, f"{cont_var}_lfl", f"{cont_var}_sen_lfl"]].dropna()
    lfl["impact"] = lfl[f"{cont_var}_sen_lfl"] - lfl[f"{cont_var}_lfl"]
    lfl["feature"] = cat_val
    return lfl


def bin_categories_lt_thresh(
    df: pd.DataFrame,
    thresh: float,
    grp_var: str,
    cat_var: str,
    cont_var: str,
    non_cat_vars: list[str],
) -> pd.DataFrame:
    """Bin categories below a certian pct contribution of spend."""
    group_on = [grp_var, cat_var]
    spend_by_range_brand = df.groupby(group_on)[cont_var].sum().reset_index()
    spend_by_range_brand = make_index(spend_by_range_brand, grp_var, cat_var, cont_var)[
        group_on + ["pct"]
    ]
    df = df.merge(spend_by_range_brand, on=group_on)
    brands_other = (
        df.query(f"pct < {thresh}")
        .groupby([grp_var] + non_cat_vars)
        .sum()
        .reset_index()
    )
    brands_other[cat_var] = "other"
    df = pd.concat([df.query(f"pct >= {thresh}"), brands_other])
    return df


def lag_year_period(ser: pd.Series) -> pd.Series:
    """Parse out year and make log of two part string date column:

    <year>-<qtr>
    """
    df = ser.str.split("-", expand=True)
    df[0] = df[0].astype(int) + 1
    lag = df[0].astype(str) + "-" + df[1]
    lag = lag.rename(ser.name)
    return lag


def lag_datetime(ser: pd.Series) -> pd.Series:
    return ser + pd.DateOffset(years=1)


def make_lag(
    df: pd.DataFrame, values, on, date: str, lag_func: str = "year_period"
) -> pd.DataFrame:
    """Add one year lagged version of passed value columns to a DataFrame and return it
    in order to create a lag you shift the current date column one year forward.

    Note: you do not need to include `date` in `on`, but including it should cause bug
    """
    func = {
        "year_period": lag_year_period,
        "datetime": lag_datetime,
        "year": lambda x: x + 1,
        "year_l2": lambda x: x + 2,
    }[lag_func]
    values, on = slibtk.listify(values), slibtk.listify(on)
    on = slibtk.uniqueify(on + [date])
    lag = df[values + on]
    lag[date] = func(lag[date])
    for col in values:
        lag = lag.rename({col: f"{col}_l1"}, axis=1)
    df = df.merge(lag, on=on, how="left")
    for col in values:
        df[f"{col}_d1"] = df[col] - df[f"{col}_l1"]
    return df


def make_lfl(df: pd.DataFrame, values) -> pd.DataFrame:
    """Add lfl columns for the passed values.

    called after make_lag()
    """
    values = slibtk.listify(values)
    for v in values:
        df[f"{v}_lfl"] = (df[f"{v}"] - df[f"{v}_l1"]) / df[f"{v}_l1"]
    return df


def make_lag_lfl(
    df: pd.DataFrame, values, on, date: str, lag_func: str = "year_period"
) -> pd.DataFrame:
    """Note: you do not need to include `date` in `on`, but including it should cause
    bug.
    see: make_lag()"""
    return make_lfl(make_lag(df, values, on, date, lag_func), values).sort_values(date)


def make_lag_lfl_indexes(
    df: pd.DataFrame,
    grp_var: str,
    cat_var: str,
    cont_var: str,
    date: str,
    lag_func: str = "year_period",
) -> pd.DataFrame:
    on = [grp_var, cat_var]
    df = make_lag_lfl(df, cont_var, on, date, lag_func=lag_func)
    return make_indexes(
        df, grp_var=grp_var, cat_var=cat_var, cont_var=cont_var, split_var=date
    )


def sum_make_lag_lfl(
    df: pd.DataFrame, values, on, date: str, lag_func: str = "year_period"
) -> pd.DataFrame:
    on, values = slibtk.listify(on), slibtk.listify(values)
    return (
        df.groupby([date] + on)[values]
        .sum()
        .reset_index()
        .pipe(make_lag_lfl, values, on, date, lag_func=lag_func)
    )


def make_gap(df: pd.DataFrame, ctrl_var: str, date: str, lfl_var: str) -> pd.DataFrame:
    df = df.set_index([date, ctrl_var])[lfl_var].unstack().reset_index().dropna()
    df["gap"] = df[True] - df[False]
    return df


# OUTLIER DETECTION ####################################################################


def outliers_zscore(ser: pd.Series, thresh: float | None = None) -> pd.Series:
    """Return series of bools where true is outlier based on the zscore method."""
    thresh = thresh if thresh else 3.0
    return np.abs(scipy.stats.zscore(ser)) > thresh


def outliers_iqr(ser: pd.Series, thresh: float = 1.5) -> pd.Series:
    """Return series of bools where true is outlier based on the iqr method."""
    q1 = ser.quantile(0.25)
    q3 = ser.quantile(0.75)
    iqr = q3 - q1
    return (ser < (q1 - (thresh * iqr))) | ((q3 + (thresh * iqr)) < ser)


def thresholds_std(ser: pd.Series, n: int = 3) -> tuple[float, float]:
    median = np.mean(ser)
    stds = np.std(ser) * n
    return median - stds, median + stds


def thresholds_iqr(ser: pd.Series, thresh: float = 1.5) -> tuple[float, float]:
    """Return series of bools where true is outlier based on the iqr method."""
    q1 = np.quantile(ser, 0.25)
    q3 = np.quantile(ser, 0.75)
    iqr = q3 - q1
    return q1 - (thresh * iqr), q3 + (thresh * iqr)


def rm_outliers(
    df: pd.DataFrame, col: str, thresh: float | None = None, method: str = "iqr"
) -> pd.DataFrame:
    """Filter out outliers use iqr or zscore."""
    methods = {"zscore": outliers_zscore, "iqr": outliers_iqr}
    cond_outliers = methods[method](df[col])
    logger.info(f"{cond_outliers.sum()} outliers removed.")
    df = df[~cond_outliers]
    return df


# EVALUATION ###########################################################################


def cv_results_to_df(grid: model_selection.GridSearchCV) -> pd.DataFrame:
    """Transform grid search object into DataFrame of cv_results, unpacking the
    parameters."""
    cv_results = pd.DataFrame(grid.cv_results_)
    return pd.concat(
        [cv_results.drop("params", axis=1), pd.json_normalize(cv_results["params"])],
        axis=1,
    )


def mae_pos(y_true, y_pred) -> float:
    """Mean absolute error in positive direction."""
    res = y_pred - y_true
    return np.where(res < 0, 0, res).sum() / res.abs().sum()


def plot_roc_auc(y_true: pd.Series, y_prob: pd.Series, **kwargs) -> go.Figure:
    """Plot roc curve with auc using plotly."""
    auc = metrics.roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    title, _ = kwargs.pop("title", "roc auc"), kwargs.pop("path", None)
    fig = px.line(df, "fpr", "tpr", title=f"{title} - auc: {auc:.2f}", **kwargs)
    return fig


def plot_confusion_matrix(
    y_true: pd.Series, y_pred: pd.Series, **kwargs
) -> pd.DataFrame:
    cols = [f"pred_{i}" for i in y_true.unique()]
    rows = [f"true_{i}" for i in y_true.unique()]
    matrix = pd.DataFrame(
        metrics.confusion_matrix(y_true, y_pred), index=rows, columns=cols
    )
    total = matrix.sum().sum()
    matrix["pred_total"] = matrix.sum(1)
    matrix.loc["true_total"] = matrix.sum()
    matrix.iloc[-1, -1] = total
    dviztk.px_table(matrix.reset_index(), **kwargs)
    return matrix


def feature_importance(model, x: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a DataFrame of features and their importance."""
    feature_names = x.columns if x is not None else model.feature_name_
    return pd.DataFrame(
        {"features": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)


def plot_feature_importance(
    model,
    x: pd.DataFrame | None = None,
    title: str = "feature_importance",
    top: int = 40,
) -> tuple[pd.DataFrame, go.Figure]:
    """Plot the feature importance of the model variables as a bar chart with plotly."""
    importance = feature_importance(model, x).sort_values("importance")
    fig = px.bar(
        importance.tail(top), x="importance", y="features", orientation="h", title=title
    )
    importance = importance.sort_values("importance", ascending=False)
    return importance, fig


def plot_permutation_importance(
    model, x, y, **kwargs
) -> tuple[pd.DataFrame, go.Figure]:
    r = inspection.permutation_importance(
        model, x, y, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    r.pop("importances", None)
    pi = (
        pd.DataFrame(r, index=x.columns)
        .reset_index()
        .rename({"index": "features"}, axis=1)
        .sort_values("importances_mean")
    )
    fig = px.bar(
        pi.iloc[-40:],
        x="importances_mean",
        y="features",
        orientation="h",
        color="importances_std",
        **kwargs,
    )
    pi = pi.sort_values("importances_mean", ascending=False)
    return pi, fig


def _get_quantiles(x, feature, resolution: int = 50):
    quantiles = np.linspace(0.1, 0.9, resolution)
    return list(x[feature].quantile(quantiles).values)


def shap_force_plot(
    explainer, df: pd.DataFrame, iloc: int, path: str | Path | None = None
) -> None:
    """Create html force plot for one observation save to disk and open in browser /
    defulat html app."""
    shap_value = explainer.shap_values(df.iloc[[iloc]])
    path = dviztk.path_plotly_with_default(path)
    shap.save_html(
        path.as_posix(),
        shap.force_plot(explainer.expected_value, shap_value, df.iloc[[iloc]]),
    )


def shap_summary(
    explainer,
    x,
    path: str | Path | None = None,
    plot_size: tuple = (15, 12),
    **kwargs,
) -> plt.figure:
    """Plot shap summary of passed data saving as a matplotlib created png which is
    opened in the browser / default png app.

    example
    -------
    explainer = shap.TreeExplainer(model)
    shap_summary(explainer, x[features], path=OUT_DIR / 'shap_summary.png')
    """
    logger.info(f"{datetime.now().replace(microsecond=0)} creating shap summary plot")
    # for multi-output models you selected the output by indexing the shap_values
    shape_values = explainer.shap_values(x)
    shap.summary_plot(shape_values, x, show=False, plot_size=plot_size, **kwargs)
    path = dviztk.path_plotly_with_default(path)
    # dviztk.plt_fig_save_open(path)
    fig = plt.gcf()
    return fig


def shap_dependence_plot(
    explainer,
    df: pd.DataFrame,
    column: str,
    path: str | Path | None = None,
    interaction_index="auto",
):
    shap_values = explainer.shap_values(df)
    shap.dependence_plot(
        column, shap_values, df, interaction_index=interaction_index, show=False
    )
    path = dviztk.path_plotly_with_default(path)
    # dviztk.plt_fig_save_open(path)
    fig = plt.gcf()
    return fig


# clustering and nearest neighbours ####################################################


def distance_to_centroid(df: pd.DataFrame, centriods: np.ndarray) -> float:
    """Calcualte the distance of each observation to is corresponding centroid.

    The final column of the DataFrame must be the predicted clusters
    """
    vec, cluster = df.values[:-1], df.values[-1]
    vec_centriod = centriods[int(cluster)]
    return euclidean_distances(vec[None, :], vec_centriod[None, :])[0][0]


def add_distance_to_centroid(
    df: pd.DataFrame,
    km: cluster.KMeans,
) -> pd.Series:
    return df.apply(distance_to_centroid, args=(km.cluster_centers_,), axis=1)


def nearest_neighbours(
    feat: pd.DataFrame,
    features: list[str],
    in_trial_col: str,
    idx_col: str,
    n_neighbours: int = 15,
) -> pd.DataFrame:
    """
    Perform nearest neighbours returning the selected controls and the original trials
    in a tidy format.
    Args:
        feat: DataFrame of features and additional columns e.g. in_trial_col, index_col
        features: a list of columns to be passed to model
        in_trial_col: a string name corresponding to a booleon column, True is in the
            trial
        idx_col: a string name corresponding to a column that uniquely identifies the
            rows in the DataFrame
        n_neighbours: the number of neighbours the model returns for each trial item
        norm: True to standardise data before knn

    Returns:
        DataFrame of trials/controls and features
    """
    trial_norm, control_norm = split_df(feat, in_trial_col)
    assert feat.shape[0] == (trial_norm.shape[0] + control_norm.shape[0])
    knn = neighbors.NearestNeighbors(n_neighbours, n_jobs=-1)
    knn.fit(control_norm[features])
    neighbours = neighbours2df(
        knn.kneighbors(trial_norm[features]), trial_norm[idx_col], control_norm[idx_col]
    )
    neighbours = neighbours.merge(
        feat, how="left", left_on="idx_ctrl", right_on=idx_col, suffixes=("", "_DROP")
    ).filter(regex="^(?!.*_DROP)")
    return neighbours


def neighbours2df(
    neighbors: tuple[np.ndarray, np.ndarray],
    trial_idx: pd.Series,
    ctrl_idx: pd.Series,
) -> pd.DataFrame:
    """Transform output of sklearn nearest neighbours (with distance) into a tidy
    DataFrame format."""
    neighbors = add_trial_to_idx_column(neighbors, trial_idx)
    df = np.stack([arr.ravel() for arr in neighbors], 1)
    df = pd.DataFrame(df, columns=["distance", "idx_ctrl"])
    n_trial, n_ctrl = neighbors[0].shape
    # order of nearest neighbours corresponding to single input
    df.insert(0, "ord", repeating_rng(n_ctrl, n_trial))
    # the index corresponding to which trial the row belongs
    idx_ctrl_repeated = list(
        itertools.chain.from_iterable([[no] * n_ctrl for no in trial_idx.tolist()])
    )
    df.insert(0, "idx_trial", idx_ctrl_repeated)
    # making a category col that indicates if the idx_ctrl col is a trial or a ctrl
    in_trial = list(
        itertools.chain.from_iterable(
            [["trial"] + (["ctrl"] * (n_ctrl - 1)) for _ in range(n_trial)]
        )
    )
    df["in_trial"] = in_trial
    assert (
        df["idx_trial"].unique() == trial_idx.values
    ).all(), "the idx_trial should include only but all the original trial idxs"
    # replaceing the df index values of the ctrls with the values from the idx series in
    # the df
    df.loc[df["in_trial"] == "ctrl", "idx_ctrl"] = df["idx_ctrl"].map(ctrl_idx)
    return df


def add_trial_to_idx_column(
    neighbors: tuple[np.ndarray, np.ndarray], trial_idx: pd.Series
):
    """Add the idx of the trial and the distance (distance to its self is 0) to the
    sklearn knn output so it can be transformed into a tidy format with neighbors."""
    dist, idxs = neighbors
    dist_trial = np.zeros(dist.shape[0]).reshape(-1, 1)
    trial_idx = trial_idx.values.reshape(-1, 1)
    return np.hstack([dist_trial, dist]), np.hstack([trial_idx, idxs])


def repeating_rng(rng, n_repeat):
    return list(range(rng)) * n_repeat


def split_df(df: pd.DataFrame, flag_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data frame on sequence of booleons returning the subset that matches true
    first."""
    return (
        df[df[flag_col]].reset_index(drop=True),
        df[~df[flag_col]].reset_index(drop=True),
    )


def kmeans_elbow(
    data,
    n=15,
    title: str | None = None,
    plot_elbow: bool = True,
    use_tqdm: bool = True,
    random_state: int = 1,
) -> tuple[pd.DataFrame, go.Figure]:
    """Perform elbow method using kmeans visualising the loss with plotly."""
    # n samples must be greater than n of clusters
    n = min([data.shape[0], n])
    losses = {}
    idxs = tqdm(range(1, n), total=n) if use_tqdm else range(1, n)
    for i in idxs:
        km = cluster.KMeans(i, random_state=random_state).fit(data)
        losses[i] = km.inertia_
    losses = pd.Series(losses).to_frame(name="loss").reset_index()
    losses["diff"] = losses["loss"] - losses["loss"].shift(1)
    losses["grad"] = losses["diff"] / losses["loss"]
    losses["pct_of_loss"] = losses["loss"] / losses["loss"].iloc[0]
    if plot_elbow:
        fig = dviztk.line_seconday_axis(
            losses, x="index", y_primary="pct_of_loss", y_secondary="grad", title=title
        )
    return losses, fig


# DIMENSIONALITY REDUCTION #############################################################


def pca_explained_var(pca: decomposition.PCA, fmt: bool = True) -> pd.DataFrame:
    """Return the variance captured by each principal component."""
    ratios = {
        "var": pca.explained_variance_,
        "var_ratio": pca.explained_variance_ratio_,
        "var_ratio_cum": pca.explained_variance_ratio_.cumsum(),
    }
    df = pd.DataFrame(ratios)
    if fmt:
        df = pd.concat([df.iloc[:, 0], df.iloc[:, 1:].map("{:.1%}".format)], axis=1)
    return df


# FORECASTING ##########################################################################


def make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)


def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def add_datepart(
    df, field_name, prefix=None, drop=False, time=False, dowcondmonth=False
) -> pd.DataFrame:
    logger.info("adding dateparts")
    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub("[Dd]ate$", "", field_name))
    attr = [
        "Year",
        "Month",
        "Week",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
    ]
    if time:
        attr = attr + ["Hour", "Minute", "Second"]
    # Pandas removed `dt.week` in v1.1.10
    week = (
        field.dt.isocalendar().week.astype(field.dt.day.dtype)
        if hasattr(field.dt, "isocalendar")
        else field.dt.week
    )
    for n in attr:
        df[prefix + n] = getattr(field.dt, n.lower()) if n != "Week" else week
    mask = ~field.isna()
    df[prefix + "Elapsed"] = np.where(
        mask, field.values.astype(np.int64) // 10**9, np.nan
    )
    if drop:
        df.drop(field_name, axis=1, inplace=True)

    if dowcondmonth:
        for month in df["Month"].unique():
            df.loc[df["Month"] == month, "DayofweekCondOnMonth"] = df[
                "Dayofweek"
            ] + int(7 * month)
        df["DayofweekCondOnMonth"] = df["DayofweekCondOnMonth"].astype(int)
    # assert df['DayofweekCondOnMonth'].nunique() == (7 * 12), 'should have 7 unique
    # categories for each month'

    return df


def lag_features(
    df: pd.DataFrame, cat_var: Any, cont_var: str, lags: Sequence[int] | None = None
) -> pd.DataFrame:
    logger.info("creating lag features")
    lags = lags if lags else [91, 98, 105, 112, 119, 126, 182, 364, 365, 546, 728]
    for lag in tqdm(lags):
        df[f"{cont_var}_lag_{lag}"] = df.groupby(cat_var)[cont_var].transform(
            lambda x: x.shift(lag)  # noqa: B023
        )
    return df


def rolling_mean_features(
    df: pd.DataFrame, cat_var: Any, cont_var: str, lags: Sequence[int] | None = None
) -> pd.DataFrame:
    logger.info("creating rolling mean features")
    lags = (
        lags
        if lags
        else [91, 98, 105, 112, 119, 126, 186, 200, 210, 250, 300, 365, 546, 700]
    )
    cat_var = slibtk.listify(cat_var)
    for lag in tqdm(lags):
        df[f"{cont_var}_roll_mean_{lag}"] = (
            df.groupby(cat_var)[cont_var]
            .shift(1)
            .rolling(window=lag, min_periods=2)
            .mean()
        )
    return df


def dow_rolling_mean_features(df: pd.DataFrame, cont_var: str, cats) -> pd.DataFrame:
    logger.info("creating dow rolling mean features")
    lags = [12, 13, 14, 16, 20, 52, 78, 104, 156]
    for lag in tqdm(lags):
        df[f"{cont_var}_dow_roll_mean_{lag}"] = (
            df.groupby([*cats, "Dayofweek"])[cont_var]
            .shift(1)
            .rolling(window=lag, min_periods=2)
            .mean()
        )
    return df


def ewm_features(
    df: pd.DataFrame,
    cat_var: Any,
    cont_var: str,
    lags: Sequence[float] | None = None,
    alphas: Sequence[float] | None = None,
) -> pd.DataFrame:
    logger.info("creating ewm features")
    alphas = alphas if alphas else [0.95, 0.9, 0.8, 0.7, 0.5]
    lags = lags if lags else [91, 98, 105, 112, 180, 270, 365, 546, 728]
    for lag in tqdm(lags):
        for alpha in alphas:
            df[f'{cont_var}_ewm_{str(alpha).replace(".", "")}_lag_{lag}'] = df.groupby(
                cat_var
            )[cont_var].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())  # noqa: B023

    return df


def dow_category_sum_features(df: pd.DataFrame, cats, value) -> pd.DataFrame:
    for i in tqdm([12, 13, 14, 16, 20, 52, 78, 104, 156]):
        quantity_dow_article_sum = (
            df.groupby([*cats, "Dayofweek"])[value]
            .sum()
            .shift(i)
            .reset_index()
            .rename({value: f"{value}_dow_category_sum_{i}"})
        )
        df = pd.merge(df, quantity_dow_article_sum, how="left", on=[*cats, "Dayofweek"])
    return df


# WORD2VEC / GENSIM ####################################################################


def word2vec(sequences):
    model = Word2Vec(
        sentences=sequences,
        workers=os.cpu_count(),
        min_count=1,
        epochs=5,
        compute_loss=True,
        sg=1,
    )
    wv_df_ty = wv2df(model.wv)
    wv_df_ty.index = wv_df_ty.index.astype(str)


def wv2df(wv) -> pd.DataFrame:
    """Transform gensim keyed vector object to DataFrame.

    Must be done this way to ensure correct order is maintained
    between the vocab and vectors.
    wv: Word2VecKeyedVectors
    """
    vocab = list(wv.index_to_key)
    vectors = [wv[word] for word in tqdm(vocab)]
    return pd.DataFrame(
        vectors, index=vocab, columns=[f"dim{i}" for i in range(wv.vectors.shape[1])]
    )


# GRADIENT BOOSTING ####################################################################


def prediction_uncertainty(x, y, n_neighbors=21):
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(x)
    distances, indices = nn.kneighbors(x)
    stds = []
    for i in range(indices.shape[0]):
        idxs = indices[i, 1:].tolist()
        std = y[idxs].std()
        stds.append(std)
    return stds


def random_forest_uncertainty(rf, x) -> np.ndarray:
    trees = np.array([tree.predict_proba(x)[:, -1] for tree in rf.estimators_]).T
    return trees.std(1)
