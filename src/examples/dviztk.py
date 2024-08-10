import itertools
import os
import random
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pandas._libs.tslibs.timestamps import Timestamp
from plotly.offline import plot
from plotly.subplots import make_subplots
from tqdm import tqdm

from dstk import slibtk


def add_vertline(fig: go.Figure, x: float | str, y1: float, y0: float = 0) -> go.Figure:
    """Add vertical line to at x=1 to plotly figure with height of y1."""
    fig.add_shape(
        type="line",
        x0=x,
        y0=y0,
        x1=x,
        y1=y1,
        line=dict(width=5, dash="dot", color="red"),
    )
    return fig


def add_vertlines(
    fig: go.Figure, xs: Sequence[float | str], y1: float, y0: float = 0
) -> go.Figure:
    for x in xs:
        fig = add_vertline(fig, x, y1, y0)
    return fig

def add_periodic_vertical_lines(
    fig: go.Figure,
    start: str | Timestamp,
    end: str | Timestamp,
    freq: str,
    y1: float,
    y0: float = 0,
) -> go.Figure:
    """
    add vertical lines to plotly figures that repeat at a certain frequency ie every sunday
    Args:
        fig: plotly figure
        start: first date of period
        end: last date of the period
        freq: pandas time series frequency directive ie W-THU
        y1: max value of line
        y0: minimum value of line

    Returns:
        fig: the plotly figures with lines added as a trace
    """
    dates_dow = pd.date_range(start, end, freq=freq)
    for day in dates_dow:
        fig.add_shape(
            type="line",
            x0=day,
            y0=y0,
            x1=day,
            y1=y1,
            line=dict(width=1, dash="dot"),
        )
    return fig


# visualising decision boundaries and regression surfaces ##############################################################


def px_scatter3d_regression(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    model,
    path: Path | None = None,
    title: str | None = None,
    color_discrete_sequence=None,
    resolution: int = 200,
    marker_size: int = 5,
    *args,
    **kwargs,
) -> None:
    """
    plot a 3 dimensional scatter plot where z is the target variable and super impose a regression surface corresponding
    to model where model has a sklearn style api (fit, predict)
    Args:
        df: DataFrame where x and y are features and z is a continuoius label.
        x: column name of feature input 1
        y: column name of feature input 2
        z: column name of continuious label
        model: model object with fit and predict methods ie sklearn
        path: save location of plotly html output
        *args: passed to px.scatter_3d
        **kwargs: passed to px.scatter_3d

    Returns:
        None
    """
    path = path_plotly_with_default(path)
    model.fit(df[[x, y]], df[z])
    df["pred"] = model.predict(df[[x, y]])
    surface = make_predicted_surface(
        df[x], df[y], predictor=model.predict, resolution=resolution
    )
    color_discrete_sequence = (
        color_discrete_sequence if color_discrete_sequence else colorlst
    )
    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color_discrete_sequence=color_discrete_sequence,
        title=title,
        *args,
        **kwargs,
    )
    fig.update_traces(marker=dict(size=marker_size))
    fig.add_trace(surface)
    plot(fig, filename=path.as_posix())


def make_predicted_surface(
    x: pd.Series, y: pd.Series, predictor: Callable, resolution: int = 200
) -> go.Surface:
    """
    for a given set of values x and y, and a trained model, estimate the grid surface for all permutations
    Args:
        x: first independent variable
        y: Second independent variable
        predictor: Function that is applied to the nx2 array of grid coordinates and returns an nx1 array of predictions
        resolution: number of points on each axis

    Returns:
        surface: plotly surface trace object
    """
    # setting up grid
    x_axis = np.linspace(min(x), max(x), resolution)
    y_axis = np.linspace(min(y), max(y), resolution)
    xx, yy = np.meshgrid(x_axis, y_axis)
    coord = pd.DataFrame(
        np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]), columns=[x.name, y.name]
    )
    # predicting and formatting z into a grid
    pred = predictor(coord)
    pred = np.array(pred).reshape(xx.shape)
    surface = {"z": pred.tolist(), "x": x_axis, "y": y_axis}
    trace = go.Surface(
        x=surface["x"],
        y=surface["y"],
        z=surface["z"],
        opacity=0.7,
        colorscale=px.colors.sequential.YlGnBu,
        # reversescale=False,
    )
    return trace


def make_3d_grid(
    df: pd.DataFrame, x: str, y: str, z: str, resolution: int = 50
) -> pd.DataFrame:
    """Make a DataFrame of dimensional coordinates that uniformly cover a 3
    dimensional space."""
    # getting axis ranges
    axis_mins = df[[x, y, z]].min().tolist()
    axis_maxs = df[[x, y, z]].max().tolist()
    # filling axis ranges and combining permutations of all combinations into a DataFrame
    xax = np.linspace(axis_mins[0], axis_maxs[0], resolution)
    yax = np.linspace(axis_mins[1], axis_maxs[1], resolution)
    zax = np.linspace(axis_mins[2], axis_maxs[2], resolution)
    data = list(itertools.product(xax, yax, zax))
    return pd.DataFrame(data, columns=[x, y, z])


def px_scatter3d_classification(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    lbls: str,
    predictor: Callable,
    path: Path,
    resolution: int = 30,
) -> None:
    """
    plot a 3d set of observations colored by their label with the classification boundary of a predictor superimposed
    on the 3 dimensional plot in the form of equally spaced points that uniformly cover all three dimensions.
    Args:
        df: DataFrame where x and y are features and z is a continuoius label.
        x: column name of feature input 1
        y: column name of feature input 2
        z: column name of feature input 3
        lbls: classification labels
        predictor: Function that is applied to the nx2 array of grid coordinates and returns an nx1 array of predictions
        path: save location of plotly html output
        resolution: number of points on each axis

    Returns:
        None
    """
    color_map = {lbl: make_rgb() for lbl in df[lbls].nunique()}
    fig = px.scatter_3d(df, x, y, z, color="targ", color_discrete_map=color_map)
    grid = make_3d_grid(df, x, y, z, resolution=resolution)
    grid["pred"] = "cb_" + predictor(grid)
    fig_lbls = px.scatter_3d(
        grid,
        x,
        y,
        z,
        color="pred",
        opacity=0.05,
        color_discrete_map={f"cb_{k}": v for k, v in color_map.items()},
    )
    for data in fig_lbls.data:
        fig.add_trace(data)
    plot(fig, filename=path.as_posix())

