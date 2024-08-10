from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
from pandas._libs.tslibs.timestamps import Timestamp


def add_vertline(fig: go.Figure, x: float | str, y1: float, y0: float = 0) -> go.Figure:
    """Add vertical line to at x=1 to plotly figure with height of y1."""
    fig.add_shape(
        type="line",
        x0=x,
        y0=y0,
        x1=x,
        y1=y1,
        line={"width": 5, "dash": "dot", "color": "red"},
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
    add vertical lines to plotly figures that repeat at a certain frequency
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
            line={"width": 1, "dash": "dot"},
        )
    return fig
