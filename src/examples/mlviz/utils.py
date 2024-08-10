import itertools
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn import decomposition, preprocessing

colorlst = ["#FC0FC0", "blue", "#052d3f", "#6FFFE9", "#316379", "#84a2af"]


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
    """Make a DataFrame of dimensional coordinates that uniformly cover a 3 dimensional
    space."""
    # getting axis ranges
    axis_mins = df[[x, y, z]].min().tolist()
    axis_maxs = df[[x, y, z]].max().tolist()
    # filling axis ranges and combining permutations of all combinations into a DataFrame
    xax = np.linspace(axis_mins[0], axis_maxs[0], resolution)
    yax = np.linspace(axis_mins[1], axis_maxs[1], resolution)
    zax = np.linspace(axis_mins[2], axis_maxs[2], resolution)
    data = list(itertools.product(xax, yax, zax))
    return pd.DataFrame(data, columns=[x, y, z])


def path_plotly_with_default(
    path: Path | str | None = None, fname: str = "plot.html"
) -> Path:
    """If a path-like object is passed return that or return a generic empty file
    path."""
    (Path.home() / ".plotly").mkdir(exist_ok=True, parents=True)
    path = Path(path) if path else next_fname(path=Path.home() / ".plotly" / fname)
    return path


def next_fname(path: Path | str | None, zfill: int = 2) -> Path:
    """Return next incremental file that does not exist
    (path.root)_{next_num}.(path.suffix)"""
    path = Path(path)
    parent, stem, suffix = path.parent, path.stem, path.suffix
    i = 0
    while (parent / f"{stem}_{str(i).zfill(zfill)}{suffix}").exists():
        i += 1
    return parent / f"{stem}_{str(i).zfill(zfill)}{suffix}"


def apply2features(
    df: pd.DataFrame,
    processor: Callable,
    features: list | None = None,
    not_features: list | None = None,
) -> pd.DataFrame:
    """Apply a processor that accepts a DataFrame to a subset of columns that is
    selected either including features or excluding not_features.

    Args:
        df: DataFrame
        processor: callable that accepts a DataFrame
        features: optional list of columns to be passed to processor
        not_features: optional list of columns to be excluded from processor
    Returns:
        df: DataFrame with subset of columns processed
    """
    assert features or not_features, "features or not_features must be passed."
    not_features = (
        not_features
        if not_features
        else [col for col in df.columns if col not in features]
    )
    return df.set_index(not_features).pipe(processor).reset_index()


def standardise(
    df: pd.DataFrame,
    features: list | None = None,
    not_features: list | None = None,
) -> pd.DataFrame:
    """Return a df where every columns is standardised, or pass the specific list of
    features to include or exclude."""
    processor = lambda x: pd.DataFrame(
        preprocessing.StandardScaler().fit_transform(x),
        index=x.index,
        columns=x.columns,
    )
    if features or not_features:
        return apply2features(
            df=df, processor=processor, features=features, not_features=not_features
        )
    return processor(df)


def pca(
    x: pd.DataFrame | np.ndarray,
    n_components=None,
    show_var: bool = True,
    random_state: int = 1,
) -> dict[str, pd.DataFrame | np.ndarray]:
    """
    run pca and return principle components and fitted estimator
    Args:
        x:
        n_components:
        show_var:
    Returns:
        pcs: Principal components as numpy arrays with pca (fitted pca model object) as an attribute
    """
    pca = decomposition.PCA(n_components, random_state=random_state)
    pcs = pca.fit_transform(x)
    if show_var:
        print(pca_explained_var(pca))
    if isinstance(x, pd.DataFrame):
        pcs = pd.DataFrame(
            pcs, index=x.index, columns=[f"dim{i}" for i in range(pcs.shape[1])]
        )
    result = {"pcs": pcs, "pca": pca}
    return result


def pca_explained_var(pca: decomposition.PCA, fmt: bool = True) -> pd.DataFrame:
    """Return the variance captured by each principle component."""
    ratios = {
        "var": pca.explained_variance_,
        "var_ratio": pca.explained_variance_ratio_,
        "var_ratio_cum": pca.explained_variance_ratio_.cumsum(),
    }
    df = pd.DataFrame(ratios)
    if fmt:
        df = pd.concat(
            [df.iloc[:, 0], df.iloc[:, 1:].applymap("{:.1%}".format)], axis=1
        )
    return df
