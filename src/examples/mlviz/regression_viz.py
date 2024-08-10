"""Visualising regression surfaces of different models in 3d to help understand what a
different models make decisions.

We have lower dimensional brains that do a good job of perseving 3d but terrible anything beyond that by visuzalising
surfaces in lower dimension it helps given you an intuative understanding what what it is doing in higher dimensions
which we cannot visualise.


todo:
 - add neural network
 - add hyper parameter tuning
"""

import logging
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd
import plotly.express as px
import umap
import utils
import xgboost as xgb
from sklearn import ensemble, linear_model, svm, tree
from sklearn.datasets import load_boston
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", "{:.5f}".format)

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# loading and preprocessing data #######################################################################################
boston = load_boston()
x = pd.DataFrame(boston["data"], columns=boston["feature_names"])
y = boston["target"]
x["lbl"] = y

# normalising data
boston_norm = utils.standardise(x)
y = boston_norm.iloc[:, -1]
x_norm = boston_norm.iloc[:, :-1]

# pca
res = utils.pca(x_norm, n_components=2)
pcs, pca = res["pcs"], res["pca"]
pcs["lbl"] = y
pcs["type"] = "pca"

# umap
reduced = umap.UMAP(n_components=2).fit_transform(x_norm)
reduced = pd.DataFrame(reduced, columns=[f"dim{i}" for i in range(2)])
reduced["lbl"] = y
reduced["type"] = "umap"

# VISUALISING REGRESSION SURFACE FOR DIFFERENT MODELS ##################################################################

# iterating over several models plotting them as 3d regression surfaces
n_trees = 300
params_xgb = {
    "colsample_bytree": 1.0,
    "gamma": 0.5,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
}
params_lgbm = {
    "lambda_l1": 1.5,
    "lambda_l2": 1,
    "min_data_in_leaf": 30,
    "num_leaves": 31,
    "reg_alpha": 0.1,
}
models = {
    "Linear Regression": linear_model.LinearRegression(),
    "SVM (Linear)": svm.SVR(kernel="linear"),
    "SVM (rbf)": svm.SVR(kernel="rbf", C=100, gamma=0.01),
    "Decision Tree": tree.DecisionTreeRegressor(),
    f"Random Forest ({n_trees})": ensemble.RandomForestRegressor(n_trees),
    "XGBoost": xgb.XGBRegressor(**params_xgb),
    "LightGBM": lgbm.LGBMRegressor(**params_lgbm),
}

for idx, (nm, model) in enumerate(models.items()):
    utils.px_scatter3d_regression(
        pcs,
        "dim0",
        "dim1",
        "lbl",
        model,
        OUTPUT / f"{idx:03} {nm}.html",
        title=nm,
        marker_size=5,
    )

# HYPERPARAMETER TUNING RANDOM FOREST ##################################################################################


utils.px_scatter3d_regression(
    pcs,
    "dim0",
    "dim1",
    "lbl",
    model,
    OUTPUT / f"{idx:03} random_forrest.html",
    title="random_forest",
    marker_size=5,
)

x = "dim0"
y = "dim1"
z = "lbl"
resolution = 200
marker_size: int = 3
colorlst = ["#FC0FC0", "blue", "#052d3f", "#6FFFE9", "#316379", "#84a2af"]

data = []
for i in tqdm(range(1, 300, 10)):
    model = ensemble.RandomForestRegressor(i)
    model.fit(pcs[[x, y]], pcs[z])

    # pcs['pred'] = model.predict(pcs[[x, y]])
    x_axis = np.linspace(min(pcs[x]), max(pcs[x]), resolution)
    y_axis = np.linspace(min(pcs[y]), max(pcs[y]), resolution)
    xx, yy = np.meshgrid(x_axis, y_axis)
    coord = pd.DataFrame(
        np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]), columns=[x, y]
    )
    coord["pred"] = model.predict(coord)

    coord["ntree"] = i
    data.append(coord.copy())
data = pd.concat(data)

fig = px.scatter_3d(
    data,
    "dim0",
    "dim1",
    "pred",
    animation_frame="ntree",
)
fig.update_traces(marker=dict(size=marker_size))
fig.plot()

# other stuff

surface = utils.make_predicted_surface(
    pcs[x], pcs[y], predictor=model.predict, resolution=resolution
)

color_discrete_sequence = colorlst
fig = px.scatter_3d(pcs, x=x, y=y, z=z, color_discrete_sequence=color_discrete_sequence)
fig.update_traces(marker=dict(size=marker_size))
fig.add_trace(surface)
fig.plot()
