"""Name space for large string literals used as into to callable typing macro."""


standard_lib: str = """from __future__ import annotations
from typing import *
import logging
import collections
from datetime import datetime, timedelta
from pathlib import Path
import itertools
import functools
import subprocess
import io
import os
import gc
import re
import sys
import time
import logging
import pickle
import json
import random
import string
import requests
import copy
import shutil
"""

data_sci: str = f"""{standard_lib}
import numpy as np
from scipy import stats
import scipy
from tqdm.auto import tqdm
import pandas as pd
from sklearn import (
    model_selection,
    metrics,
    preprocessing,
    ensemble,
    neighbors,
    cluster,
    decomposition,
    inspection,
    linear_model,
    pipeline
)
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna import Trial
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{{:.3f}}'.format)
"""

train_test_split: str = """train, test = model_selection.train_test_split(x, stratify=x['y'], test_size=0.2)"""

please_info = "Please let me know if you require any more information."
any_help = "Any help would be much appreciated."
please_queries = "Please let me know if you have any queries."

sql_template_sum: str = """select col,
    count(*) as n,
    count(*) / sum(count(*)) over() as pct,
    row_number() over (order by count(*) desc) as rank,
from table
    group by col
order by n desc
;
"""

sql_template_duplicates: str = """select col, count(*) as n
from table
group by col
having n > 1"""

melt_plot: str = """melt = df.melt(['cat'])
fig = px.line(melt, 'cat', 'value', color='variable')
display_plotly(fig)
"""

px_3d_scatter: str = """fig = px.scatter_3d(df, "dim0", "dim1", "dim2", color="")
fig.update_traces(marker=dict(size=3))
display_plotly(fig)"""
