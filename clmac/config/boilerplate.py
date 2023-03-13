"""Name space for large string literals used as into to callable typing
macro."""

logger = """import logging

logging.basicConfig(**log_config)
logger = logging.getLogger(__name__)"""

log_config = """logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt = '%d-%m-%Y %H:%M:%S',
        level=logging.INFO,
)"""

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
 linear_model
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


warnings.simplefilter(action='ignore')
"""

big_query: str = """client_bq = bigquery.Client()
df.to_gbq(destination_table='dataset.some_data',
            project_id=os.environ['GCLOUD_PROJECT'],
            if_exists='replace',
            location='europe-west2')
"""

deep_learning: str = """import torch.nn as nn
import torch
import torch.nn.functional as F
from fastai import *
from fastai.vision import *
"""

train_test_split: str = """train, test = model_selection.train_test_split(x, stratify=x['y'], test_size=0.2)"""

iris_dataset: str = """import seaborn as sns
x = sns.load_dataset('iris').rename({'species': 'y'}, axis=1)
train, test = model_selection.train_test_split(x, stratify=x['y'], test_size=0.2)
"""

months: str = """'January',
'February',
'March',
'April',
'May',
'June',
'July',
'August',
'September',
'October',
'November',
'December',
"""

months_abbr: str = """'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',"""

please_info = "Please let me know if you require any more information."
any_help = "Any help would be much appreciated."
please_queries = "Please let me know if you have any queries."

sql_template_cum_sum: str = """with wrapped as (
select col,
       count(*) as n,
       count(*) / sum(count(*)) over() as pct,
       row_number() over (order by count(*) desc) as rank,
from table
    group by col
order by n desc
    )
select * except (rank),
       sum(pct) over (order by rank) as cum_pct,
from wrapped;
;
"""

sql_template_duplicates: str = """select col, count(*) as n
from table
group by col
having n > 1"""
