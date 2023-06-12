"""Name space for large string literals used as into to callable typing macro."""


standard_lib: str = """from __future__ import annotations
from typing import *
import logging
from collections import defaultdict
from datetime import datetime, timedelta, date
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
import scipy
from tqdm.auto import tqdm
import pandas as pd
from sklearn import *
import lightgbm as lgb
import optuna
from optuna import Trial
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{{:.3f}}'.format)
"""


please_info = "Please let me know if you require any more information."
any_help = "Any help would be much appreciated."
please_queries = "Please let me know if you have any queries."

sql_sum: str = """select i,
    count(*) as n,
    count(*) / sum(count(*)) over() as pct,
    row_number() over (order by count(*) desc) as rank,
from table
    group by i
order by n desc
;"""

sql_duplicates: str = """select i, count(*) as n
from table
group by i
having n > 1"""

melt_plot: str = """melt = df.melt([''])
fig = px.line(melt, '', 'value', color='variable')
fig"""


px_3d_scatter: str = """fig = px.scatter_3d(df, "dim0", "dim1", "dim2", color="")
fig.update_traces(marker=dict(size=3))
display_plotly(fig)"""
