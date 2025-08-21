import os
import sys
import inspect
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine 


current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import *
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


# 3D合法算子名列表

# 算术算子
ARITHMETIC_OP = ["add", "subtract", "multiply", "divide", "power", "mean2"]
# 数学函数算子
MATH_OP = ["log", "sqrt", "abs_val", "neg", "sigmoid", "hardsigmoid", "leakyrelu", "gelu", "sign", "power2", "power3", "curt", "inv"]
# 时序算子
TIME_SERIES_OP = ["ts_lag", "ts_diff", "ts_pct_change", "ts_mean", "ts_std", "ts_ewm", "ts_max", "ts_min", "ts_argmin", "ts_argmax", 
                  "ts_max_to_min", "ts_sum", "ts_max_mean", "ts_cov", "ts_corr", "ts_rankcorr", "ts_to_wm", "ts_rank", "ts_median"]
# 截面算子 (排名、标准化等)
CROSS_SECTION_OP = ["rank", "rank_div", "rank_sub", "rank_mul", "zscore", "min_max_scale", "umr", "regress_residual", "sortreverse"]
# 条件算子
CONDITIONAL_OP = ["if_then_else", "series_max"]
# 常数算子
CONSTANT_OP = ["return_const_1", "return_const_5", "return_const_10", "return_const_20"]



# 给出4D合法算子名列表

CROSS_OP = ["cs_norm", "cs_rank"]
UNARY_OP = ["custom_abs", "sign", "log", "sqrt", "square"]
SM_BINARY_OP = ["add", "multiply", "custom_min", "custom_max"]
ASM_BINARY_OP = ["minus", "signedpower", "divide", "isover", "cs_neut"]
ROLLING_UNARY_OP = ["delay", "delta", "pctchange", "rolling_sum", "rolling_rank", "rolling_mean", "rolling_std", "rolling_max", "rolling_min",
                "rolling_weighted_mean_linear_decay"]
SM_ROLLING_BINARY_OP = ["rolling_covariance", "rolling_corr"]
ASM_ROLLING_BINARY_OP = []
