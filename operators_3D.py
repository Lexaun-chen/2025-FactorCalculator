"""
CPU算子库

包含:
- 基本算术运算算子 (加减乘除等)
- 基本数学函数算子 (log、sqrt等)
- 基本时序算子 (移动平均、滞后等)
- 基本截面算子 (排名、标准化等)
"""

import numpy as np
import pandas as pd
import bottleneck as bn
import warnings
import numba

# 保护函数,处理异常情况
def protected_div(x, y):
    """
    保护除法,避免除以零
    
    参数:
        x: 分子
        y: 分母
        
    返回:
        除法结果,当分母为零时返回1
    """
    try:
        if np.isscalar(y) and abs(y) < 1e-10:
            return 1.0
        return x / y
    except (ZeroDivisionError, FloatingPointError, ValueError):
        return 1.0

def protected_log(x):
    """
    保护对数,避免对负数或零取对数
    
    参数:
        x: 输入值
        
    返回:
        对数结果,当输入小于等于零时返回0
    """
    try:
        if np.isscalar(x) and x <= 0:
            return 0.0
        return np.log(x)
    except (ValueError, FloatingPointError):
        return 0.0

def protected_sqrt(x):
    """
    保护平方根,避免对负数取平方根
    
    参数:
        x: 输入值
        
    返回:
        平方根结果,当输入小于零时返回0
    """
    try:
        if np.isscalar(x) and x < 0:
            return np.sqrt(abs(x))
        return np.sqrt(x)
    except (ValueError, FloatingPointError):
        return 0.0

# 算术运算算子
def add(x, y):
    """加法运算"""
    return x + y

def subtract(x, y):
    """减法运算"""
    return x - y

def multiply(x, y):
    """乘法运算"""
    return x * y

def divide(x, y):
    """除法运算 (保护版本)"""
    return protected_div(x, y)

def power(x, y):
    """
    幂运算 (保护版本)
    """
    try:
        if np.isscalar(y):
            # 限制指数范围,避免计算过大的幂
            y_clipped = np.clip(y, -10, 10)
            if np.isscalar(x) and x < 0:
                # 对于负底数,使用绝对值
                return np.power(abs(x), y_clipped)
            return np.power(x, y_clipped)
        return np.power(abs(x), y)
    except (ValueError, OverflowError, FloatingPointError):
        return 1.0

# 数学函数算子
def log(x):
    """自然对数 (保护版本)"""
    return protected_log(x)

def sqrt(x):
    """平方根 (保护版本)"""
    return protected_sqrt(x)

def abs_val(x):
    """绝对值"""
    return abs(x)

def neg(x):
    """取负"""
    return -x

def sigmoid(x):
    """Sigmoid函数"""
    try:
        if np.isscalar(x) and x > 100:
            return 1.0
        elif np.isscalar(x) and x < -100:
            return 0.0
        return 1.0 / (1.0 + np.exp(-x))
    except (OverflowError, FloatingPointError):
        if x > 0:
            return 1.0
        return 0.0

def hardsigmoid(x):
    """
    Hard Sigmoid函数 - 分段线性近似的Sigmoid
    
    参数:
        x: 输入值
        
    返回:
        Hard Sigmoid结果: max(0, min(1, (x+1)/2))
    """
    try:
        if np.isscalar(x):
            return max(0.0, min(1.0, (x + 1.0) / 2.0))
        else:
            # 向量化处理: max(0, min(1, (x+1)/2))
            # 按照公式顺序实现: 先计算(x+1)/2,再取min(1,结果),最后取max(0,结果)
            result = (x + 1.0) / 2.0
            result = np.minimum(1.0, result)
            result = np.maximum(0.0, result)
            return result
        
    # 分离逻辑：先判断x的情况 跳过危险计算
    # 浮点数上溢或者下溢；非数值输入（如 inf 或 nan）：可能触发 FloatingPointError
    except (OverflowError, FloatingPointError):
        if x > 0:
            return min(1.0, (x + 1.0) / 2.0)
        return 0.0

# TODO:让alpha变成可变参数
def leakyrelu(x, alpha=0.1):
    """
    Leaky ReLU函数 - 允许负值有小梯度的ReLU变体
    
    参数:
        x: 输入值
        alpha: 负值区域的斜率,默认为0.01
        
    返回:
        Leaky ReLU结果: x if x > 0 else alpha * x
    """
    try:
        if np.isscalar(x):
            return x if x > 0 else alpha * x
        else:
            # 向量化处理：逐元素处理
            return np.where(x > 0, x, alpha * x)
    except (OverflowError, FloatingPointError):
        return 0.0

def gelu(x):
    """
    GELU函数 - Gaussian Error Linear Unit
    
    参数:
        x: 输入值
        
    返回:
        GELU结果: x * Φ(x),其中Φ是标准正态分布的累积分布函数
    """
    try:
        # 使用近似公式: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    except (OverflowError, FloatingPointError):
        return 0.0

def sign(x):
    """符号函数"""
    try:
        return np.sign(x)
    except (ValueError, FloatingPointError):
        return 0.0

def power2(x):
    """
    Raise the input to the power of 2
    """
    return power(x,2)

def power3(x):
    """
    Raise the input to the power of 3
    """
    return power(x,3)

def curt(x):
    """
    Calculate the cube root 永远返回实数结果 power幂运算会返回虚数
    """
    return np.cbrt(x)

def inv(x):
    """
    Calculate the inverse (1/x)
    """
    return divide(1,x)

def mean2(x, y):
    return (x+y)/2

# 条件算子
def if_then_else(input1, input2, output1, output2):
    mask = (input1 >= input2)
    return np.where(mask, output1, output2)

def series_max(x, y):
    return np.maximum(x, y)

# 时序算子
def ts_lag(x, periods=1):
    """滞后算子"""
    if not np.isscalar(periods):
        periods = int(periods.item(0))
        
    if isinstance(x, np.ndarray):   
        result = np.empty_like(x)     
        if x.ndim == 1:  # 一维数组
            result[:periods] = np.nan
            result[periods:] = x[:-periods]
        elif x.ndim == 2:  # 二维数组 (时间 × 股票)
            result[:periods, :] = np.nan  # 所有列的前periods行设为NaN
            result[periods:, :] = x[:-periods, :]  # 所有列同时进行滞后操作
        else:
            # 更高维数组,返回原始数组
            return x
        return result
    return x

def ts_diff(x, periods=1):
    """差分算子"""
    if not np.isscalar(periods):
        periods = int(periods.item(0))
        
    if isinstance(x, np.ndarray):
        # 一维二维都是一个算法
        lagged = ts_lag(x, periods)
        result = x - lagged
        return result
    return x

def ts_pct_change(x, periods=1):
    """百分比变化算子"""
    if not np.isscalar(periods):
        periods = int(periods.item(0))
        
    if isinstance(x, np.ndarray):
        lagged = ts_lag(x, periods)
        with np.errstate(divide='ignore', invalid='ignore'): #忽略除零警告与无效操作警告   1 / 0  # 静默返回inf
            return (x / lagged) - 1   
    return x

def ts_mean(x, window=5):
    """滚动平均算子 (移动平均)"""
    if not np.isscalar(window):
        window = int(window.item(0))
        
    if isinstance(x, np.ndarray):
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_mean(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_mean(x, window=window, min_count=1, axis=0)
        return result
    return x

def ts_std(x, window=5):
    """滚动标准差算子"""
    if not np.isscalar(window):
        window = int(window.item(0))
        
    if isinstance(x, np.ndarray):
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_std(x, window=window, min_count=2)
        elif x.ndim == 2:
            result = bn.move_std(x, window=window, min_count=2, axis=0)
            
        return result
    return x

@numba.jit(nopython=True)
def calc_ts_ewm_vectorized(x, alpha):
    """numba 优化的 ts_ewm 计算核心 - 向量化实现"""
    result = np.empty_like(x)
    
    if x.ndim == 1:
        result[0] = x[0]
        for i in range(1, len(x)):
            result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
    elif x.ndim == 2:
        # 第一行直接赋值
        result[0, :] = x[0, :]
        
        # 对后续每一行进行向量化计算
        for i in range(1, x.shape[0]):
            result[i, :] = alpha * x[i, :] + (1 - alpha) * result[i-1, :]
    
    return result

def ts_ewm(x, halflife = 1):
    """指数加权移动平均算子 - 使用 numba 优化的向量化实现"""
    if not np.isscalar(halflife):
        halflife = int(halflife.item(0))
        
    if halflife <= 0:
        warnings.warn("halflife应该大于0,已设置为1")
        halflife = 1

    if isinstance(x, np.ndarray):      
        alpha = 1 - np.exp(-np.log(2)/halflife) #确保权重在halflife期后衰减到初始值的一半
        return calc_ts_ewm_vectorized(x, alpha)
    return x

def ts_max(x, window=5):
    """
    Calculate the maximum value over a rolling window
    """
    if not np.isscalar(window):
        window = int(window.item(0))

    if isinstance(x, np.ndarray):
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_max(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_max(x, window=window, min_count=1, axis=0)
        return result
    return x

def ts_min(x, window=5):
    """
    Calculate the minimum value over a rolling window
    """
    if not np.isscalar(window):
        window = int(window.item(0))

    if isinstance(x, np.ndarray):
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_min(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_min(x, window=window, min_count=1, axis=0)
        return result
    return x

def ts_argmin(x, window=5):
    """
    Calculate the position of the minimum value over a rolling window
    """
    if not np.isscalar(window):
        window = int(window.item(0))
        
    if isinstance(x, np.ndarray) and x.ndim==2:
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_argmin(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_argmin(x, window=window, min_count=1, axis=0)
        return result
    else:
        raise ValueError("输入不是二维ndarray")

def ts_argmax(x, window=5):
    """
    Calculate the position of the maximum value over a rolling window
    """
    if not np.isscalar(window):
        window = int(window.item(0))
        
    if isinstance(x, np.ndarray) and x.ndim==2:
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_argmax(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_argmax(x, window=window, min_count=1, axis=0)
        return result
    else:
        raise ValueError("输入不是二维ndarray")

def ts_max_to_min(x, window=5):
    if not np.isscalar(window):
        window = int(window.item(0))
        
    return ts_max(x, window) - ts_min(x, window)

def ts_sum(x, window=5):
    """
    Calculate the sum over a rolling window
    """
    if not np.isscalar(window):
        window = int(window.item(0))

    if isinstance(x, np.ndarray):
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_sum(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_sum(x, window=window, min_count=1, axis=0)
        return result
    return x

@numba.jit(nopython=True, parallel=True)
def calc_ts_max_mean(x, window=5, num=3):    
    result = np.full_like(x, np.nan)
    
    # 并行处理不同的股票
    for j in numba.prange(x.shape[1]):
        start = int(window - 1)
        end = x.shape[0]
        for i in range(start, end):
            window_data = x[max(0, i-window+1):i+1, j]
            
            valid_mask = ~np.isnan(window_data)
            valid_count = np.sum(valid_mask)
            valid_data = window_data[valid_mask]

            if valid_count <= num:
                # 如果有效值数量小于等于num，直接使用所有有效值
                result[i, j] = np.mean(valid_data)
            else:
                # 使用partition只部分排序数组
                pivot = valid_count - num
                np.partition(valid_data, pivot)
                result[i, j] = np.mean(valid_data[pivot:])
    
    return result

def ts_max_mean(x, window=5, num=3):
    """
    计算滚动窗口内最大的num个值的平均值
    
    参数:
        x: 输入序列
        window: 滚动窗口大小,默认为5
        num: 取最大的几个值,默认为3
        
    返回:
        滚动窗口内最大的num个值的平均值
    """
    if not np.isscalar(window):
        window = int(window.item(0))
    if not np.isscalar(num):
        num = int(num.item(0))
    window = max(window,num)
    num = min(window,num)

    if isinstance(x, np.ndarray):
        return calc_ts_max_mean(x, window, num)
    return x

def ts_cov(x, y, window=5):
    # !慢
    """
    计算两个序列在滚动窗口内的协方差,按InnerCode分组
    
    参数:
        x: 第一个输入序列
        y: 第二个输入序列
        window: 滚动窗口大小,默认为5
        
    返回:
        按InnerCode分组的滚动协方差
    """
    if not np.isscalar(window):
        window = int(window.item(0))

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        result = np.full_like(x, np.nan)
        
        # 对于numpy数组,第一维是时间,第二维是股票
        if x.ndim == 2 and y.ndim == 2:
            # 对每个股票计算滚动协方差
            for j in range(x.shape[1]):
                for i in range(window-1, x.shape[0]):
                    x_window = x[max(0, i-window+1):i+1, j]
                    y_window = y[max(0, i-window+1):i+1, j]
                    
                    # 找出两个窗口中都有效的数据点
                    valid_mask = ~np.isnan(x_window) & ~np.isnan(y_window)
                    if np.sum(valid_mask) >= 2:  # 至少需要2个点才能计算协方差
                        result[i, j] = np.cov(x_window[valid_mask], y_window[valid_mask])[0, 1]
        
        return result
    return x

@numba.jit(nopython=True, parallel=True)
def calc_ts_corr(x, y, window=5):
    result = np.full_like(x, np.nan)
    
    # 并行处理不同的股票
    for j in numba.prange(x.shape[1]):
        start = int(window - 1)
        end = x.shape[0]
        for i in range(start, end):
            x_window = x[max(0, i-window+1):i+1, j]
            y_window = y[max(0, i-window+1):i+1, j]
            
            # 找出两个窗口中都有效的数据点
            valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))
            valid_count = np.sum(valid_mask)
            
            if valid_count >= 2:
                x_valid = x_window[valid_mask]
                y_valid = y_window[valid_mask]
                
                # 计算相关系数
                mean_x = np.mean(x_valid)
                mean_y = np.mean(y_valid)
                num = 0.0
                den_x = 0.0
                den_y = 0.0
                
                dx = x_valid - mean_x
                dy = y_valid - mean_y
                
                num = np.sum(dx * dy)
                den_x = np.sum(dx * dx)
                den_y = np.sum(dy * dy)
                
                if den_x > 0 and den_y > 0:
                    result[i, j] = num / np.sqrt(den_x * den_y)
    
    return result

def ts_corr(x, y, window=5):
    """
    计算两个序列在滚动窗口内的相关系数,按InnerCode分组
    
    参数:
        x: 第一个输入序列
        y: 第二个输入序列
        window: 滚动窗口大小,默认为5
        
    返回:
        按InnerCode分组的滚动相关系数
    """
    if not np.isscalar(window):
        window = int(window.item(0))

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return calc_ts_corr(x,y,window)
    return x

@numba.jit(nopython=True, parallel=True)
def calc_ts_rankcorr(x, y, window=5):
    result = np.full_like(x, np.nan)
    
    # 并行处理不同的股票
    for j in numba.prange(x.shape[1]):
        start = int(window - 1)
        end = x.shape[0]
        for i in range(start, end):
            x_window = x[max(0, i-window+1):i+1, j]
            y_window = y[max(0, i-window+1):i+1, j]
            
            # 找出两个窗口中都有效的数据点
            valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))
            valid_count = np.sum(valid_mask)
            
            if valid_count >= 2:
                x_valid = x_window[valid_mask]
                y_valid = y_window[valid_mask]
                
                # 计算排名
                x_ranks = np.argsort(np.argsort(x_valid))
                y_ranks = np.argsort(np.argsort(y_valid))
                
                # 计算相关系数
                mean_x = np.mean(x_ranks)
                mean_y = np.mean(y_ranks)
                num = 0.0
                den_x = 0.0
                den_y = 0.0
                
                dx = x_valid - mean_x
                dy = y_valid - mean_y
                
                num = np.sum(dx * dy)
                den_x = np.sum(dx * dx)
                den_y = np.sum(dy * dy)
                
                if den_x > 0 and den_y > 0:
                    result[i, j] = num / np.sqrt(den_x * den_y)
    
    return result

def ts_rankcorr(x, y, window=5):
    """
    计算两个序列在滚动窗口内的秩相关系数,按InnerCode分组
    
    参数:
        x: 第一个输入序列
        y: 第二个输入序列
        window: 滚动窗口大小,默认为5
        
    返回:
        按InnerCode分组的滚动秩相关系数
    """
    if not np.isscalar(window):
        window = int(window.item(0))

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return calc_ts_rankcorr(x,y,window)
    return x

@numba.jit(nopython=True, parallel=True)
def calc_ts_to_wm(x, window=5):
    result = np.full_like(x, np.nan)

    weights_template = np.arange(1, window+1, dtype=np.float64)
    
    # 并行处理不同的股票
    for j in numba.prange(x.shape[1]):
        start = int(window - 1)
        end = x.shape[0]
        for i in range(start, end):
            window_data = x[max(0, i-window+1):i+1, j]
            
            valid_mask = ~np.isnan(window_data)
            valid_count = np.sum(valid_mask)
            
            if valid_count >= max(1, int(0.75*window)):
                valid_data = window_data[valid_mask]
                max_val = np.max(valid_data)
                
                # 使用对应的权重
                weights = weights_template[valid_mask]
                weights = weights / np.sum(weights)  # 归一化
                
                weighted_avg = np.sum(valid_data * weights)
                result[i, j] = max_val / (weighted_avg + 1e-10) #防止除0
    
    return result

def ts_to_wm(x, window=5):
    """
    对过去'window'天应用线性衰减加权 - 优化版本
    
    参数:
        x: 输入序列
        window: 滚动窗口大小,默认为5
        
    返回:
        窗口内最大值除以加权平均值,权重为线性递增
    """
    if not np.isscalar(window):
        window = int(window.item(0))
    
    if isinstance(x, pd.Series):
        # 按InnerCode分组并应用滚动计算
        result = x.groupby(level='InnerCode').apply(
            lambda x: x.sort_index(level='TradingDay').rolling(
                window=window, min_periods=max(1, int(0.75*window))
            ).apply(weighted_max_div_mean, raw=True)
        )
        
        # 处理groupby.apply后的MultiIndex
        if isinstance(result.index, pd.MultiIndex) and len(result.index.names) > 2:
            result = result.droplevel(0)
        
        return result
    elif isinstance(x, np.ndarray):
        return calc_ts_to_wm(x,window)
    return x

def ts_rank(x, window=5):
    """
    计算'window'内的排名百分比（归一化到[0,1]范围）
    
    参数:
        x: 输入序列（numpy数组）
        window: 滚动窗口大小, 默认为5
        
    返回:
        每个位置窗口内相对排名百分比（0最小，1最大）
    """
    if not np.isscalar(window):
        window = int(window.item(0))
    if isinstance(x, np.ndarray) and x.ndim <= 2:
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_rank(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_rank(x, window=window, min_count=1, axis=0)
        # move_rank 返回[-1,1], 标准化到[0,1]
        return (result+1)/2
    else:
        raise ValueError("输入数据不是一维或二维ndarray")

def ts_median(x, window=5):
    """
    计算'window'内的中位数
    
    参数:
        x: 输入序列
            - 一维数组: 单一时间序列
            - 二维数组: 多列时间序列（行表示时间，列表示不同变量）
        window: 滚动窗口大小，默认为5
        
    返回:
        每个位置的对应窗口的中位数
    """
    if not np.isscalar(window):
        window = int(window.item(0))
    if isinstance(x, np.ndarray) and x.ndim<=2:
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_median(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_median(x, window=window, min_count=1, axis=0)
        return result
    else:
        raise ValueError("输入数据不是一维或二维nndarray")



# 截面算子 (这些在DEAP中也需要特殊处理)
def rank(x):
    """
    计算输入数据在截面上的百分比排名（归一化到[0,1]区间）
    参数:
        x: 输入序列
            - Pandas Series
            - 一维、二维numpy数组(行表示时间，列表示不同变量)
    返回:
        归一后的截面百分比排名
    """
    if isinstance(x, pd.Series):
        # 原有pandas实现保持不变
        if x.index.nlevels > 1:
            return x.groupby(level='TradingDay').rank(pct=True)
        else:
            return x.rank(pct=True)
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            # 一维数组处理
            valid_mask = ~np.isnan(x)
            result = np.full_like(x,np.nan)
            if np.sum(valid_mask) > 0:
                # 计算百分比排名 (0到1之间)
                ranks = np.argsort(np.argsort(x[valid_mask]))
                result[valid_mask] = ranks / (np.sum(valid_mask) - 1) if np.sum(valid_mask) > 1 else 0.5
            return result
        elif x.ndim == 2:
            # 二维数组处理 - 按行(时间)进行排名
            result = np.full_like(x,np.nan)
            for t in range(x.shape[0]):
                valid_mask = ~np.isnan(x[t, :])
                if np.sum(valid_mask) > 1:  # 至少需要2个有效值才能计算有意义的排名
                    # 计算百分比排名
                    ranks = np.argsort(np.argsort(x[t, valid_mask]))
                    result[t, valid_mask] = ranks / (np.sum(valid_mask) - 1)
                elif np.sum(valid_mask) == 1:
                    # 只有一个有效值时,排名为0.5
                    result[t, valid_mask] = 0.5
                else:
                    # 无效值时,排名为0
                    result[t, valid_mask] = 0
            return result
    return x

def rank_div(x, y):
    """乘法运算"""
    return divide(rank(x),rank(y))

def rank_sub(x, y):
    """减法运算"""
    return rank(x) - rank(y)

def rank_mul(x, y):
    """乘法运算"""
    return rank(x) * rank(y)

def zscore(x):
    """
    Z分数标准化算子 计算标准分数（(x - μ)/σ）
    参数:
        x: 输入序列
            - Pandas Series
            - 一维、二维numpy数组(行表示时间，列表示不同变量)
    返回:
        归一后的截面百分比排名
    """
    if isinstance(x, pd.Series):
        # 原有pandas实现保持不变
        if x.index.nlevels > 1:
            #transform自动忽略组内NaN值
            grouped = x.groupby(level='TradingDay')
            mean = grouped.transform('mean') 
            std = grouped.transform('std')
            std = std.replace(0, 1) # 处理零标准差 避免除0
            return (x - mean) / std
        else:
            mean = x.mean()
            std = x.std() #自动跳过NaN值
            if std == 0:
                return pd.Series(0, index=x.index) # 返回全零序列
            return (x - mean) / std
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            # 一维数组处理
            valid_mask = ~np.isnan(x)
            result = np.full_like(x,np.nan)
            if np.sum(valid_mask) > 0:
                mean = np.mean(x[valid_mask])
                std = np.std(x[valid_mask])
                if std > 0:
                    result[valid_mask] = (x[valid_mask] - mean) / std
                else:
                    result[valid_mask] = 0 # 返回全零序列
            return result
        
        elif x.ndim == 2:
            # 二维数组处理 - 按行(时间)进行标准化
            result = np.full_like(x,np.nan)
            
            # 计算每行的均值和标准差 (忽略NaN)
            # 使用nanmean和nanstd可以避免显式循环
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                means = np.nanmean(x, axis=1, keepdims=True) #保持二维形状
                stds = np.nanstd(x, axis=1, keepdims=True)
            
            # 处理标准差为0或NaN的情况
            stds[stds == 0] = 1.0
            stds[np.isnan(stds)] = 1.0
            
            # 向量化计算Z分数
            valid_mask = ~np.isnan(x) 
            # means 和 stds 形状为 (n,1)，通过 repeat 扩展为 (n,m) 以匹配 x 的形状。
            result[valid_mask] = (x[valid_mask] - means.repeat(x.shape[1], axis=1)[valid_mask]) / stds.repeat(x.shape[1], axis=1)[valid_mask]
            
            return result
    return x

def min_max_scale(x):
    """
    最小-最大缩放算子（归一化到[0,1]区间）
    支持输入类型：
    - Pandas Series（单层或多层索引）
    - NumPy 数组（一维或二维）

    返回：
        与输入形状相同的数组/Series，每个元素被线性映射到[0,1]区间

    """
    if isinstance(x, pd.Series):
        # 原有pandas实现保持不变
        if x.index.nlevels > 1:
            grouped = x.groupby(level='TradingDay')
            min_val = grouped.transform('min')
            max_val = grouped.transform('max')
            denominator = max_val - min_val
            denominator = denominator.replace(0, 1)
            # 计算归一化值并限制范围[0,1]
            normalized = (x - daily_min) / denominator
            return np.clip(normalized, 0, 1)  
        else:
            min_val = x.min()
            max_val = x.max()
            if max_val == min_val:
                return pd.Series(0, index=x.index)
            return (x - min_val) / (max_val - min_val)
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            # 一维数组处理
            valid_mask = ~np.isnan(x)
            result = np.full_like(x,np.nan)
            if np.sum(valid_mask) > 0:
                min_val = np.min(x[valid_mask])
                max_val = np.max(x[valid_mask])
                if max_val > min_val:
                    result[valid_mask] = (x[valid_mask] - min_val) / (max_val - min_val)
                else:
                    result[valid_mask] = (x[valid_mask] - min_val)
            return result
        elif x.ndim == 2:
            # 二维数组处理 - 按行(时间)进行缩放
            result = np.full_like(x,np.nan)
            
            # 计算每行的最小值和最大值 (忽略NaN)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                min_vals = np.nanmin(x, axis=1, keepdims=True)
                max_vals = np.nanmax(x, axis=1, keepdims=True)
            
            # 计算分母并处理为0的情况
            denominators = max_vals - min_vals
            denominators[denominators == 0] = 1.0
            denominators[np.isnan(denominators)] = 1.0
            
            # 向量化计算缩放值
            valid_mask = ~np.isnan(x)
            result[valid_mask] = (x[valid_mask] - min_vals.repeat(x.shape[1], axis=1)[valid_mask]) / denominators.repeat(x.shape[1], axis=1)[valid_mask]
            
            return result
    return x

def umr(x1, x2):
    """
    自定义算子: (x1-mean(x1))*x2
    仅保留x1与x2均有效的项
    参数:
        x1: 第一个输入序列（二维数组，时间×股票）
        x2: 第二个输入序列（二维数组，时间×股票）
   
    返回:
        (x1-mean(x1))*x2,其中mean(x1)是x1在每个交易日的截面均值
        无效位置填充NaN
    """
    if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        result = np.full_like(x1, np.nan)
        
        # 对于numpy数组,第一维是时间,第二维是股票
        if x1.ndim == 2 and x2.ndim == 2:
            for t in range(x1.shape[0]):
                # 计算每个时间点的截面均值
                # valid_mask = ~np.isnan(x1[t, :])

                # 找出x1和x2同时有效的位置
                joint_valid_mask = ~np.isnan(x1[t, :]) & ~np.isnan(x2[t, :])

                if np.sum(joint_valid_mask) > 0:
                    x1_mean = np.mean(x1[t, joint_valid_mask])
                    # 计算 (x1-mean(x1))*x2
                    result[t, joint_valid_mask] = (x1[t, joint_valid_mask] - x1_mean) * x2[t, joint_valid_mask]
        
        return result
    return x1 * x2  # 非数组输入简单相乘



def regress_residual(x, y):
    """
    参数:
        x1: 第一个输入序列（二维数组，时间×股票）
        x2: 第二个输入序列（二维数组，时间×股票）

    使用 NumPy 矩阵运算进行向量化的截面回归,实验发现比statmodel的算法快五倍
    
    返回:
    resid: 形状为 (time, assets) 的 2D numpy 数组,回归残差
    """
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        raise ValueError("截面回归算子输入数据不是二维ndarray")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("输入数组必须是二维的")

    resid = np.full_like(y, np.nan)
    for t in range(x.shape[0]):
        valid_mask = (~np.isnan(x[t, :])) & (~np.isnan(y[t, :]))
        if np.sum(valid_mask) > 0:
            x_t = x[t, valid_mask]
            y_t = y[t, valid_mask]
            # 添加常数项
            X = np.column_stack([np.ones(x_t.shape[0]), x_t])
            # 使用 NumPy 的矩阵运算计算 OLS
            # (X'X)^(-1)X'y
            try:
                beta = np.linalg.solve(X.T @ X, X.T @ y_t)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(X.T @ X) @ (X.T @ y_t)
            # 计算拟合值和残差
            y_pred = X @ beta
            resid[t, valid_mask] = y_t - y_pred
    
    return resid

def sortreverse(x, n=10):
    """
    x:二维数组，时间×股票
    使用 NumPy 进行向量化的截面计算, X 截面前&后 n 名对应的 X 乘以-1

    返回:
    result: 形状为 (time, assets) 的 2D numpy 数组,是乘以-1之后的因子值
    """
    if not (isinstance(x, np.ndarray)):
        raise ValueError("截面回归算子输入数据不是二维ndarray")
    if x.ndim != 2:
        raise ValueError("输入数组必须是二维的")

    # 计算每一行非NaN的数量,并把width填充成x的形状
    width = np.sum(~np.isnan(x),axis=1)
    width = np.expand_dims(width, axis=1)
    width = np.repeat(width,x.shape[1], axis=1)

    # 每个值在当前截面上的排名
    ranking = np.argsort(np.argsort(x,axis=1,),axis=1)#这个排名其实把NaN排成最大的了
    small_mask = (ranking < n) #最小的n个值的位置
    large_mask = (ranking >= width - n) #最大的n个值的位置
    neg_mask = small_mask|large_mask
    return np.where(neg_mask, x*-1, x)


"""
sortreverse

# 测试1：含NaN的数组
x = np.array([1, np.nan, 3])
print(x * -1)  # 输出: [-1. nan -3.]（NaN保持不变）

# 测试2：整数类型
x = np.array([1, 2, 3], dtype=int)
print(x * -1)  # 输出: [-1 -2 -3]（保持int类型）

# 测试3：布尔类型（自动升级为int）
x = np.array([True, False])
print(x * -1)  # 输出: [-1  0]

"""

def return_const_1(x):
    return np.full_like(x,int(1))

def return_const_5(x):
    return np.full_like(x,int(5))

def return_const_10(x):
    return np.full_like(x,int(10))

def return_const_20(x):
    return np.full_like(x,int(20))
