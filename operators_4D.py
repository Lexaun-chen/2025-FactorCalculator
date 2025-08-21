import numpy as np
import bottleneck as bn
from scipy import stats
from scipy.ndimage import convolve1d

CROSS_OP = ["cs_norm", "cs_rank"]
UNARY_OP = ["custom_abs", "sign", "log", "sqrt", "square"]
SM_BINARY_OP = ["add", "multiply", "custom_min", "custom_max"]
ASM_BINARY_OP = ["minus", "signedpower", "divide", "isover", "cs_neut"]
ROLLING_UNARY_OP = ["delay", "delta", "pctchange", "rolling_sum", "rolling_rank", "rolling_mean", "rolling_std", "rolling_max", "rolling_min",
                "rolling_weighted_mean_linear_decay"]
SM_ROLLING_BINARY_OP = ["rolling_covariance", "rolling_corr"]
ASM_ROLLING_BINARY_OP = []

def custom_abs(data):
    """
    返回绝对值
    """
    result = np.abs(data)
    return result

def sign(data):
    """
    返回符号
    """
    result = np.sign(data)
    return result

def sqrt(d):  
    """
    平方根：负号与原输入相同
    """
    result = np.sqrt(np.abs(d)) * np.sign(d)
    return result

def square(d):
    """
    平方
    """
    result = np.square(d)
    return result

def log(data):
    """
    sign(x)*log(|x|+1) 
    """
    result = np.log(np.abs(data) + 1) * np.sign(data)  
    return result

def multiply(data1, data2):
    """
    乘法
    """
    result = data1 * data2
    return result

def divide(data1, data2):
    """
    乘法（防除0）
    """
    result = data1 / (data2 + 1e-5)
    return result

def add(data1, data2):
    """
    加法
    """
    result = data1 + data2
    return result

def minus(data1, data2):
    """
    减法
    """
    result = data1 - data2
    return result

def signedpower(data1, data2):
    """
    计算 data1 的 data2 次幂（可逐元素操作，并保留符号和 NaN 传播）
    负数的小数次幂会返回 NaN
    """
    result = np.power(data1, data2)
    return result

"""
安全处理负数分数幂
def signedpower(data1, data2):
    return np.abs(data1)**data2 * np.sign(data1)**(data2 * (data2 % 1 == 0))
"""


def custom_min(data1, data2):
    """
    输入：形状可广播的两个数组
    处理：对均非Nan的每个位置逐元素比较 保留较小值（Nan自动传播）
    返回：浮点型数组（Nan自动传播）
    """
    result = np.minimum(data1, data2)
    return result


def custom_max(data1, data2):
    """
    输入：形状可广播的两个数组
    处理：对均非Nan的每个位置逐元素比较 保留较大值
    返回：浮点型数组（Nan自动传播）
    """
    result = np.maximum(data1, data2)
    return result

#是比较 data1 和 data2 的对应元素
def isover(data1, data2):
    """
    输入：形状可广播的两个数组
    处理：对均非Nan的每个位置逐元素比较 
    返回：布尔数组 （Nan自动传播）
    """
    mask = np.logical_or(np.isnan(data1), np.isnan(data2))
    result = data1 > data2
    result = result.astype(np.float64)
    result[mask] = np.nan
    return result

def cross_sectional_rank(data):
    """
    输入： ndarray (2D) 每行代表一个横截面
    返回：横截面数据的标准化排名（归一化到 [0, 1] 区间），保留NaN 值。
    """
    data = stats.rankdata(data, method='average', axis=1, nan_policy='omit')
    count = np.sum(np.isnan(data), axis=1, keepdims=True)
    denominator = np.ones_like(count)*data.shape[1] - count
    # 处理全 NaN 行（避免除以零）
    denominator = np.where(denominator==0, np.nan, denominator)
    return data / denominator

def cross_sectional_normalize(data):
    """
    输入： ndarray (2D) 每行代表一个横截面
    返回：横截面数据标准化数值，防止除0并保留NaN 值。
    """
    mu, sigma = np.nanmean(data, axis=1, keepdims=True), np.nanstd(data, axis=1, keepdims=True)
    data = (data - mu) / (sigma + 1e-5)
    return data

def valid_num_count(data):
    """
    输入： ndarray 任意维度
    输出：形状(*input_shape[:-1], 1) 的数组，表示沿最后一个轴（axis=-1）的非NaN值数量
    """
    value = np.sum(np.logical_not(np.isnan(data)), axis=-1)
    return value.reshape(*value.shape, 1)

def equal_weighted_combine(data, axis=-1):
    """
    输入： ndarray 任意维度
    处理：沿指定轴计算平均值（自动跳过NaN值）
    输出：形状为 (*input_shape[:axis], 1, *input_shape[axis+1:]) 的数组
    """
    data = cross_sectional_normalize(data)
    value = np.nanmean(data, axis=axis)
    return value.reshape(*value.shape, 1)

def add_combine(signal, factor, num):
    """
    输入：signal，factor : ndarray 维度相同
    将原始信号与（归一化后）因子按指定权重进行加权组合，num越大，信号权重越高，跳过Nan
    计算：(signal*num + factor) / (num + valid_factor_count)
    返回：加权组合后的结果矩阵，形状与输入相同
    """
    factor = cross_sectional_normalize(factor)
    pair = np.asarray([signal*num, factor]).transpose(1, 2, 0)
    numerator = np.nansum(pair, axis=-1)
    denominator = (num + np.logical_not(np.isnan(factor)))
    denominator[denominator==0.0] = np.nan
    return numerator / denominator

def cross_sectional_correlation(data1, data2, method='pearson'):
    """
    输入：data1，data2 : ndarray 维度相同
    计算两个数据集之间的横截面相关系数（Pearson或Spearman，逐行计算）
    支持Pearson和Spearman相关系数计算，自动跳过NaN值
    注意：小样本量(特征量很小)准确性
    """
    assert data1.shape == data2.shape
    if method == "spearman":
        data1, data2 = cross_sectional_rank(data1), cross_sectional_rank(data2)
    mask = np.logical_or(np.isnan(data1), np.isnan(data2))
    # 只计算均非Nan的位置
    data1[mask], data2[mask] = np.nan, np.nan
    # 沿特征轴，即axis=1
    mean1, mean2 = np.nanmean(data1, axis=1), np.nanmean(data2, axis=1)
    mean12 = np.nanmean(data1*data2, axis=1)
    square1, square2 = np.nanmean(data1**2, axis=1), np.nanmean(data2**2, axis=1)
    var1, var2 = square1 - mean1**2, square2 - mean2**2
    var1, var2 = np.where(var1 < 0.0, 0.0, var1), np.where(var2 < 0.0, 0.0, var2) #方差小于0时设为0（防止浮点误差）
    std1, std2 = np.sqrt(var1), np.sqrt(var2)
    numerator = mean12 - mean1 * mean2
    denominator = std1 * std2
    denominator[denominator==0.0] = np.nan
    corr = numerator / denominator
    return corr

def cross_sectional_neutralize(data1, data2):
    """
    输入：data1，data2 : ndarray 维度相同
    回归系数: beta = Cov(data1,data2)/Var(data2)
    输出：data1 - beta*data2 维度不变
    """
    mask = np.logical_or(np.isnan(data1), np.isnan(data2))
    data1[mask], data2[mask] = np.nan, np.nan
    data1, data2 = cross_sectional_normalize(data1), cross_sectional_normalize(data2)
    num = np.nansum(data1*data2, axis=1, keepdims=True) # 协方差
    den = np.nansum(data2**2, axis=1, keepdims=True) # 方差
    den[den==0.0] = np.nan #防除0
    result = data1 - (num / den)*data2
    return result

def rolling_cumulative_product(data, gap=1, window=5):
    """
    计算滚动窗口期的累积乘积（收益率序列的滚动复利计算）。
    输入：
        data : ndarray
        gap :滞后间隔 表示计算时跳过的初始数据点数 （默认为1）
        window : 滚动窗口大小(默认为5)
    公式：∏(1 + r_t) - 1
    返回：ndarray 形状与输入相同
    """
    data = data + 1
    product = np.ones_like(data)
    for i in range(window):
        sdata = np.roll(data, -(i+1+gap), axis=0)
        sdata[-(i+1+gap):, :] = np.nan
        product = product * sdata
    return product - 1

def delay(data, window=5, axis=0):
    """
    对输入数据沿指定轴进行滞后处理（延迟window）
    """

    sdata = np.roll(data, window, axis=axis)
    if axis == 0:
        sdata[:window, :] = np.nan
    else:
        sdata[:, :, :window] = np.nan
    return sdata

def delta(data, window=5, axis=0):
    """
    计算数据在指定窗口期前后的差值（变化量）
    """
    sdata = np.roll(data, window, axis=axis)
    if axis == 0:
        sdata[:window, :] = np.nan
    else:
        sdata[:, :, :window] = np.nan
    return data - sdata

def pctchange(data, window=5, axis=0):
    """
    计算数据在指定窗口期前后的变化比（原始量）
    """
    num = delta(data, window, axis=axis)
    data[data==0.0] = np.nan
    return divide(num, data)

def rolling_rank(data, window=5, axis=0):
    """
    计算数据在滚动窗口内的标准化排名（百分位排名）
    线性变换：将[1,window]映射到[(window+1)/2/window, 1]
    """
    result = bn.move_rank(data, window=window, axis=axis)
    result = (result * ((window-1)/2) + (window+1)/2) / window
    return result

def rolling_max(data, window=5, axis=0):
    """计算滚动窗口最大值 (使用bottleneck加速)"""
    result = bn.move_max(data, window=window, axis=axis)
    return result

def rolling_min(data, window=5, axis=0):
    """计算滚动窗口最小值 (使用bottleneck加速)"""
    result = bn.move_min(data, window=window, axis=axis)
    return result

def rolling_argmax(data, window=5, axis=0):
    """计算滚动窗口内最大值的索引位置 (使用bottleneck加速)"""
    result = bn.move_argmax(data, window=window, axis=axis)
    return result

def rolling_argmin(data, window=5, axis=0):
    """计算滚动窗口内最小值的索引位置 (使用bottleneck加速)"""
    result = bn.move_argmin(data, window=window, axis=axis)
    return result

def rolling_sum(data, window=5, axis=0):
    """
    计算滚动窗口求和
    输入：
        data : ndarray
        window : 滚动窗口大小(默认为5)
        axis : 计算轴向(默认为0)
    返回：ndarray 形状与输入相同，前window-1个位置填充NaN
    """
    kernel = np.ones(window)
    # 执行一维卷积（边缘用0填充）,结果长度为 len(data)+window-1
    rolling_sum = convolve1d(data, kernel, axis=axis, mode='constant', cval=0.0)
     # 计算有效结果长度[N - window + 1]
    valid_len = data.shape[axis]-window+1
    beg = np.floor((window-1)/2).astype("int")
    if axis == 0:
        rolling_sum = rolling_sum[beg:beg+valid_len, :]
        head = np.full((window-1, data.shape[1]), np.nan)
    else: #！高纬兼容待处理
        rolling_sum = rolling_sum[:, :, beg:beg+valid_len]
        head = np.full((data.shape[0], data.shape[1], window-1), np.nan)
    return np.concatenate((head, rolling_sum), axis=axis)

def rolling_mean(data, window=5, axis=0):
    """
    计算滚动窗口平均值
    输入：
        data : ndarray
        window : 滚动窗口大小(默认为5)
        axis : 计算轴向(默认为0)
    返回：ndarray 形状与输入相同，前window-1个位置填充NaN
    """
    kernel = np.ones(window)
    rolling_mean = convolve1d(data, kernel, axis=axis, mode='constant', cval=0.0) / window
    valid_len = data.shape[axis]-window+1
    beg = np.floor((window-1)/2).astype("int")
    if axis == 0:
        rolling_mean = rolling_mean[beg:beg+valid_len, :]
        head = np.full((window-1, data.shape[1]), np.nan)
    else:
        rolling_mean = rolling_mean[:, :, beg:beg+valid_len]
        head = np.full((data.shape[0], data.shape[1], window-1), np.nan)
    return np.concatenate((head, rolling_mean), axis=axis)

def rolling_weighted_mean_linear_decay(data, window=5, axis=0):
    """
    计算滚动窗口线性衰减加权平均值
    输入：
        data : ndarray
        window : 滚动窗口大小(默认为5)
        axis : 计算轴向(默认为0)
    返回：ndarray 形状与输入相同，前window-1个位置填充NaN
    """
    kernel = np.arange(window, 0, -1).astype(float)
    weighted_sum = convolve1d(data, kernel, axis=axis, mode='constant', cval=0.0)
    valid_len = data.shape[axis]-window+1
    beg = np.floor((window-1)/2).astype("int")
    if axis == 0:
        weighted_sum = weighted_sum[beg:beg+valid_len, :]
        head = np.full((window-1, data.shape[1]), np.nan)
    else:
        weighted_sum = weighted_sum[:, :, beg:beg+valid_len]
        head = np.full((data.shape[0], data.shape[1], window-1), np.nan)
    weighted_avg = weighted_sum / kernel.sum()
    return np.concatenate((head, weighted_avg), axis=axis)

def rolling_std(data, window=5, axis=0):
    """
    计算滚动窗口标准差
    输入：
        data : ndarray
        window : 滚动窗口大小(默认为5)
        axis : 计算轴向(默认为0)
    返回：ndarray 形状与输入相同，前window-1个位置填充NaN
    """
    kernel = np.ones(window)
    mean = convolve1d(data, kernel, axis=axis, mode='constant', cval=0.0) / window    
    square = convolve1d(data**2, kernel, axis=axis, mode='constant', cval=0.0)
    var = square / window - mean ** 2
    var = np.where(var < 0.0, 0.0, var)
    rolling_std = np.sqrt(var)
    beg = np.floor((window-1)/2).astype("int")
    valid_len = data.shape[axis]-window+1
    if axis == 0:
        rolling_std = rolling_std[beg:beg+valid_len, :]
        head = np.full((window-1, data.shape[1]), np.nan)
    else:
        rolling_std = rolling_std[:, :, beg:beg+valid_len]
        head = np.full((data.shape[0], data.shape[1], window-1), np.nan)
    return np.concatenate((head, rolling_std), axis=axis)

def rolling_covariance(data1, data2, window=5, axis=0):
    """
    计算滚动窗口协方差
    输入：
        data1 : ndarray
        data2 : ndarray
        window : 滚动窗口大小(默认为5)
        axis : 计算轴向(默认为0)
    返回：ndarray 形状与输入相同，前window-1个位置填充NaN
    """
    kernel = np.ones(window)
    mean1 = convolve1d(data1, kernel, axis=axis, mode='constant', cval=0.0) / window
    mean2 = convolve1d(data2, kernel, axis=axis, mode='constant', cval=0.0) / window
    mean12 = convolve1d(data1*data2, kernel, axis=axis, mode='constant', cval=0.0) / window
    cov = mean12 - mean1 * mean2        
    beg = np.floor((window-1)/2).astype("int")
    valid_len = data1.shape[axis]-window+1
    if axis == 0:
        cov = cov[beg:valid_len+beg, :]
        head = np.full((window-1, data1.shape[1]), np.nan)
    else:
        cov = cov[:, :, beg:beg+valid_len]
        head = np.full((data1.shape[0], data1.shape[1], window-1), np.nan)
    return np.concatenate((head, cov), axis=axis)

def rolling_corr(data1, data2, window=5, axis=0):
    assert data1.shape == data2.shape
    kernel = np.ones(window)
    mean1 = convolve1d(data1, kernel, axis=axis, mode='constant', cval=0.0) / window
    mean2 = convolve1d(data2, kernel, axis=axis, mode='constant', cval=0.0) / window
    mean12 = convolve1d(data1*data2, kernel, axis=axis, mode='constant', cval=0.0) / window
    square1 = convolve1d(data1**2, kernel, axis=axis, mode='constant', cval=0.0)
    square2 = convolve1d(data2**2, kernel, axis=axis, mode='constant', cval=0.0)
    var1 = square1 / window - mean1 ** 2
    var2 = square2 / window - mean2 ** 2
    var1 = np.where(var1 < 0.0, 0.0, var1)
    var2 = np.where(var2 < 0.0, 0.0, var2)
    std1 = np.sqrt(var1)
    std2 = np.sqrt(var2)
    numerator = mean12 - mean1 * mean2
    denominator = std1 * std2
    denominator[denominator==0.0] = np.nan
    corr = numerator / denominator
    beg = np.floor((window-1)/2).astype("int")
    valid_len = data1.shape[axis]-window+1
    if axis == 0:
        corr = corr[beg:valid_len+beg, :]
        head = np.full((window-1, data1.shape[1]), np.nan)
    else:
        corr = corr[:, :, beg:beg+valid_len]
        head = np.full((data1.shape[0], data1.shape[1], window-1), np.nan)
    return np.concatenate((head, corr), axis=axis)
