import pandas as pd

def read_Fundamental(date, name):
    data = pd.read_parquet("./Fundamental/Lib/{}/{}.parquet".format(name, date))
    return data[name].values.reshape(-1, 12)


def read_Minute(date, name):
    """
    读取分钟数据的单个特征
    
    Parameters:
        date: 日期字符串
        name: 特征名称
        
    Returns:
        numpy.ndarray: 该特征在该日期的数据，形状为(N,)，N为股票数量
    """
    # 读取特征数据
    data = pd.read_parquet("./Minute/Lib/{}/{}.parquet".format(name, date))
    
    # 按SecuCode排序确保顺序一致
    data = data.sort_values('SecuCode')
    
    # 返回特征值，形状为(N,)
    return data[name].values
