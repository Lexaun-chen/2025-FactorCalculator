import os
import numpy as np
import pandas as pd

def save_Fundamental(data, date, feature_name):
    """
    保存基本面数据，每个特征的不同维度数据各保存为单独路径下的一个文件
        
    Parameters:
        data: DataFrame，包含DataDate, SecuCode和某特征三种维度数据
        date: 日期字符串
        feature_name: 特征名
    """

    for col in ('CumLatest', 'Quarterly', 'TTM'):

        df = data[['DataDate', 'SecuCode', 'EndDateRank', col]]
        df = df.rename({col: '{}_{}'.format(feature_name, col)}, axis=1)

        if not os.path.exists("./Fundamental/Lib/{}_{}".format(feature_name, col)):
            os.makedirs("./Fundamental/Lib/{}_{}".format(feature_name, col))

        df.to_parquet("./Fundamental/Lib/{}_{}/{}.parquet".format(feature_name, col, date), index=False)


def save_Minute(data, date, factor_columns):
    """
    保存分钟数据，每个特征单独保存为一个文件
    
    Parameters:
        data: DataFrame，包含DataDate, SecuCode和各个特征列
        date: 日期字符串
        factor_columns: 特征列名列表
    """
    # 为每个特征创建目录并保存数据
    for feature_name in factor_columns:

        feature_dir = "./Minute/Lib/{}".format(feature_name)
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        
        # 提取该特征的数据
        feature_data = data[['DataDate', 'SecuCode', feature_name]].copy()
        
        # 保存为parquet文件
        feature_data.to_parquet("{}/{}.parquet".format(feature_dir, date), index=False)
