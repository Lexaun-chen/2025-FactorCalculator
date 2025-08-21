import os
import numpy as np
import pandas as pd
import duckdb
import connectorx as cx
from dotenv import load_dotenv
from sqlalchemy import create_engine
from pathlib import Path  

load_dotenv()

def download_Fundamental(date, name, mode='oceanbase'):
    if mode == "sqlserver":
        engine = create_engine('mssql+pymssql://{}:{}@192.168.55.18/Fundamental'.format(os.getenv("MS_USER"), os.getenv("MS_PASSWORD")))
        query = "SELECT * FROM Fundamental..{} WHERE DataDate = '{}'".format(name, date)
        data = pd.read_sql(query, engine)
        data.to_parquet(os.path.join("./Fundamental/raw", name, "{}.parquet".format(date)), index=False)
    elif mode == "oceanbase":
        conn = "mysql://{}%40public%23Thetis:{}@192.168.55.161:2883/".format(os.getenv("OB_USER"), os.getenv("OB_PASSWORD"))
        sql_query = "select * from Fundamental.{} where DataDate = '{}'".format(name, date)
        data = cx.read_sql(conn, sql_query)
        data.to_parquet(os.path.join("./Fundamental/raw", name, "{}.parquet".format(date)), index=False)
    else:
        raise NameError
        

def cal_miniute_data(file_pth):
    """
    使用原始分钟数据生成自定义特征
    file_pth: 原始分钟数据文件路径（parquet格式）
    Returns:
        - factor_df 因子数据框
            第一列: 'security_code' 
            第二列: 'DataDate' 
            后续列: 自定义特征列
        - features_list (pd.Index): 特征列名列表
    """
    df = duckdb.sql(f"select security_code, trading_day, start_time, minute_return from read_parquet('{file_pth}') order by security_code, start_time").df()
    
    # 计算各种日内特征
    early_ret = df.groupby('security_code')['minute_return'].apply(lambda x: np.prod(x.head(6) + 1) - 1)  # 开盘前6分钟累计收益
    tail_ret = df.groupby('security_code')['minute_return'].apply(lambda x: np.prod(x.tail(6) + 1) - 1)   # 收盘前6分钟累计收益
    max_ret = df.groupby('security_code')['minute_return'].max()      # 日内最大收益率
    min_ret = df.groupby('security_code')['minute_return'].min()      # 日内最小收益率
    mean_ret = df.groupby('security_code')['minute_return'].mean()    # 日内平均收益率
    intra_vol = df.groupby('security_code')['minute_return'].std()    # 日内波动率
    intra_skew = df.groupby('security_code')['minute_return'].skew()  # 日内偏度
    
    dt = df['trading_day'].iloc[0]
    
    factor = pd.concat([early_ret, tail_ret, max_ret, min_ret, mean_ret,
                        intra_vol, intra_skew],
                       axis=1, join='outer')

    factor.columns = ['early_ret', 'tail_ret', 'max_ret', 'min_ret', 'mean_ret',
                      'intra_vol', 'intra_skew']
    
    factor['DataDate'] = dt
    factor = factor.reset_index()
    features = factor.columns

    return factor, features


def download_Minute(date, mode='five_minute'):
    
    fold_path = Path('/data/cephfs/minute') / str(mode)  

    if not fold_path.exists():
        raise FileNotFoundError(f"指定的目录不存在: {fold_path}")  
    elif not fold_path.is_dir():
        raise NotADirectoryError(f"路径不是有效目录: {fold_path}")  

    file_path = fold_path / f"{date}.parquet"  
    data, features = cal_miniute_data(file_path)
    data.to_parquet(os.path.join("./Minute/raw", "{}.parquet".format(date)), index=False)
