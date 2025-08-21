# 标准库导入
import sys
from pathlib import Path

# 第三方库导入
import numpy as np
import pandas as pd
import connectorx as cx
from dotenv import load_dotenv

# 本地模块导入
from save import *

# 父级模块导入
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import *

# 加载环境变量
load_dotenv()


def process_Fundamental(date, name, secucode):

    feature_name = name[12:] #Item1002

    query = 'select * from smartquant.returndaily where datadate ={} and IfTradingDay = 1'.format(date)
    data = read_ob(query)

    data = pd.read_parquet("./Fundamental/raw/{}/{}.parquet".format(name, date))
    data[['CumLatest', 'Quarterly', 'TTM']] = data[['CumLatest', 'Quarterly', 'TTM']].astype('float')
    data['DataDate'] = pd.to_datetime(data['DataDate'])
    returndaily = pd.read_parquet("./ReturnDaily/{}.parquet".format(date))
    mapping = pd.Series(returndaily['SecuCode'].values, index=returndaily['InnerCode'].values)
    data['SecuCode'] = data['InnerCode'].map(mapping)
    data = data.loc[~data['SecuCode'].isna()]  # 删除映射失败的行
    
    new_data = []
    for i in range(12):
        """
        按月份处理数据：循环12次，处理12个不同的EndDateRank
        """
        subset = data.loc[data['EndDateRank']==i+1]
        subset = subset.sort_values("SecuCode", ascending=True)
        subset_ = pd.DataFrame(np.full((len(secucode), data.shape[1]), np.nan), columns=data.columns)
        
        # 设置特定列的数据类型
        subset_ = subset_.astype({'DataDate': 'datetime64[ns]', 'SecuCode': 'object', "InfoPublDate": 'object', "EndDate": "object",
                             "UpdateTime": 'datetime64[ns]'})

        """
        检查secucode中的每个证券代码是否存在于当前季度的实际数据中

        返回一个布尔型Series，长度与secucode相同，True表示该证券在当前财务期间有数据
        """

        cond = secucode.isin(subset['SecuCode'].values.tolist())
        subset_.loc[cond] = subset.values
        subset_.loc[:, 'DataDate'] = pd.to_datetime(date)
        subset_.loc[:, 'SecuCode'] = secucode.values
        subset_.loc[:, 'EndDateRank'] = i+1
        new_data.append(subset_)
    new_data = pd.concat(new_data) 
    
    # 按证券代码和EndDateRank排序
    new_data = new_data.sort_values(["SecuCode", "EndDateRank"], ascending=[True, False])
    save_Fundamental(new_data, date, feature_name)




def process_Minute(date, name, secucode):
    """
    处理分钟数据
    Parameters:
        date: 日期字符串
        name: 特征名称（None表示处理该日期的所有特征）
        secucode: 股票代码序列
    """
    data = pd.read_parquet("./Minute/raw/{}.parquet".format(date))
    data['DataDate'] = pd.to_datetime(data['DataDate'])

    # 获取交易日股票池
    query = 'select * from smartquant.returndaily where datadate ={} and IfTradingDay = 1'.format(date)
    universe = read_ob(query)
    
    if 'security_code' in data.columns:
        data = data.rename(columns={'security_code': 'SecuCode'})
    
    # 只保留在交易日股票池中的股票
    data = data[data['SecuCode'].isin(universe['SecuCode'])]
    
    # 填充为完整股票池
    full_universe = pd.DataFrame({
        'SecuCode': secucode.values,
        'DataDate': pd.to_datetime(date)
    })
    
    new_data = full_universe.merge(
        data, 
        on=['SecuCode', 'DataDate'], 
        how='left'
    )

    factor_columns = [col for col in data.columns if col not in ['DataDate', 'SecuCode']]
    new_data[factor_columns] = new_data[factor_columns].fillna(np.nan)
    new_data = new_data[['DataDate', 'SecuCode'] + factor_columns]
    new_data = new_data.sort_values(["SecuCode"], ascending=True)

    # 为每个特征单独保存
    save_Minute(new_data, date, factor_columns)
