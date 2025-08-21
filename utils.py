import os
import numpy as np
import pandas as pd
import connectorx as cx
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

def read_ob(sql_query):
    conn = "mysql://{}%40public%23Thetis:{}@192.168.55.161:2883/".format(os.getenv("OB_USER"), os.getenv("OB_PASSWORD"))
    df = cx.read_sql(conn, sql_query)
    return df

def get_secucode(start, end):
    sql_query = "select SecuCode from SmartQuant.ReturnDaily where DataDate >= '{}' and DataDate <= '{}'".format(start, end)
    df = read_ob(sql_query)
    return np.sort(df['SecuCode'].unique()).tolist()

def get_month_first_trading_day(start, end):
    query = "SELECT * FROM SmartQuant.CalenderDay_TradingDay WHERE DataDate >= '{}' AND DataDate <= '{}'".format(start, end)
    df = read_ob(query)
    df = df.loc[df['IfTradingDay'] == 1]
    df['month'] = df['DataDate'].dt.month
    df = df.loc[df['month'].diff() != 0]  #每月首日数据
    df['DataDate'] = df['DataDate'].apply(lambda x: x.strftime('%Y%m%d'))
    return df['DataDate'].values.tolist() 

# def get_trading_day(start, end, margins=0):
#     engine = create_engine('mssql+pymsql://{}:{}@192.168.55.18/SmartQuant'.format(os.getenv("MS_USER"), os.getenv("MS_PASSWORD")))
#     qusery1 = "SELECT * FROM SmartQuant..CalenderDay_TradingDay WHERE DataDate >= '{}' AND DataDate <= '{}'".format(start, end)
#     df1 = pd.read_sql(query1, engine)
#     df1 = df1.loc[df1['IfTradingDay'] == 1]
#     df1['DataDate'] = df1['DataDate'].apply(lambda x: x.strftime('%Y%m%d'))
    
#     # 额外获取结束日期之后的若干个（margin）交易日
#     query2 = "SELECT * FROM SmartQuant..CalenderDay_TradingDay WHERE DataDate > '{}'".format(end)
#     df2 = pd.read_sql(query2, engine)
#     df2 = df2.loc[df2['IfTradingDay'] == 1]
#     df2['DataDate'] = df2['DataDate'].apply(lambda x: x.strftime('%Y%m%d'))
#     df2 = df2.iloc[:margins, :]
#     return df1['DataDate'].values.tolist() + df2['DataDate'].values.tolist()    


def get_trading_day(start, end, margins=0):
    query1 = "SELECT * FROM SmartQuant.CalenderDay_TradingDay WHERE DataDate >= '{}' AND DataDate <= '{}'".format(start, end)
    df1= read_ob(query1)
    df1 = df1.loc[df1['IfTradingDay'] == 1]
    df1['DataDate'] = df1['DataDate'].apply(lambda x: x.strftime('%Y%m%d'))
    
    # 额外获取结束日之前的若干个（margin）交易日
    query2 = "SELECT * FROM SmartQuant.CalenderDay_TradingDay WHERE DataDate > '{}'".format(end)
    df2= read_ob(query2)
    df2 = df2.loc[df2['IfTradingDay'] == 1]
    df2['DataDate'] = df2['DataDate'].apply(lambda x: x.strftime('%Y%m%d'))
    df2 = df2.iloc[:margins, :]
    return df1['DataDate'].values.tolist() + df2['DataDate'].values.tolist()    


if __name__  ==  "__main__":

    query = "SELECT * FROM SmartQuant.CalenderDay_TradingDay WHERE DataDate >= '2024-03-05' AND DataDate <= '2024-03-10'"

    df = read_ob(query)

    # a = os.getenv("OB_USER")

    print(df.head())