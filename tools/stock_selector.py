import pandas as pd
import numpy as np
from typing import List, Union
from iFinDPy import *
from sqlalchemy import create_engine
import sys
sys.path.append(r'C:\Users\hazc\Desktop\Concept-Factor\dependencies')
import concept_helper as cp


class DataFetcher:
    def __init__(self):
        self.connection_str = 'oracle+cx_oracle://rejy:jcFXLzL10@10.224.6.3:1522/?service_name=orcl'
        self.engine = create_engine(self.connection_str)
        self.conn = self.engine.connect()

    def get_daily_data(self, start_date: str, end_date: str, stock_codes: List[str] = None):
        """
        获取股票日行情数据
        """
        query = f"""
        SELECT InnerCode, TradingDay, ClosePrice, TurnoverVolume, TurnoverValue, Ifsuspend, TurnoverValuePDayRY, TotalMV
        FROM JYDB.QT_DailyQuote
        WHERE TradingDay >= TO_TIMESTAMP('{start_date}', 'YYYY-MM-DD HH24:MI:SS.FF')
        AND TradingDay <= TO_TIMESTAMP('{end_date}', 'YYYY-MM-DD HH24:MI:SS.FF')
        AND InnerCode IN {{','.join(stock_codes)}}
        ORDER BY TradingDay
        """
        return pd.read_sql(query, self.conn)
    def get_main_data(self, start_date: str, end_date: str):
        """
        获取股票行情数据
        """
        query = f"""
        SELECT InnerCode, TradingDay, ClosePrice, TurnoverVolume, TurnoverValue, Ifsuspend, TurnoverValuePDayRY, TotalMV
        FROM JYDB.QT_StockPerformance
        WHERE TradingDay >= TO_TIMESTAMP('{start_date}', 'YYYY-MM-DD HH24:MI:SS.FF')
        AND TradingDay <= TO_TIMESTAMP('{end_date}', 'YYYY-MM-DD HH24:MI:SS.FF')
        ORDER BY TradingDay, InnerCode
        """
        return pd.read_sql(query, self.conn)
    
    def get_calender_data(self, start_date: str, end_date: str, secumarket: List[int] = [83, 90], endlevel: str = 'Week'):
        """
        获取交易日历数据
        """
        query = f"""
        SELECT TradingDate, IfTradingDay, If{endlevel}End
        FROM JYDB.QT_TradingDayNew
        WHERE TradingDay >= TO_TIMESTAMP('{start_date}', 'YYYY-MM-DD HH24:MI:SS.FF')
        AND TradingDay <= TO_TIMESTAMP('{end_date}', 'YYYY-MM-DD HH24:MI:SS.FF')
        AND SecuMarket IN {{','.join(map(str, secumarket))}}
        ORDER BY TradingDay
        """
        return pd.read_sql(query, self.conn)

    def get_codes_map(self):
        """
        获取股票代码映射表
        """
        query = """
        SELECT InnerCode, SecuCode, SecuAbbr, ChiName, COMPANYCODE
        FROM JYDB.SecuMain
        WHERE SecuCategory = 1
        AND SecuMarket IN (18, 83, 90)
        """
        return pd.read_sql(query, self.conn)
    
    def get_base_pool(self, date:str):
         """
         筛选出基础股票池
         """
         cp.thslogin()
         codes_ls = THS_DP('block', '{};001005010'.format(date), 'date:N,thscode:Y,security_name:N').data[['THSCODE']]
         codes_ls['secucode'] = codes_ls['THSCODE'].apply(lambda x: x.split('.')[0])
         inner_codes = self.get_codes_map()
         inner_codes['innercode'] = inner_codes['innercode'].astype(np.int64)
         df = pd.merge(codes_ls, inner_codes, on='secucode', how='inner')[['THSCODE', 'innercode']]
         df.rename(columns={'THSCODE': 'wind_code'}, inplace=True)
         return df
    
    def get_mkv_data(self, start_year: int, end_year:int):
        """
        获取股票每月末交易日的前一年日均市值数据
        """
        query_zb = f"""
        WITH ALLTradingDays AS (
            SELECT TradingDay, INNERCODE, TOTALMV
            FROM JYDB.QT_STOCKPERFORMANCE s
            WHERE SecuMarket IN 83
            AND EXTRACT(YEAR FROM TradingDay) BETWEEN {start_year} AND {end_year}
            ),
            MonthlyLastTrade AS (
            SELECT TradingDate AS LAST_TRADE_DAY
            FROM JYDB.QT_TradingDayNew
            WHERE IfTradingDay = 1 AND IfMonthEnd = 1 AND SecuMarket IN 83
            AND EXTRACT(YEAR FROM TradingDate) BETWEEN {start_year} AND {end_year}
            ),
            YearlyData AS (
            SELECT
            m.LAST_TRADE_DAY,
            a.INNERCODE,
            a.TRADINGDAY,
            a.TOTALMV
            FROM MonthlyLastTrade m
            JOIN ALLTradingDays a
            ON a.TradingDay <= m.LAST_TRADE_DAY AND a.TradingDay >= DATE_SUB(m.LAST_TRADE_DAY, INTERVAL 1 YEAR)
            ),
            AVGMarket AS (
            SELECT
            y.INNERCODE,
            y.LAST_TRADE_DAY,
            AVG(y.TOTALMV) OVER (PARTITION BY y.INNERCODE, y.LAST_TRADE_DAY) AS avg_market_past_year
            FROM YearlyData y
            )
            SELECT * FROM AVGMarket
        """
        query_kc = f"""
        WITH ALLTradingDays AS (
            SELECT TradingDay, INNERCODE, TOTALMV
            FROM JYDB.LC_STIBPerformance s
            WHERE SecuMarket IN 83
            AND EXTRACT(YEAR FROM TradingDay) BETWEEN {start_year} AND {end_year}
            ),
            MonthlyLastTrade AS (
            SELECT TradingDate AS LAST_TRADE_DAY
            FROM JYDB.QT_TradingDayNew
            WHERE IfTradingDay = 1 AND IfMonthEnd = 1 AND SecuMarket IN 83
            AND EXTRACT(YEAR FROM TradingDate) BETWEEN {start_year} AND {end_year}
            ),
            YearlyData AS (
            SELECT
            m.LAST_TRADE_DAY,
            a.INNERCODE,
            a.TRADINGDAY,
            a.TOTALMV
            FROM MonthlyLastTrade m
            JOIN ALLTradingDays a
            ON a.TradingDay <= m.LAST_TRADE_DAY AND a.TradingDay >= DATE_SUB(m.LAST_TRADE_DAY, INTERVAL 1 YEAR)
            ),
            AVGMarket AS (
            SELECT
            y.INNERCODE,
            y.LAST_TRADE_DAY,
            AVG(y.TOTALMV) OVER (PARTITION BY y.INNERCODE, y.LAST_TRADE_DAY) AS avg_market_past_year
            FROM YearlyData y
            )
            SELECT * FROM AVGMarket
        """
        df_zb = pd.read_sql(query_zb, self.conn)
        df_kc = pd.read_sql(query_kc, self.conn)
        df_final = pd.concat([df_zb, df_kc], axis=0)
        return df_final
    
    
class DataCleaner:

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
    
    def set_index(self, columns_name = ['InnerCode', 'TradingDay']):
                  self.data.set_index(columns_name, inplace=True)
                  return self
    
    def handle_missing(self, method: str = 'fill', fill_value: float = np.nan):
        """
        处理缺失值
        """
        if method == 'drop':
             self.data.dropna(inplace=True)
        elif method == 'fill':
             self.data.fillna(value=fill_value, inplace=True)
        elif method == 'ffill':
             self.data.fillna(method='ffill', inplace=True)
        return self
    
    def remove_duplicates(self):
        """
        去除重复值
        """
        self.data.drop_duplicates(inplace=True)
        return self
    
    def convert_dtype(self, columns_name, dtype: str):
        """
        转换数据类型
        """
        self.data[columns_name] = self.data[columns_name].astype(dtype)
        return self
    
    def get_data(self):
        return self.data
    
class DataHandler:
    def mv_handler(self, start_year: int, end_year: int, stock_pool: pd.DataFrame ):
        """
        处理市值数据, 返还csv表格
        """
        mkv_data = DataFetcher().get_mkv_data(start_year, end_year)
        df_final = pd.merge(mkv_data, stock_pool, on='innercode', how='inner')[['wind_code', 'last_trade_day', 'avg_market_past_year']]
        df_final['avg_market_past_year'] = df_final['avg_market_past_year'].apply(lambda x: '{:.2f}'.format(x))
        df_final['last_trade_day'] = df_final['last_trade_day'].astype(str)
        df_final['last_trade_day'] = df_final['last_trade_day'].apply(lambda x: x.replace('-', ''))
        df_final = df_final.pivot(index='wind_code', columns='last_trade_day', values='avg_market_past_year')
        # df_final = df_final.fillna('')
        return df_final

        



class StockSelector:
    def __init__(self, conn):
        self.conn = conn


    def select_stock(self, date):
        """
        根据不同条件的筛选结果, 筛选交集股票池
        """

    def select_top_market_cap(self, date: str, date_range: str, quantile: float = 0.8):
        """
        选择不同时间范围, 选取市值前quantile的股票
        """


