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
    
    def get_calender_data(self, start_date: str, end_date: str, secumarket: List[int] = [(83)], endlevel: list = [(1,2),(1,2),(1,2),(1,2)]):
        """
        获取交易日历数据
        endlevel 为列表元组,不加限制条件就是(1,2)
        """
        query = f"""
        SELECT TradingDate AS TradingDay
        FROM JYDB.QT_TradingDayNew
        WHERE TradingDate >= TO_TIMESTAMP('{start_date}', 'YYYY-MM-DD HH24:MI:SS.FF')
        AND TradingDate <= TO_TIMESTAMP('{end_date}', 'YYYY-MM-DD HH24:MI:SS.FF')
        AND IfTradingDay = 1 AND IfWeekEnd IN {endlevel[0]} AND IfMonthEnd IN {endlevel[1]} AND IfQuarterEnd IN {endlevel[2]} AND IfYearEnd IN {endlevel[3]}
        AND SecuMarket = 83
        ORDER BY TradingDate
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
        获取股票市值数据
        """
        query_zb = f"""
        SELECT TRADINGDAY, INNERCODE, TOTALMV FROM JYDB.QT_StockPerformance
        WHERE EXTRACT(YEAR FROM TRADINGDAY) BETWEEN {start_year} AND {end_year}
        """
        query_kc = f"""
        SELECT TRADINGDAY, INNERCODE, TOTALMV FROM JYDB.LC_STIBPerformance
        WHERE EXTRACT(YEAR FROM TRADINGDAY) BETWEEN {start_year} AND {end_year}
        """
        df_zb = pd.read_sql(query_zb, self.conn)
        df_kc = pd.read_sql(query_kc, self.conn)
        df_final = pd.concat([df_zb, df_kc], axis=0)
        return df_final
    
    def get_turnoverV_data(self, start_year: int, end_year: int):
        query_zb = f"""
        WITH MonthData AS (
        SELECT TRADINGDATE AS TRADINGDAY
        FROM JYDB.QT_TradingDayNew
        WHERE EXTRACT(YEAR FROM TRADINGDATE) BETWEEN {start_year} AND {end_year}
        AND IfTradingDay = 1 AND IfMonthEnd = 1
        AND SecuMarket = 83
        ORDER BY TRADINGDATE
        )
        SELECT m.TRADINGDAY, p.INNERCODE, p.TurnoverValuePDayRY
        FROM MonthData m
        JOIN JYDB.QT_StockPerformance p
        ON m.TRADINGDAY = p.TRADINGDAY
        ORDER BY m.TRADINGDAY, p.INNERCODE
        """
        query_kc = f"""
        WITH MonthData AS (
        SELECT TRADINGDATE AS TRADINGDAY
        FROM JYDB.QT_TradingDayNew
        WHERE EXTRACT(YEAR FROM TRADINGDATE) BETWEEN {start_year} AND {end_year}
        AND IfTradingDay = 1 AND IfMonthEnd = 1
        AND SecuMarket = 83
        ORDER BY TRADINGDATE
        )
        SELECT m.TRADINGDAY, p.INNERCODE, p.TurnoverValuePDayRY
        FROM MonthData m
        JOIN JYDB.LC_STIBPerformance p
        ON m.TRADINGDAY = p.TRADINGDAY
        ORDER BY m.TRADINGDAY, p.INNERCODE
        """
        df_zb = pd.read_sql(query_zb, self.conn)
        df_kc = pd.read_sql(query_kc, self.conn)
        df_final = pd.concat([df_zb, df_kc], axis=0)
        return df_final
    
    def get_dividend_index_data(self, start_year: int, end_year: int):
         query_zb = f"""
         SELECT INNERCODE, ENDDATE, IFDIVIDEND FROM JYDB.LC_Dividend
         WHERE EXTRACT(YEAR FROM ENDDATE) BETWEEN {start_year} AND {end_year}
         """
         query_kc = f"""
         SELECT INNERCODE, ENDDATE, IFDIVIDEND FROM JYDB.LC_STIBDividend
         WHERE EXTRACT(YEAR FROM ENDDATE) BETWEEN {start_year} AND {end_year}
         """
         df_zb = pd.read_sql(query_zb, self.conn)
         df_kc = pd.read_sql(query_kc, self.conn)
         df_final = pd.concat([df_zb, df_kc], axis=0)
         return df_final
    
    def get_payratio_data(self, calender: pd.DataFrame):
         """
         获取股利支付率数据
         """
         data_strings = calender['tradingday'].dt.strftime('%Y-%m-%d').tolist()
         dates_for_query = ",".join(["TO_DATE('{}', 'YYYY-MM-DD')".format(date) for date in data_strings])
         query_zb = f"""
         SELECT INNERCODE, TRADINGDAY, PE*DividendRatio AS PAYRATIO
         FROM JYDB.LC_DIndicesForValuation
         WHERE TRADINGDAY IN ({dates_for_query})
         ORDER BY TRADINGDAY, INNERCODE
         """
         query_kc = f"""
         SELECT INNERCODE, TRADINGDAY, PETTM*DividendRatioTTM AS PAYRATIO
         FROM JYDB.LC_STIBDIndiForValue
         WHERE TRADINGDAY IN ({dates_for_query})
         ORDER BY TRADINGDAY, INNERCODE
         """
         df_zb = pd.read_sql(query_zb, self.conn)
         df_kc = pd.read_sql(query_kc, self.conn)
         df_final = pd.concat([df_zb, df_kc], axis=0)
         return df_final
    
    def get_dividend_data(self, start_year:int, end_year:int):
        query_zb = f"""
        SELECT COMPANYCODE, ENDDATE, DIVIDENDPS FROM JYDB.LC_MainIndexNew
        WHERE EXTRACT(YEAR FROM ENDDATE) BETWEEN {start_year} AND {end_year}
        ORDER BY COMPANYCODE, ENDDATE
        """
        query_kc = f"""
        SELECT COMPANYCODE, ENDDATE, DIVIDENDPS FROM JYDB.LC_STIBMainIndex
        WHERE EXTRACT(YEAR FROM ENDDATE) BETWEEN {start_year} AND {end_year}
        ORDER BY COMPANYCODE, ENDDATE    
        """
        df_zb = pd.read_sql(query_zb, self.conn)
        df_kc = pd.read_sql(query_kc, self.conn)
        df_final = pd.concat([df_zb, df_kc], axis=0)
        df_final['year'] = df_final['enddate'].dt.year
        df_final.rename(columns={'enddate': 'tradingday'}, inplace=True)
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
    def mv_handler(self, df: pd.DataFrame, stock_pool: pd.DataFrame, calender: pd.DataFrame):
        """
        处理市值数据, 返还csv表格
        """
        mkv_data = df
        def get_avg_mkv(df: pd.DataFrame):
            df_sorted = df.sort_values(by=['innercode', 'tradingday'])
            df_sorted['avg_market_past_year'] = df_sorted.groupby('innercode')['totalmv'].rolling(window=240,min_periods=1).mean().reset_index(level=0,drop=True)
            return df_sorted
        mkv_data = get_avg_mkv(mkv_data)
        step_1 = pd.merge(mkv_data, stock_pool, on='innercode', how='inner')
        step_2 = pd.merge(step_1, calender, on='tradingday', how='inner')[['wind_code', 'tradingday', 'avg_market_past_year']]
        step_2['avg_market_past_year'] = step_2['avg_market_past_year'].apply(lambda x: '{:.2f}'.format(x))
        step_2['tradingday'] = step_2['tradingday'].astype(str)
        step_2['tradingday'] = step_2['tradingday'].apply(lambda x: x.replace('-', ''))
        df_final = step_2.pivot(index='wind_code', columns='tradingday', values='avg_market_past_year')
        # df_final = df_final.fillna('')
        df_final.columns.name = None
        df_final = df_final.astype(np.float64)
        return df_final
    
    def top_rank(self, df: pd.DataFrame, ratio: float = 0.8):
         """
         选择每个月末交易日排名前80%的股票
         """
         def top_rank_for_column(col):
              valid_data = col.dropna()
              ranked_data = valid_data.rank(ascending=False, method='first')
              threshold_rank = ratio * len(valid_data)
              return ranked_data[ranked_data <= threshold_rank].index
         top_rank_stock = df.apply(top_rank_for_column, axis=0)
         top_rank_stock_df = top_rank_stock.apply(pd.Series).T
         return top_rank_stock_df

    
    def turnoverV_handler(self, df: pd.DataFrame, stock_pool: pd.DataFrame):
         
         """
         处理日均成交额数据
         """
         turnoverV_data = df
         step_1 = pd.merge(turnoverV_data, stock_pool, on='innercode', how='inner')[['wind_code', 'tradingday', 'turnovervaluepdayry']]
         step_1['turnovervaluepdayry'] = step_1['turnovervaluepdayry'].apply(lambda x: '{:.2f}'.format(x))
         step_1['tradingday'] = step_1['tradingday'].astype(str)
         step_1['tradingday'] = step_1['tradingday'].apply(lambda x: x.replace('-', ''))
         df_final = step_1.pivot(index='wind_code', columns='tradingday', values='turnovervaluepdayry')
         df_final.columns.name = None
         df_final = df_final.astype(np.float64)
         return df_final
    
    def dividend_index_handler(self, df: pd.DataFrame, stock_pool: pd.DataFrame, calender: pd.DataFrame):
            """
            处理分红数据
            """
            did_data = df
            did_data['year'] = did_data['enddate'].dt.year
            did_data['status'] = np.where(did_data['ifdividend'].isin([0, 24]), 0, 1)
            did_data = did_data.groupby(['innercode', 'year'])['status'].sum().reset_index()
            def rolling_check(group):
                 return group.rolling(window=4).apply(lambda x: 1 if (x[:-2] > 0).all() else 0, raw=True)
            did_data['dividend_index'] = did_data.groupby('innercode')['status'].apply(rolling_check).reset_index(level=0, drop=True)
            did_index = did_data[['innercode', 'year', 'dividend_index']].fillna(0)
            did_index['year'] = did_index['year'].astype(np.int32)
            calender['year'] = calender['tradingday'].dt.year
            step_1 = pd.merge(did_index, calender, on='year', how='inner')
            step_2 = pd.merge(step_1, stock_pool, on='innercode', how='inner').dropna()[['wind_code', 'tradingday', 'dividend_index']]
            step_2['tradingday'] = step_2['tradingday'].astype(str)
            step_2['tradingday'] = step_2['tradingday'].apply(lambda x: x.replace('-', ''))
            df_final = step_2.pivot(index='wind_code', columns='tradingday', values='dividend_index')
            df_final.columns.name = None
            df_final = df_final.fillna(0)
            df_final = df_final.astype(np.int32)
            return df_final
    
    def payratio_handler(self, df: pd.DataFrame, stock_pool: pd.DataFrame):
            """
            处理股利支付率数据
            """
            payratio_data = df
            step_1 = pd.merge(payratio_data, stock_pool, on='innercode', how='inner')[['wind_code', 'tradingday', 'payratio']]
            step_1['payratio'] = step_1['payratio'].apply(lambda x: '{:.2f}'.format(x))
            step_1['tradingday'] = step_1['tradingday'].astype(str)
            step_1['tradingday'] = step_1['tradingday'].apply(lambda x: x.replace('-', ''))
            df_final = step_1.pivot(index='wind_code', columns='tradingday', values='payratio')
            df_final.columns.name = None
            df_final = df_final.astype(np.float64)
            return df_final
    
    def dividend_delta_handler(self, df: pd.DataFrame, stock_pool: pd.DataFrame):
            """
            处理股利数据
            """
            dividend_data = pd.merge(df, stock_pool, on='companycode', how='inner')[['wind_code', 'tradingday', 'year','dividendps']]
            dividend_year_data = dividend_data.groupby(['wind_code', 'year'])['dividendps'].sum()
            _index = dividend_year_data.index
            dividend_year_data = dividend_year_data.reset_index()
            def rolling_regression(y):
                 x = np.arange(len(y))
                 slope, _intercept = np.polyfit(x, y, 1)
                 return slope
            dividend_year_delta = dividend_year_data.groupby('wind_code')['dividendps'].rolling(window=3).apply(rolling_regression, raw=True)
            dividend_year_delta.index = _index
            return dividend_year_delta
         



        



class StockSelector:
    def __init__(self, conn):
        self.conn = conn


    def select_base_pool(self, mv_data:pd.DataFrame, turnoverV_data:pd.DataFrame, dividend_data:pd.DataFrame):
        """
        根据日均市值,成交额与是否连续分红,筛选出基础股票池
        """
        mv_rank = DataHandler().top_rank(mv_data, 0.8)
        turnoverV_rank = DataHandler().top_rank(turnoverV_data, 0.8)
        dividend_rank = dividend_data.apply(lambda col: col[col==1].index).apply(pd.Series).T
        stock_pool = []
        for i in range(len(mv_rank.columns)):
            common_stocks = np.intersect1d(mv_rank.iloc[:,i].dropna(), np.intersect1d(turnoverV_rank.iloc[:,i].dropna(), dividend_rank.iloc[:,i].dropna()))
            stock_pool.append(common_stocks)
        final_result = pd.DataFrame(stock_pool).T
        final_result.columns = mv_rank.columns
        return final_result
    
    def filter_1(self, base_pool: pd.DataFrame, payratio_data: pd.DataFrame, dividend_delta_data: pd.DataFrame, ratio: float = 0.05):
        """
        剔除股利支付率前5%以及前三年股利增长率小于0的股票
        """
        filter_stocks = []
        payratio_data = payratio_data.set_index('wind_code')
        for date in base_pool.columns:
             base_stocks = base_pool[date].dropna()
             positive_dividend_growth = dividend_delta_data[(dividend_delta_data['year'] == int(date[:4])) & (dividend_delta_data['dividendps'] > 0)]['wind_code'].unique()
             filter_1 = np.intersect1d(base_stocks, positive_dividend_growth)
             sorted_stocks = payratio_data[date].dropna().sort_values(ascending=False)
             top_5_percent_count = int(len(sorted_stocks) * ratio)
             payratio_selected_stocks = sorted_stocks.iloc[top_5_percent_count:].index
             filter_2 = np.intersect1d(base_stocks, payratio_selected_stocks)
             final_filter =np.intersect1d(filter_1, filter_2)
             filter_stocks.append(pd.Series(final_filter, name=date))
        df = pd.concat(filter_stocks, axis=1)
        return df




