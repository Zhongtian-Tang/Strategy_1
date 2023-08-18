# 红利低波项目

**通过对不同的条件进行筛选，筛选出具有高红利，低波动属性的股票池，构造投资组合，计算收益率**

## 基础数据池

### DataFetcher()

DataFetcher() 类包含所有用于数据获取相关的函数，首先获取需要的数据

```python
from stock_selector import DataFetcher
start_year = 2014
end_year = 2023
calender = DataFetcher().get_calender_data(start_year, end_year=2023, endlevel=[(1,2), (1), (1,2), (1,2)]) # 获取每月末交易日历
base_pool = pd.read("codes_map.csv") # 从数据文件夹中获取股票代码之间的映射表
mv_data = DataFetcher().get_mkv_data(start_year, end_year)
turnoverV_data = DataFetcher().get_turnoverV_data(start_year, end_year) # 已经是日均数据
dividend_index_data = DataFetcher().get_dividend_index_data

```

### DataHandler

首先选定回测起止时间，基础股票池与每月月末交易日历，然后完成第一次筛选：

1. 过去一年日均总市值排名前80%
2. 过去一年日均成交额排名前80%
3. 连续三年分红

```python
from stock_selector import DataHandler
mv_avg_data = DataHandler().mv_handler(mv_data, base_pool, calender) #获得240个交易日范围内日均市值数据
turnoverV_avg_data = DataHandler().turnoverV_handler(turnoverV_data, base_pool)
dividend_index = DataHandler().dividend_index_handler(dividend_index_data , base_pool, calender)
```

### Selector()

```python
base_pool = select_base_pool(mv_avg_data, turnoverV_avg_data, dividend_index)
```

## 筛选指标

### DataFetcher()

**股利支付率**

$$
se = \frac{dps}{eps} = \frac{dps \times pe}{price} = pe \times dividendRatio
$$

根据公式从JYDB.LC_DIndicesForValuation获取数据，计算每一期的股利支付率

**股利增长率（3年）：**

计算过去三年每股股利增长率，选取线性回归斜率大于0的股票

```python
from stock_selector import DataFetcher()
payratio_data = DataFetcher().get_payratio_data(calender)
dividend_data = DataFetcher().get_dividend_data(start_year-3, end_year)
```

### StockSelctor()

```python
from stock_selector import DataHandler(), Selector()
payratio_data = DataHandler().payratio_handler(payratio_data, base_pool)
dividend_delta_data = DataHandler().dividend_delta_data(dividend_data, base_pool)
filter_pool_1 = Selector().filter_1(stocks_pool, payratio_data, devidend_delta_data)
```
