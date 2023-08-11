# 红利低波项目

**通过对不同的条件进行筛选，筛选出具有高红利，低波动属性的股票池，构造投资组合，计算收益率**

## 模组介绍

### stock_selector

这个脚本包含四个不同的class：DataFetcher(), DataHandler(), DataCleaner(), XXX(), 分别实现数据库数据提取，数据处理，数据清洗，收益回测四种功能。

```python
from stock_selector import DataHandler, DataFetcher
start_year = 2014
end_year = 2023
trading_calender = DataFetcher().get_calender_data(start_year, end_year=2023, endlevel=[(1,2), (1), (1,2), (1,2)])
base_pool = pd.read("codes_map.csv")
```

首先选定回测起止时间，基础股票池与每月月末交易日历，然后完成第一次筛选：

1. 过去一年日均总市值排名前80%
2. 过去一年日均成交额排名前80%
3. 连续三年分红

```python
mv_data = DataHandler().mv_data(start_year, end_year, base_pool, calender)
turnoverV_data = DataHandler().turnoverV_handler(start_year, end_year, base_pool)
turnoverV_rank = top_rank(turnoverV_data, 0.8)
mv_rank = top_rank(mv_data, 0.8)
dividend_index = DataHandler().dividend_handler(start_year, end_year, base_pool, calender)
```
