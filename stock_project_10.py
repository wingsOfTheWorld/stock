"""
通过上一个模块，我已经认识到了：
1：计算长期收益应该使用后复权的值；
2：单纯地超过百分之5卖出的策略，可能会错过一波牛市；
3：我需要更好的策略判断是否买入，和是否卖出，这就是择时的重点了
所以现在第一步，我要获取静态市盈率或者动态市盈率的数据
"""


import logging
from logging import INFO
from typing import NoReturn

import akshare as ak
import pandas as pd
import baostock as bs

from pyecharts.charts import Line
from pyecharts import options as opts
from pyecharts.globals import ThemeType


class StockAnalyze:

    def __init__(self, stock_code: str):
        """
        初始化
        :param stock_code: 股票代码，"002050"（三花智控）
        """
        self.stock_code = stock_code

    def get_price_earnings_ratio(self):
        """
        获取市盈率的数据
        :return:
        """
        lg = bs.login()
        rs = bs.query_history_k_data_plus("sz.002050", "date,peTTM", start_date='1900-01-01', adjustflag='1')
        pe_df = pd.DataFrame(rs.data, columns=['日期', '市盈率（TTM）'])
        pe_df.to_csv('002050_pe_hfq.csv')
        print(pe_df)


if __name__ == '__main__':
    test_stock_analyze = StockAnalyze(stock_code="002050")
    test_stock_analyze.get_price_earnings_ratio()