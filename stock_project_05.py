"""
params	                                           cumulative_return	max_drawdown	trade_count
{'filter_ma': 100, 'long_ma': 30, 'short_ma': 5}	1.1704126059297093	1.1595364627524138	135.5
{'filter_ma': 100, 'long_ma': 30, 'short_ma': 10}	1.169066766165795	1.409013109577166	115.5
{'filter_ma': 100, 'long_ma': 30, 'short_ma': 20}	0.6106994051478947	1.510156157869409	120
{'filter_ma': 100, 'long_ma': 50, 'short_ma': 5}	1.2317421346564814	1.0614675839085952	105.5
{'filter_ma': 100, 'long_ma': 50, 'short_ma': 10}	0.9336278540794182	1.0285227820530705	85.5
{'filter_ma': 100, 'long_ma': 50, 'short_ma': 20}	0.44499358288795504	1.1298746496028975	85
{'filter_ma': 100, 'long_ma': 60, 'short_ma': 5}	0.9932660471469977	1.0894488340493154	102.5
{'filter_ma': 100, 'long_ma': 60, 'short_ma': 10}	0.8685312613707015	1.419925800917696	81.5
{'filter_ma': 100, 'long_ma': 60, 'short_ma': 20}	0.37807901598050614	1.6767977863755168	91
{'filter_ma': 200, 'long_ma': 30, 'short_ma': 5}	1.1550972057619042	0.9865655572174393	116
{'filter_ma': 200, 'long_ma': 30, 'short_ma': 10}	1.0357714644484923	1.032457658146761	100
{'filter_ma': 200, 'long_ma': 30, 'short_ma': 20}	0.5562898005780551	1.3230869986807674	107
{'filter_ma': 200, 'long_ma': 50, 'short_ma': 5}	1.3257600553216613	0.8469194346076012	89
{'filter_ma': 200, 'long_ma': 50, 'short_ma': 10}	0.9130515487221953	0.9265640558210958	71
{'filter_ma': 200, 'long_ma': 50, 'short_ma': 20}	0.5604882334354422	1.040330005001578	72
{'filter_ma': 200, 'long_ma': 60, 'short_ma': 5}	1.0825900845578036	0.9321257207761038	86
{'filter_ma': 200, 'long_ma': 60, 'short_ma': 10}	0.9755265155366778	1.3094251850585765	68
{'filter_ma': 200, 'long_ma': 60, 'short_ma': 20}	0.46008154504449983	1.4141203132465041	74

从回测结果来看，有几个关键问题需要关注和改进：
1. 最大回撤异常（>100%）
2. 年化收益率与基准不匹配

"""

import akshare as ak
import pandas as pd


class DualMAStrategy:
    def __init__(self, params):
        self.params = params
        self.data = None

    def load_data(self, symbol):
        """独立数据加载方法"""
        df = ak.fund_etf_hist_em(symbol=symbol)
        df['日期'] = pd.to_datetime(df['日期'])
        self.data = df.set_index('日期')['收盘'].rename('Close').to_frame()

    def calculate_indicators(self):
        """指标计算隔离"""
        self.data['SMA_short'] = self.data['Close'].rolling(self.params['short_ma']).mean()
        self.data['SMA_long'] = self.data['Close'].rolling(self.params['long_ma']).mean()
        self.data['SMA_filter'] = self.data['Close'].rolling(self.params['filter_ma']).mean()
        self.data.dropna(inplace=True)

    def generate_signals(self):
        """信号生成独立模块"""
        # 趋势过滤
        self.data['Signal'] = 0
        self.data.loc[self.data['Close'] > self.data['SMA_filter'], 'Signal'] = 1
        # 均线交叉
        self.data.loc[self.data['SMA_short'] > self.data['SMA_long'], 'Signal'] = 1
        self.data.loc[self.data['SMA_short'] < self.data['SMA_long'], 'Signal'] = -1
        # 止损
        self.data['Max_Price'] = self.data['Close'].cummax()
        self.data['Stop_Loss'] = self.data['Max_Price'] * 0.95
        self.data.loc[self.data['Close'] < self.data['Stop_Loss'], 'Signal'] = 0
        