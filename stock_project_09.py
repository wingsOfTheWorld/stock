"""
重新选择择时的方式
"""

import akshare as ak
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import logging
from logging import INFO
from typing import NoReturn

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class StockAnalyze:

    def __init__(self, stock_code: str):
        """
        初始化
        :param stock_code: 股票代码，"002050"（三花智控）
        """
        self.stock_code = stock_code

    def get_stock_data(self) -> pd.DataFrame:
        """
        获取基础行情数据
        :return:
        """
        basic_df = ak.stock_zh_a_hist(symbol=self.stock_code, adjust="hfq")
        basic_df['日期'] = pd.to_datetime(basic_df['日期'])
        basic_df = basic_df.set_index('日期').sort_index()
        logging.log(
            level=INFO,
            msg=f"{basic_df[['收盘', '成交量', '换手率']].tail()}"
        )
        return basic_df

    @staticmethod
    def cal_moving_average_convergence_divergence(
            df: pd.DataFrame,
            fast_period: int,
            slow_period: int,
            signal_period: int
    ) -> pd.DataFrame:
        """
        MACD指标（动态参数）
        :param df: 基础数据
        :param fast_period: EMA12
        :param slow_period: EMA26
        :param signal_period: DIF的9日EMA
        :return:
        """
        df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['收盘'],
                                                      fastperiod=fast_period,
                                                      slowperiod=slow_period,
                                                      signalperiod=signal_period)
        return df

    @staticmethod
    def avg_price(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        平均成交价（动态计算）
        :param df:
        :return:
        """
        df['平均成交价'] = df['成交额'] / (df['成交量'] * 100)  # 单位：元/股
        df['AvgPrice_MA20'] = df['平均成交价'].rolling(20).mean()
        return df

    @staticmethod
    def trading_volume(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        成交量系统（结合波动率）
        :param df:
        :return:
        """
        df['Vol_MA5'] = df['成交量'].rolling(5).mean()
        df['Vol_Ratio'] = df['成交量'] / df['Vol_MA5']
        return df

    @staticmethod
    def turnover_rate(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        换手率策略（分级阈值）
        :param df:
        :return:
        """
        df['换手率_MA5'] = df['换手率'].rolling(5).mean()
        df['换手率等级'] = pd.cut(
            df['换手率'],
            bins=[0, 3, 7, 15, 100],
            labels=['低迷', '温和', '活跃', '异常']
        )
        return df

    @staticmethod
    def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
        """
        简化版信号生成（涨跌阈值触发）
        规则：
        1. 初始持仓1000手
        2. 当日涨幅>5%时全部卖出（转为空仓）
        3. 当日跌幅>5%时用全部可用资金买入（空仓转满仓）
        """
        df['Signal'] = 0  # 初始化信号列

        # 计算日收益率（使用收盘价）
        df['Return'] = df['收盘'].pct_change()

        # 生成交易信号（考虑持仓状态）
        position = 1  # 初始持仓状态：满仓（1000手）
        for i in range(1, len(df)):
            prev_return = df['Return'].iloc[i]

            # 当前持仓状态下生成信号
            if position == 1 and prev_return > 0.05:  # 满仓且涨幅达标
                df['Signal'].iloc[i] = -1  # 卖出信号
                position = 0
            elif position == 0 and prev_return < -0.05:  # 空仓且跌幅达标
                df['Signal'].iloc[i] = 1  # 买入信号
                position = 1

        return df

    @staticmethod
    def calculate_metrics(returns: pd.Series) -> pd.Series:
        """
        关键绩效指标计算
        :param returns:
        :return:
        """
        total_return = returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        max_drawdown = (returns / returns.cummax() - 1).min()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        return pd.Series([total_return, annual_return, max_drawdown, sharpe_ratio],
                         index=['总收益', '年化收益', '最大回撤', '夏普比率'])

    def generate_strategy_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        策略回测（动态仓位管理）
        """
        # 初始化资金参数
        initial_cash = 1000 * df['收盘'].iloc[0]  # 1000手初始资金
        cash = initial_cash
        shares = 1000  # 初始持仓

        # 记录每日资产
        df['Strategy_Value'] = np.nan
        df['Strategy_Value'].iloc[0] = initial_cash  # 初始资产

        # 交易成本参数
        trade_cost_rate = 0.0025  # 双边0.25%

        for i in range(1, len(df)):
            price = df['收盘'].iloc[i]
            prev_price = df['收盘'].iloc[i - 1]

            # 执行交易信号
            if df['Signal'].iloc[i] == -1:  # 卖出
                cash += shares * price * (1 - trade_cost_rate)
                shares = 0
            elif df['Signal'].iloc[i] == 1:  # 买入
                if cash > 0:
                    shares = cash / (price * (1 + trade_cost_rate))
                    cash = 0

            # 计算当日资产
            df['Strategy_Value'].iloc[i] = cash + shares * price

        # 计算收益率
        df['Strategy_Return'] = df['Strategy_Value'].pct_change()
        df['BuyHold_Return'] = df['收盘'] / df['收盘'].iloc[0] - 1  # 持有不动收益

        # 可视化对比
        plt.figure(figsize=(14, 7))
        plt.plot(df['Strategy_Value'] / initial_cash, label='阈值策略')
        plt.plot(df['收盘'] / df['收盘'].iloc[0], label='持有不动', linestyle='--')
        plt.title('策略对比（2023-2025）')
        plt.legend()
        plt.savefig(fr'D:\stock\tmp\09\{self.stock_code}_简化策略对比.png')

        return df

    def visualization(
            self,
            df: pd.DataFrame,
            fast_period: int,
            slow_period: int,
            signal_period: int
    ) -> NoReturn:
        """
        Step 5：可视化（突出四指标关系）
        :param df:
        :param fast_period: EMA12
        :param slow_period: EMA26
        :param signal_period: DIF的9日EMA
        :return:
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 1)

        # 价格与信号
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df['收盘'], label='Price', alpha=0.8)
        ax1.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1]['收盘'],
                    marker='^', color='g', s=100, label='买入信号')
        ax1.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1]['收盘'],
                    marker='v', color='r', s=100, label='止损信号')
        ax1.set_title(f'{self.stock_code}四因子交易信号图', fontsize=14)

        # MACD指标
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df['MACD'], label=f'MACD({fast_period},{slow_period})', linewidth=1)
        ax2.plot(df['MACD_Signal'], label=f'Signal({signal_period})', linewidth=1)
        ax2.bar(df.index, df['MACD'] - df['MACD_Signal'],
                color=np.where((df['MACD'] - df['MACD_Signal']) > 0, 'g', 'r'), alpha=0.3)
        ax2.axhline(0, linestyle='--', color='gray')
        ax2.set_title('MACD指标动态', fontsize=12)

        # 量价关系
        ax3 = fig.add_subplot(gs[2])
        ax3.bar(df.index, df['成交量'] / 1e6, color=np.where(df['Vol_Ratio'] > 1.5, 'orange', 'gray'),
                alpha=0.6, label='成交量(百万股)')
        ax3.plot(df['Vol_MA5'] / 1e6, label='5日成交量均线', color='purple', linewidth=1.5)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df['平均成交价'], label='平均成交价', color='b', linewidth=1.5)
        ax3_twin.plot(df['AvgPrice_MA20'], label='20日均价线', linestyle='--', color='darkblue')
        ax3.set_title('量价关系分析', fontsize=12)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')

        # 换手率分析
        ax4 = fig.add_subplot(gs[3])
        colors = {'低迷': 'gray', '温和': 'skyblue', '活跃': 'orange', '异常': 'red'}
        for level in colors.keys():
            idx = df['换手率等级'] == level
            ax4.scatter(df.index[idx], df['换手率'][idx],
                        color=colors[level], label=level, alpha=0.6)
        ax4.plot(df['换手率_MA5'], label='5日平均换手率', color='purple', linewidth=1.5)
        ax4.set_title('换手率分级监控（参考网页9规则）', fontsize=12)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(fr'D:\stock\tmp\08\{self.stock_code}四因子交易信号图.png')

    def main(self):
        """
        主程序
        :return:
        """
        # 1. 获取基础行情数据
        basic_df = self.get_stock_data()
        # 2. MACD指标
        df = self.cal_moving_average_convergence_divergence(basic_df, 7, 22, 9)
        # 3. 平均成交价
        df = self.avg_price(df)
        # 4. 成交量系统
        df = self.trading_volume(df)
        # 5. 换手率策略（分级阈值）
        df = self.turnover_rate(df)
        # 6. 信号生成（多条件复合验证）
        df = self.generate_signal(df)
        # 7. 策略回测（增加换手率分析）
        self.generate_strategy_result(df)


if __name__ == '__main__':
    test_stock_analyze = StockAnalyze(stock_code="002050")
    test_stock_analyze.main()
