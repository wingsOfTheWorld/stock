"""
将方法封装成类
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
    def generate_signal(
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        信号生成（多条件复合验证）
        :param df:
        :return:
        """
        # 初始化信号列
        df['Signal'] = 0

        # 条件1：MACD金叉（动量验证）
        condition_macd = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))

        # 条件2：量价配合（网页9规则）
        condition_volume = (df['Vol_Ratio'] > 1.5) & (df['平均成交价'] > df['AvgPrice_MA20'])

        # 条件3：换手率策略（网页5/9规则）
        condition_turnover = df['换手率等级'].isin(['活跃', '异常']) & (df['换手率'] > df['换手率_MA5'])

        # 综合买入信号（需同时满足）
        buy_condition = condition_macd & condition_volume & condition_turnover
        df.loc[buy_condition, 'Signal'] = 1

        # 动态止盈止损（网页7规则）
        df['Max_Price'] = df['收盘'].cummax()
        df.loc[df['收盘'] < 0.93 * df['Max_Price'], 'Signal'] = -1  # 7%回撤止损

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

    def generate_strategy_result(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        策略回测（增加换手率分析）
        :param df:
        :return:
        """
        # 计算收益率
        df['Return'] = df['收盘'].pct_change()
        df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']

        # 交易成本（含冲击成本）
        trade_cost = 0.0025  # 双边0.25%
        df['Strategy_Return'] = df['Strategy_Return'] - abs(df['Signal'].diff().shift(1)) * trade_cost

        df.to_csv(rf'D:\stock\tmp\08\{self.stock_code}_基础数据.csv')

        # 计算累计收益率
        df['Strategy_CumReturn'] = (1 + df['Strategy_Return']).cumprod()
        df['BuyHold_CumReturn'] = (1 + df['Return']).cumprod()

        # 生成交易记录明细
        trades = df[df['Signal'].diff() != 0].copy()
        trades['交易类型'] = np.where(trades['Signal'] == 1, '买入', '卖出')
        trades['持仓天数'] = trades.index.to_series().diff().dt.days.shift(-1)

        strategy_metrics = self.calculate_metrics(df['Strategy_Return'].dropna())
        buyhold_metrics = self.calculate_metrics(df['Return'].dropna())

        # 结果保存为CSV
        result_df = pd.DataFrame({
            '日期': df.index,
            '收盘价': df['收盘'],
            '信号': df['Signal'],
            '策略收益率': df['Strategy_Return'],
            '策略累计收益': df['Strategy_CumReturn'],
            '持有收益率': df['Return'],
            '持有累计收益': df['BuyHold_CumReturn'],
            'MACD': df['MACD'],
            '换手率': df['换手率']
        })
        result_df.to_csv(r'D:\stock\tmp\08\strategy_comparison.csv', index=True)

        # 输出关键指标对比表
        metrics_df = pd.DataFrame([strategy_metrics, buyhold_metrics],
                                  index=['多因子策略', '持有不动'])
        metrics_df.to_csv(rf'D:\stock\tmp\08\{self.stock_code}_多因子策略与持有不动对比.csv', index=True)

        # 策略对比可视化
        plt.figure(figsize=(14, 7))
        plt.plot(df['Strategy_CumReturn'], label='多因子策略', color='#2ca02c')
        plt.plot(df['BuyHold_CumReturn'], label='持有不动', color='#1f77b4', linestyle='--')

        # 标注关键交易点
        buy_dates = df[df['Signal'] == 1].index
        plt.scatter(buy_dates, df.loc[buy_dates, 'Strategy_CumReturn'],
                    marker='^', color='r', s=100, label='买入信号')

        plt.title('策略收益对比（2023-2025）', fontsize=14)
        plt.xlabel('日期')
        plt.ylabel('累计收益率')
        plt.legend()
        plt.grid(True)
        plt.savefig(fr'D:\stock\tmp\08\{self.stock_code}_策略比较结果.png')

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
