"""
重新选择择时的方式
"""

import logging
from logging import INFO
from typing import NoReturn

import akshare as ak
import pandas as pd
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

    @staticmethod
    def regular_transfer(value: float, decimal_num: int = 2) -> str:
        """
        格式化转换百分比
        :param value:
        :param decimal_num:
        :return:
        """
        return f"{value * 100:.{decimal_num}f}%"

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

    def implement_strategy(self, basic_df: pd.DataFrame) -> pd.DataFrame:
        """
        交易策略
        :param basic_df:
        :return:
        """
        basic_df['持有不动'] = basic_df['收盘'] * 1000

        # 初始化仓位和资金列
        basic_df['持仓'] = 0
        basic_df['资金余额'] = 0.0

        # 首日初始化
        first_date = basic_df.index[0]
        basic_df.at[first_date, '持仓'] = 1000
        basic_df.at[first_date, '资金余额'] = 0.0

        # 状态缓存变量
        prev_hold = 1000
        prev_cash = 0.0

        # 初始资金
        initial_fund = basic_df['开盘'].iloc[0] * 1000

        for i, (index, row) in enumerate(basic_df.iterrows()):
            if i > 0:
                # 传递前日状态
                basic_df.at[index, '持仓'] = prev_hold
                basic_df.at[index, '资金余额'] = prev_cash

                current_hold = basic_df.at[index, '持仓']
                current_cash = basic_df.at[index, '资金余额']

                if row['涨跌幅'] >= 5:  # 涨幅≥5%卖出
                    sell_amount = current_hold * row['收盘']
                    basic_df.at[index, '资金余额'] = current_cash + sell_amount
                    basic_df.at[index, '持仓'] = 0

                elif row['涨跌幅'] <= -5:  # 跌幅≤-5%买入
                    # 整手交易计算 (100股倍数)
                    buy_volume = (current_cash // (row['收盘'] * 100)) * 100
                    if buy_volume > 0:  # 仅当可买入时操作
                        buy_amount = buy_volume * row['收盘']
                        basic_df.at[index, '资金余额'] = current_cash - buy_amount
                        basic_df.at[index, '持仓'] = current_hold + buy_volume

                # 更新状态缓存
                prev_hold = basic_df.at[index, '持仓']
                prev_cash = basic_df.at[index, '资金余额']

                print(f"日期: {index} | 持仓: {prev_hold} | 资金: {prev_cash:.2f}")

        # 计算总资产
        basic_df['总资产'] = basic_df['持仓'] * basic_df['收盘'] + basic_df['资金余额']

        hold_yield_rate = self.regular_transfer((basic_df['持有不动'].iloc[-1] - initial_fund) / initial_fund)
        strategy_yield_rate = self.regular_transfer((basic_df['总资产'].iloc[-1] - initial_fund) / initial_fund)

        print(f'持有不动策略累计收益率：{hold_yield_rate}')
        print(f'交易策略累计收益率：{strategy_yield_rate}')

        return basic_df

    def visualize_assets(self, basic_df: pd.DataFrame) -> NoReturn:
        """
        可视化资产走势
        :param basic_df: 包含日期、持有不动、总资产的DataFrame
        """
        # 准备数据
        dates = basic_df.index.strftime("%Y-%m-%d").tolist()
        static_assets = basic_df['持有不动'].round(2).tolist()
        dynamic_assets = basic_df['总资产'].round(2).tolist()

        # 创建折线图对象
        line = Line(
            init_opts=opts.InitOpts(
                theme=ThemeType.LIGHT,
                width="1200px",
                height="600px"
            )
        )

        # 添加x轴数据（日期）
        line.add_xaxis(xaxis_data=dates)

        # 添加两条折线
        line.add_yaxis(
            series_name="持有不动策略",
            y_axis=static_assets,
            is_smooth=True,
            color="#5793f3",
            linestyle_opts=opts.LineStyleOpts(width=3)
        )
        line.add_yaxis(
            series_name="动态择时策略",
            y_axis=dynamic_assets,
            is_smooth=True,
            color="#d14a61",
            linestyle_opts=opts.LineStyleOpts(width=3, type_="dashed")
        )

        # 全局配置
        line.set_global_opts(
            title_opts=opts.TitleOpts(
                title="资产收益对比分析",
                subtitle=f"股票代码：{self.stock_code}",
                pos_left="20%"
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross"
            ),
            legend_opts=opts.LegendOpts(pos_top="8%"),
            xaxis_opts=opts.AxisOpts(
                name="日期",
                axislabel_opts=opts.LabelOpts(rotate=45),
                splitline_opts=opts.SplitLineOpts(is_show=False)
            ),
            yaxis_opts=opts.AxisOpts(
                name="资产金额（元）",
                axislabel_opts=opts.LabelOpts(formatter="{value} 元"),
                splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=True,
                    type_="slider",
                    range_start=0,
                    range_end=100
                ),
                opts.DataZoomOpts(type_="inside")
            ]
        )

        # 生成HTML文件
        line.render("tmp/09/asset_comparison.html")
        print("可视化文件已生成：asset_comparison.html")

    def main(self):
        basic_df = self.get_stock_data()
        result_df = self.implement_strategy(basic_df)
        self.visualize_assets(result_df)


if __name__ == '__main__':
    test_stock_analyze = StockAnalyze(stock_code="002050")
    test_stock_analyze.main()
