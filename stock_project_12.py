"""
2025/07/05，手动写出量化交易策略的模板
"""
import pandas as pd
import akshare as ak
import baostock as bs


class FirstQuantitativeStrategy:

    def __init__(self, stock_code, stock_market, initial_cash, min_price_earnings_ratio):
        self.stock_code = stock_code  # 选择的股票
        self.stock_market = stock_market  # 股票所在的市场
        self.initial_cash = initial_cash  # 初始资金
        self.min_price_earnings_ratio = min_price_earnings_ratio  # 最低市盈率
        self.basic_df = None

    @staticmethod
    def regular_transfer(value: float, decimal_num: int = 2) -> str:
        """
        格式化转换百分比
        :param value:
        :param decimal_num:
        :return:
        """
        return f"{value * 100:.{decimal_num}f}%"

    def generate_basic_df(self):
        """
        生成基础数据
        :return:
        """
        capital_flow = ak.stock_individual_fund_flow(stock=self.stock_code, market=self.stock_market)
        capital_flow['日期'] = pd.to_datetime(capital_flow['日期'])
        hist_data = ak.stock_zh_a_hist(symbol=self.stock_code, start_date='20241231')
        hist_data['日期'] = pd.to_datetime(hist_data['日期'])
        hist_data = hist_data.set_index('日期').sort_index()

        # 获取市盈率数据
        lg = bs.login()
        rs = bs.query_history_k_data_plus("sz.002050", "date,peTTM", start_date='2024-12-31', adjustflag='1')
        pe_df = pd.DataFrame(rs.data, columns=['日期', '市盈率（TTM）'])
        pe_df['日期'] = pd.to_datetime(pe_df['日期'])
        pe_df = pe_df.set_index('日期').sort_index()

        merge_df = pd.merge(left=hist_data, right=capital_flow, how='outer', on='日期')
        merge_df = pd.merge(left=merge_df, right=pe_df, how='outer', on='日期')

        merge_df = merge_df[[
            '日期', '收盘', '成交量', '成交额', '涨跌幅_x', '换手率', '主力净流入-净额', '主力净流入-净占比',
            '超大单净流入-净额',
            '超大单净流入-净占比', '小单净流入-净额', '小单净流入-净占比', '市盈率（TTM）'
        ]]

        merge_df = merge_df.rename(columns={'涨跌幅_x': '涨跌幅'})

        print(f'成交额最大值：{merge_df['成交额'].max()}')
        print(f'成交额最小值：{merge_df['成交额'].min()}')

        merge_df['成交额与最大成交额占比'] = round((merge_df['成交额'] / merge_df['成交额'].max()) * 100, 2)
        self.basic_df = merge_df
        self.basic_df.to_csv(f'tmp/12/basic_data_{self.stock_code}.csv', index=False)

    def generate_quantitative_strategy(self):
        """
        生成量化策略
        :return:
        """
        self.basic_df['买卖信号'] = 0  # 1表示买入，-1表示卖出
        self.basic_df['信号推荐持有占比'] = 0.0  # 持有股票资产占资金量的占比
        for index, row in self.basic_df.iterrows():
            # 成交额波动大才操作
            if row['成交额与最大成交额占比'] > 20:
                # 跟随主力资金操作
                if row['主力净流入-净占比'] > 10:
                    if row['主力净流入-净占比'] > row['涨跌幅']:
                        self.basic_df.at[index, '买卖信号'] = 1
                        if row['主力净流入-净占比'] > 20:
                            self.basic_df.at[index, '信号推荐持有占比'] = 1.0
                        else:
                            self.basic_df.at[index, '信号推荐持有占比'] = 0.5
                elif row['主力净流入-净占比'] < -10:
                    if row['主力净流入-净占比'] < row['涨跌幅']:
                        self.basic_df.at[index, '买卖信号'] = -1
                        if row['主力净流入-净占比'] < -20:
                            self.basic_df.at[index, '信号推荐持有占比'] = 0.0
                        else:
                            self.basic_df.at[index, '信号推荐持有占比'] = 0.5
        # 估值修正信号
        for index, row in self.basic_df.iterrows():
            if float(row['市盈率（TTM）']) < self.min_price_earnings_ratio:
                self.basic_df.at[index, '买卖信号'] = 1
                self.basic_df.at[index, '信号推荐持有占比'] = 1.0

    def calculate_the_rate_of_return(self):
        """
        计算收益率
        :return:
        """
        # 持有不动收益率
        hold_on_return = (self.basic_df['收盘'].iloc[-1] - self.basic_df['收盘'].iloc[0]) / self.basic_df['收盘'].iloc[
            0]
        hold_on_return = self.regular_transfer(hold_on_return)

        # 初始化仓位和资金列
        self.basic_df['持仓'] = 0
        self.basic_df['资金余额'] = self.initial_cash

        self.basic_df['资产总额'] = self.basic_df['持仓'] * self.basic_df['收盘'] + self.basic_df['资金余额']
        self.basic_df['股票资产'] = self.basic_df['持仓'] * self.basic_df['收盘']
        self.basic_df['股票资产占比'] = self.basic_df['股票资产'] / self.basic_df['资产总额']

        # 使用逐日迭代更新
        for i in range(1, len(self.basic_df)):
            # 继承前一日状态
            prev_row = self.basic_df.iloc[i - 1]
            current_hold = prev_row['持仓']
            current_cash = prev_row['资金余额']

            # 获取当日数据
            curr_row = self.basic_df.iloc[i]
            signal = curr_row['买卖信号']
            target_ratio = curr_row['信号推荐持有占比']
            price = curr_row['收盘']

            # 计算当前资产总额（使用前日持仓+当日价格）
            current_stock_value = current_hold * price  # 股票资产
            total_assets = current_stock_value + current_cash

            # 修复2：正确识别卖出信号（原代码误用==1）
            if signal == 1:  # 买入信号
                target_stock_value = total_assets * target_ratio
                if current_stock_value < target_stock_value:
                    # 计算最大可买数量（100股整数倍）
                    max_affordable = current_cash // (price * 100) * 100
                    need_buy_value = target_stock_value - current_stock_value
                    buy_num = min(int((need_buy_value // (price * 100)) * 100), max_affordable)

                    # 更新当日仓位状态
                    current_hold += buy_num
                    current_cash -= buy_num * price
                    total_assets = current_hold * price + current_cash

                    print(f'日期：{curr_row['日期']}，'
                          f'\n当前买入数量：{buy_num}，'
                          f'\n当前持有仓位：{current_hold}，'
                          f'\n当前现金：{current_cash}，'
                          f'\n当前资产总额：{total_assets}')

            elif signal == -1:  # 卖出信号（修正信号值）
                if current_stock_value > total_assets * target_ratio:
                    need_sell_value = current_stock_value - total_assets * target_ratio
                    sell_num = (need_sell_value // (price * 100)) * 100

                    # 确保不超过当前持仓
                    sell_num = min(sell_num, current_hold)
                    current_hold -= sell_num
                    current_cash += sell_num * price
                    total_assets = current_hold * price + current_cash
                    print(f'日期：{curr_row['日期']}，'
                          f'\n当前卖出数量：{sell_num}，'
                          f'\n当前持有仓位：{current_hold}，'
                          f'\n当前现金：{current_cash}，'
                          f'\n当前资产总额：{total_assets}')

            # 保存当日最终状态
            self.basic_df.at[self.basic_df.index[i], '持仓'] = current_hold
            self.basic_df.at[self.basic_df.index[i], '资金余额'] = current_cash
            self.basic_df.at[self.basic_df.index[i], '股票资产'] = current_hold * price
            self.basic_df.at[self.basic_df.index[i], '资产总额'] = current_hold * price + current_cash
            total = (current_hold * price) / (current_hold * price + current_cash)
            self.basic_df.at[self.basic_df.index[i], '股票资产占比'] = total

        strategy_return = (self.basic_df['资产总额'].iloc[-1] - self.basic_df['资产总额'].iloc[0]) / self.basic_df[
            '资产总额'].iloc[0]
        strategy_return = self.regular_transfer(strategy_return)

        print(f'持有不动收益率：{hold_on_return}')
        print(f'量化交易策略收益率：{strategy_return}')

    def main(self):
        # 1.生成基础数据df
        self.generate_basic_df()
        # 2.生成量化策略
        self.generate_quantitative_strategy()
        # 3.与持有不动比较收益率
        self.calculate_the_rate_of_return()


if __name__ == '__main__':
    test_strategy = FirstQuantitativeStrategy(
        stock_code='002050',
        stock_market='sz',
        initial_cash=100000.0,
        min_price_earnings_ratio=28.0
    )
    test_strategy.main()
