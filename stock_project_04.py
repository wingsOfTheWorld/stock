"""
将前三个版本的内容结合成类方法扩展
"""

import logging
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt
from logging import INFO
from sklearn.model_selection import ParameterGrid


# 设置中文字体（根据你的系统选择可用字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 适用于Mac
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 适用于Linux

# 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


class StockParameter:

    def __init__(self, short_ma_list, long_ma_list, filter_ma_list, etf_list):
        """
        初始化
        :param short_ma_list: 短期均线候选周期，[5, 10, 20]
        :param long_ma_list: 长期均线候选周期，[30, 50, 60]
        :param filter_ma_list: 趋势过滤均线候选周期，[100, 200]
        :param etf_list: ETF列表，['510300', '510500', '588000']，沪深300/中证500/科创50
        """

        self.short_ma_list = short_ma_list
        self.long_ma_list = long_ma_list
        self.filter_ma_list = filter_ma_list
        self.etf_list = etf_list

    def main(self):
        """
        主流程
        :return:
        """
        # 1. 定义参数网格
        param_grid = {
            'short_ma': self.short_ma_list,
            'long_ma': self.long_ma_list,
            'filter_ma': self.filter_ma_list,
            'etf': self.etf_list
        }
        result_list = []

        for params in ParameterGrid(param_grid):
            logging.log(INFO, f'\n正在测试参数组合：{params}')

            # 2. 获取数据
            df = ak.fund_etf_hist_em(symbol=params['etf'])
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期').sort_index()
            data = df[['收盘']].rename(columns={'收盘': 'Close'})

            # 3. 计算均线
            data['SMA_short'] = data['Close'].rolling(params['short_ma']).mean()
            data['SMA_long'] = data['Close'].rolling(params['long_ma']).mean()
            data['SMA_filter'] = data['Close'].rolling(params['filter_ma']).mean()
            data = data.dropna()

            # 4. 生成信号（带趋势过滤）
            data['Signal'] = 0
            data.loc[(data['SMA_short'] > data['SMA_long']) &
                     (data['Close'] > data['SMA_filter']), 'Signal'] = 1
            data.loc[data['SMA_short'] < data['SMA_long'], 'Signal'] = -1

            # 5. 添加手续费（按单边0.1%计算）
            trade_cost = 0.001  # 千分之一手续费

            # 6. 计算每日收益率
            data['Return'] = data['Close'].pct_change()

            # 7. 每次交易时扣除成本
            data['Strategy_Return'] = data['Signal'].shift(1) * data['Return'] - abs(
                data['Signal'].diff().shift(1)) * trade_cost

            # 8. 设置5%移动止损
            data['Max_Price'] = data['Close'].rolling(20).max()
            data['Stop_Loss'] = data['Max_Price'] * 0.95
            data.loc[data['Close'] < data['Stop_Loss'], 'Signal'] = 0

            # 9. 获取结果
            cumulative_returns = (1 + data[['Return', 'Strategy_Return']]).cumprod()  # 累计收益
            annual_return = cumulative_returns.iloc[-1] ** (252 / len(data)) - 1  # 年化收益率
            max_draw_down = (cumulative_returns.cummax() - cumulative_returns).max()  # 最大回撤
            trade_count = abs(data['Signal'].diff()).sum() / 2  # 交易次数

            result_list.append({
                'etf': params['etf'],
                'short_ma': params['short_ma'],
                'long_ma': params['long_ma'],
                'filter_ma': params['filter_ma'],
                '买入持有年化收益': f'{annual_return['Return']:.2%}',
                '策略年化收益': f'{annual_return['Strategy_Return']:.2%}',
                '最大回撤': f'{max_draw_down['Strategy_Return']:.2%}',
                '交易次数': f'{trade_count:.0f}',
            })

            plt.figure(figsize=(14, 8))

            # 主图：价格与均线
            ax1 = plt.subplot(211)
            ax1.plot(data['Close'], label='价格', color='black', alpha=0.8)
            ax1.plot(data['SMA_short'], label=f'{params['short_ma']}日均线', linestyle='--')
            ax1.plot(data['SMA_long'], label=f'{params['long_ma']}日均线', linestyle='--')
            ax1.plot(data['SMA_filter'], label=f'{params['filter_ma']}日均线', color='purple')
            ax1.set_title('价格与均线系统')
            ax1.legend()

            # 副图：累计收益
            ax2 = plt.subplot(212)
            ax2.plot(cumulative_returns['Return'], label='被动持有')
            ax2.plot(cumulative_returns['Strategy_Return'], label='改进策略')
            ax2.fill_between(cumulative_returns.index, cumulative_returns['Strategy_Return'], color='skyblue',
                             alpha=0.3)
            ax2.set_title('累计收益对比')
            ax2.legend()

            plt.tight_layout()
            # 保存图像为 PNG 文件
            plt.savefig(fr'D:\stock\tmp\04\{params['etf']}_{params['short_ma']}_{params['long_ma']}_'
                        fr'{params['filter_ma']}.png')

        return result_list


if __name__ == '__main__':
    test_short_ma_list = [5, 10, 20]
    test_long_ma_list = [30, 50, 60]
    test_filter_ma_list = [100, 200]
    test_etf_list = ['510300', '510500', '588000']
    test_stock_parameter = StockParameter(short_ma_list=test_short_ma_list,
                                          long_ma_list=test_long_ma_list,
                                          filter_ma_list=test_filter_ma_list,
                                          etf_list=test_etf_list)
    test_result = test_stock_parameter.main()
    test_result_df = pd.DataFrame(test_result)
    test_result_path = r'D:\stock\tmp\04\result.xlsx'
    with pd.ExcelWriter(test_result_path, 'openpyxl', 'utf-8', mode='w') as writer:
        test_result_df.to_excel(writer, sheet_name='测试结果', index=False)
