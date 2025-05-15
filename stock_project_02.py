"""
问题诊断（为什么策略跑输大盘）
1.参数周期问题：
    20/60日均线组合在特定时间段（2020-2023）可能产生滞后
    观察期间可能存在震荡市（均线策略在趋势市表现更好）
2.交易机制缺陷：
    未考虑交易成本（默认零手续费）
    未处理信号频繁切换带来的摩擦损耗
3.数据时间段特殊性：
    2020年疫情后的V型反转 + 2021-2023震荡市组合可能对趋势跟踪策略不利
"""


import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体（根据你的系统选择可用字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 适用于Mac
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 适用于Linux

# 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 获取数据
etfs = ['510300', '510500', '588000']  # 沪深300/中证500/科创50
for symbol in etfs:

    df = ak.fund_etf_hist_em(symbol=symbol)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期').sort_index()
    data = df[['收盘']].rename(columns={'收盘': 'Close'})
    print(data.head(5))

    # 测试更灵敏的均线组合（如10/50日）
    data['SMA10'] = data['Close'].rolling(10).mean()
    data['SMA50'] = data['Close'].rolling(50).mean()
    data = data.dropna()

    # 更新信号逻辑
    data['Signal'] = 0
    data.loc[data['SMA10'] > data['SMA50'], 'Signal'] = 1
    data.loc[data['SMA10'] < data['SMA50'], 'Signal'] = -1

    # 在计算策略收益前添加手续费（按单边0.1%计算）
    trade_cost = 0.001  # 千分之一手续费

    # 计算每日收益率
    data['Return'] = data['Close'].pct_change()

    # 每次交易时扣除成本
    data['Strategy_Return'] = data['Signal'].shift(1) * data['Return'] - abs(data['Signal'].diff().shift(1)) * trade_cost

    # 增加200日均线作为牛熊分界线
    data['SMA200'] = data['Close'].rolling(200).mean()

    # 仅当价格在200日均线上方时允许做多
    data.loc[data['Close'] < data['SMA200'], 'Signal'] = 0  # 过滤熊市信号

    # 设置5%移动止损
    data['Max_Price'] = data['Close'].rolling(20).max()
    data['Stop_Loss'] = data['Max_Price'] * 0.95
    data.loc[data['Close'] < data['Stop_Loss'], 'Signal'] = 0

    # 累计收益
    cumulative_returns = (1 + data[['Return', 'Strategy_Return']]).cumprod()

    # 年化收益率
    annual_return = cumulative_returns.iloc[-1] ** (252 / len(data)) - 1

    # 最大回撤
    max_draw_down = (cumulative_returns.cummax() - cumulative_returns).max()

    # 交易次数
    trade_count = abs(data['Signal'].diff()).sum() / 2  # 每笔交易含买卖

    print(f"""
    策略绩效报告：
    1. 买入持有年化收益: {annual_return['Return']:.2%}
    2. 策略年化收益: {annual_return['Strategy_Return']:.2%}
    3. 最大回撤（策略）: {max_draw_down['Strategy_Return']:.2%}
    4. 总交易次数: {trade_count:.0f}次
    """)

    plt.figure(figsize=(14, 8))

    # 主图：价格与均线
    ax1 = plt.subplot(211)
    ax1.plot(data['Close'], label='价格', color='black', alpha=0.8)
    ax1.plot(data['SMA10'], label='10日均线', linestyle='--')
    ax1.plot(data['SMA50'], label='50日均线', linestyle='--')
    ax1.plot(data['SMA200'], label='200日均线', color='purple')
    ax1.set_title('价格与均线系统')
    ax1.legend()

    # 副图：累计收益
    ax2 = plt.subplot(212)
    ax2.plot(cumulative_returns['Return'], label='被动持有')
    ax2.plot(cumulative_returns['Strategy_Return'], label='改进策略')
    ax2.fill_between(cumulative_returns.index, cumulative_returns['Strategy_Return'], color='skyblue', alpha=0.3)
    ax2.set_title('累计收益对比')
    ax2.legend()

    plt.tight_layout()
    # 保存图像为 PNG 文件
    plt.savefig(fr'D:\stock\tmp\{symbol}.png')

