"""
ParameterGrid 是 scikit-learn 库中用于生成参数网格的工具，它可以帮助你系统性地遍历所有可能的参数组合。
在量化策略开发中，它常用于参数优化——通过穷举测试不同参数组合的表现，寻找最优参数。
"""

from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import pandas as pd
import akshare as ak

# 参数网格定义
param_grid = {
    'short_ma': [5, 10, 20],  # 短期均线候选周期
    'long_ma': [30, 50, 60],  # 长期均线候选周期
    'filter_ma': [100, 200]  # 趋势过滤均线候选周期
}

# 存储所有参数组合的结果
results = []

# 遍历所有参数组合
for params in ParameterGrid(param_grid):
    print(f"\n正在测试参数组合：{params}")

    # 获取数据（以沪深300为例）
    df = ak.fund_etf_hist_em(symbol="510300")
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期').sort_index()
    data = df[['收盘']].rename(columns={'收盘': 'Close'})

    # 计算均线
    data['SMA_short'] = data['Close'].rolling(params['short_ma']).mean()
    data['SMA_long'] = data['Close'].rolling(params['long_ma']).mean()
    data['SMA_filter'] = data['Close'].rolling(params['filter_ma']).mean()
    data = data.dropna()

    # 生成信号（带趋势过滤）
    data['Signal'] = 0
    data.loc[(data['SMA_short'] > data['SMA_long']) &
             (data['Close'] > data['SMA_filter']), 'Signal'] = 1
    data.loc[data['SMA_short'] < data['SMA_long'], 'Signal'] = -1

    # 计算收益（含0.1%手续费）
    data['Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Signal'].shift(1) * data['Return'] - abs(data['Signal'].diff().shift(1)) * 0.001

    # 计算关键指标
    cumulative_returns = (1 + data['Strategy_Return']).cumprod()
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

    # 存储结果
    results.append({
        'params': params,
        'cumulative_return': cumulative_returns.iloc[-1],
        'max_drawdown': max_drawdown,
        'trade_count': abs(data['Signal'].diff()).sum() / 2
    })

# 转换为DataFrame分析结果
results_df = pd.DataFrame(results)
print("\n参数优化结果排序：")
print(results_df.sort_values('cumulative_return', ascending=False).head())
results_df.to_csv(r'D:\python_project\tmp\参数优化组合.csv')
