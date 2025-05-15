"""
我们从最基础的移动平均线策略开始，分步骤实现数据获取、策略编写和可视化。
"""

# 方法一：使用akshare（推荐国内用户）
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

# 获取数据
df = ak.fund_etf_hist_em(symbol="510300")
df['日期'] = pd.to_datetime(df['日期'])
df = df.set_index('日期').sort_index()
data = df[['收盘']].rename(columns={'收盘': 'Close'})
print(data.head(5))

# 计算短期（20日）和长期（60日）均线
data['SMA20'] = data['Close'].rolling(20).mean()
data['SMA60'] = data['Close'].rolling(60).mean()
# 删除空值（前59天无法计算60日均线）
data = data.dropna()

# 生成信号
# 初始化信号列
data['Signal'] = 0
# 金叉（短期均线上穿长期）：买入信号=1
data.loc[data['SMA20'] > data['SMA60'], 'Signal'] = 1
# 死叉（短期均线下穿长期）：卖出信号=-1
data.loc[data['SMA20'] < data['SMA60'], 'Signal'] = -1
# 为了简化，我们假设每次信号变化时立即交易
data['Position'] = data['Signal'].diff()

# 回测
# 计算每日收益率
data['Return'] = data['Close'].pct_change()
# 计算策略收益率（假设每次满仓操作）
data['Strategy_Return'] = data['Signal'].shift(1) * data['Return']

# 设置中文字体（根据你的系统选择可用字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 适用于Mac
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 适用于Linux

# 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 修改后的可视化代码
plt.figure(figsize=(12, 6))
plt.plot((1 + data[['Return', 'Strategy_Return']]).cumprod())
plt.title("沪深300ETF双均线策略回测")
plt.legend(['持有不动', '均线策略'])
plt.xlabel("日期")
plt.ylabel("累计收益率")
plt.grid(True)
plt.show()
