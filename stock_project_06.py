"""
针对三花智控（002050）的多指标量化择时策略开发方案
"""

import akshare as ak
import pandas as pd
import talib  # 技术指标计算库
import matplotlib.pyplot as plt

"""
Step 1：数据准备
"""

# 获取三花智控历史数据（复权）
stock_code = "002050"
stock_zh = ak.stock_zh_a_hist(symbol=stock_code, adjust="hfq")  # 后复权数据
stock_zh['日期'] = pd.to_datetime(stock_zh['日期'])
df = stock_zh.set_index('日期').sort_index()

# 获取实时市盈率（需要安装akshare最新版）
pe_df = ak.stock_a_lg_indicator(symbol=stock_code)
latest_pe = pe_df.iloc[-1]['市盈率-动态']

"""
Step 2：多维度指标计算
"""

# 技术面指标
# 1. 均线系统
df['MA5'] = df['收盘'].rolling(5).mean()
df['MA20'] = df['收盘'].rolling(20).mean()

# 2. MACD
df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['收盘'],
                                              fastperiod=12,
                                              slowperiod=26,
                                              signalperiod=9)

# 3. RSI
df['RSI14'] = talib.RSI(df['收盘'], timeperiod=14)

# 4. 量价关系
df['成交额/成交量'] = df['成交额'] / (df['成交量'] * 100)  # 计算平均成交价格
df['Vol_MA5'] = df['成交量'].rolling(5).mean()  # 成交量5日均线

# 基本面指标
# 动态市盈率分位点（需历史数据）
# 假设已有历史市盈率数据pe_history
# 计算当前PE所处百分位
current_pe_percentile = (pe_history < latest_pe).mean()

"""
Step 3：信号生成逻辑
"""
# 初始化信号列
df['Signal'] = 0

# 条件1：均线交叉（趋势）
condition_ma = (df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1))

# 条件2：MACD金叉（动量）
condition_macd = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))

# 条件3：RSI超卖修复（反转）
condition_rsi = (df['RSI14'] > 30) & (df['RSI14'].shift(1) <= 30)

# 条件4：量价配合（主力资金）
condition_volume = (df['成交额'] > 1e8) & (df['成交额/成交量'] > df['成交额/成交量'].rolling(20).mean())

# 条件5：估值合理（基本面）
condition_pe = current_pe_percentile < 0.7  # PE处于历史30%分位以下

# 综合信号（需全部满足）
buy_condition = condition_ma & condition_macd & condition_rsi & condition_volume & condition_pe
df.loc[buy_condition, 'Signal'] = 1

# 止损信号
df['Max_Price'] = df['收盘'].cummax()
df.loc[df['收盘'] < 0.95 * df['Max_Price'], 'Signal'] = 0  # 回撤5%止损

"""
Step 4：策略回测
"""
# 计算收益率
df['Return'] = df['收盘'].pct_change()
df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']

# 交易成本（双边0.2%）
trade_cost = 0.002
df['Strategy_Return'] = df['Strategy_Return'] - abs(df['Signal'].diff().shift(1)) * trade_cost

# 可视化
plt.figure(figsize=(14, 10))

# 价格与信号
ax1 = plt.subplot(311)
ax1.plot(df['收盘'], label='Price')
ax1.plot(df['MA5'], label='5日均线', linestyle='--')
ax1.plot(df['MA20'], label='20日均线', linestyle='--')
ax1.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1]['收盘'], marker='^', color='g', label='买入信号')
ax1.set_title('价格与交易信号')

# MACD与RSI
ax2 = plt.subplot(312)
ax2.plot(df['MACD'], label='MACD')
ax2.plot(df['MACD_Signal'], label='Signal Line')
ax2.bar(df.index, df['RSI14'] - 50, label='RSI-50', alpha=0.3)
ax2.axhline(70, linestyle='--', color='r')
ax2.axhline(30, linestyle='--', color='g')
ax2.set_title('MACD与RSI')

# 累计收益
ax3 = plt.subplot(313)
cum_ret = (1 + df[['Return', 'Strategy_Return']]).cumprod()
ax3.plot(cum_ret['Return'], label='持有不动')
ax3.plot(cum_ret['Strategy_Return'], label='策略收益')
ax3.set_title('累计收益对比')
plt.legend()
plt.tight_layout()
plt.show()

"""
策略优化方向
"""

# 动态权重调整
# 给不同条件赋予权重（需用机器学习优化）
conditions = {
    'ma': 0.3,
    'macd': 0.2,
    'rsi': 0.15,
    'volume': 0.25,
    'pe': 0.1
}

df['Score'] = (condition_ma * conditions['ma'] +
               condition_macd * conditions['macd'] +
               condition_rsi * conditions['rsi'] +
               condition_volume * conditions['volume'] +
               condition_pe * conditions['pe'])

# 设置阈值触发交易
df.loc[df['Score'] > 0.7, 'Signal'] = 1

# 行业对比增强
# 获取行业平均PE
industry_pe = ak.stock_board_industry_pe_ths(symbol="通用设备")  # 三花所属行业

# 计算相对估值
df['Industry_PE'] = industry_pe['市盈率']
df['PE_Ratio'] = latest_pe / df['Industry_PE']
condition_industry_pe = df['PE_Ratio'] < 1  # 低于行业平均

"""
注意事项
数据频率统一：确保所有指标使用相同时间粒度（日线/周线）

过拟合风险：避免在少量参数上过度优化，建议使用滚动窗口测试

市场环境适应：

趋势市：侧重MACD+均线

震荡市：侧重RSI+成交量

实时数据更新：需对接实时行情接口（如Tushare Pro）
"""
