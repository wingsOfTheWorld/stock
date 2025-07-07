"""
一个完整的量化投资系统框架，融合行业选择、公司筛选、波段操作和长期持有策略。
"""

import pandas as pd
import akshare as ak
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 设置中文字体（根据你的系统选择可用字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 适用于Mac
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 适用于Linux

# 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


class QuantumInvestmentSystem:
    def __init__(self, capital=1000000, risk_level=0.3):
        """
        初始化投资系统
        :param capital: 初始资金 (默认100万)
        :param risk_level: 风险承受水平 (0-1)
        """
        self.capital = capital
        self.risk_level = risk_level
        self.portfolio = {}
        self.transaction_log = []

        # 核心关注行业
        self.target_industries = [
            "机器人", "人工智能", "量子通信", "量子计算", "可控核聚变"
        ]

        # 行业龙头映射
        self.industry_leaders = {
            "机器人": ["002050", "300124", "603486"],
            "人工智能": ["300308", "002230", "603019"],
            "量子通信": ["002224", "600487", "000988"],
            "量子计算": ["000977", "603927", "002281"],
            "可控核聚变": ["600416", "002318", "300035"]
        }

    def get_industry_data(self):
        """获取行业数据"""
        industry_df = ak.stock_board_industry_name_em()
        return industry_df[industry_df['板块名称'].isin(self.target_industries)]

    def screen_companies(self):
        """筛选目标公司"""
        target_stocks = []

        # 获取行业数据
        industry_df = self.get_industry_data()

        for industry in self.target_industries:
            # 获取行业成分股
            try:
                stocks = ak.stock_board_industry_cons_em(symbol=industry)['代码'].tolist()
            except:
                stocks = self.industry_leaders.get(industry, [])

            # 基本面分析
            for stock in stocks:
                try:
                    # 获取公司基本信息
                    profile = ak.stock_individual_info_em(symbol=stock)

                    # 获取财务数据
                    finance = ak.stock_financial_report_sina(stock, "balance")

                    # 筛选条件：市值>100亿，ROE>10%，负债率<60%
                    if (profile.get('总市值', 0) > 100 and
                            finance['ROE'].iloc[-1] > 10 and
                            finance['资产负债率'].iloc[-1] < 60):
                        target_stocks.append(stock)
                except:
                    continue
        print(f'target_stocks: {list(set(target_stocks))}')
        return list(set(target_stocks))

    def analyze_capital_flow(self, stock_code):
        """分析主力资金流向"""
        # 获取资金流数据
        capital_flow = ak.stock_individual_fund_flow(stock=stock_code, market='sz')
        capital_flow['日期'] = pd.to_datetime(capital_flow['日期'])
        capital_flow = capital_flow.set_index('日期').sort_index()

        # 主力行为分析
        capital_flow['主力行为'] = 0  # 0:观望，1:吸筹，-1:出货

        # 吸筹信号：连续3天主力净流入但股价下跌
        buy_cond = (capital_flow['主力净流入-净额'] > 0) & (capital_flow['涨跌幅'] < 0)
        capital_flow.loc[buy_cond.rolling(3).sum() >= 2, '主力行为'] = 1

        # 出货信号：连续3天主力净流出但股价上涨
        sell_cond = (capital_flow['主力净流入-净额'] < 0) & (capital_flow['涨跌幅'] > 0)
        capital_flow.loc[sell_cond.rolling(3).sum() >= 2, '主力行为'] = -1

        return capital_flow

    def generate_trading_signals(self, stock_code):
        """生成交易信号"""
        # 获取历史数据
        hist_data = ak.stock_zh_a_hist(symbol=stock_code, adjust="hfq")
        hist_data['日期'] = pd.to_datetime(hist_data['日期'])
        hist_data = hist_data.set_index('日期').sort_index()

        # 获取资金流分析
        capital_flow = self.analyze_capital_flow(stock_code)

        # 合并数据
        merged_data = hist_data.merge(capital_flow[['主力行为', '主力5日平均']],
                                      left_index=True, right_index=True, how='left')

        # 技术指标计算
        merged_data['MA20'] = merged_data['收盘'].rolling(20).mean()
        merged_data['RSI'] = self.calculate_rsi(merged_data['收盘'])

        # 机器学习特征工程
        features = merged_data[['开盘', '收盘', '最高', '最低', '成交量', 'MA20', 'RSI', '主力5日平均']]
        features['价格波动率'] = features['收盘'].pct_change().rolling(5).std()
        features['量价背离'] = np.where(
            (features['收盘'] > features['收盘'].shift(1)) &
            (features['成交量'] < features['成交量'].shift(1)), 1, 0
        )

        # 目标变量：未来5日收益率
        target = (features['收盘'].shift(-5) / features['收盘'] - 1) > 0.05

        # 训练随机森林模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # 将特征和目标合并到同一DataFrame
        combined = pd.concat([features, target.rename('target')], axis=1)

        # 统一删除NaN
        combined_clean = combined.dropna()

        # 重新分割数据
        X = combined_clean.drop('target', axis=1)
        y = combined_clean['target']

        # 训练模型
        model.fit(X, y)
        # model.fit(features.dropna(), target.dropna())

        # 生成预测信号
        merged_data['预测信号'] = model.predict(features)

        # 综合交易信号
        merged_data['交易信号'] = 0

        # 买入信号：主力吸筹且模型看好
        merged_data.loc[
            (merged_data['主力行为'] == 1) &
            (merged_data['预测信号'] == 1) &
            (merged_data['RSI'] < 40), '交易信号'] = 1

        # 卖出信号：主力出货或超涨
        merged_data.loc[
            (merged_data['主力行为'] == -1) |
            (merged_data['RSI'] > 70), '交易信号'] = -1

        return merged_data

    def calculate_rsi(self, prices, window=14):
        """计算相对强弱指数(RSI)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def position_management(self, stock_data, stock_code):
        """仓位管理"""
        signals = stock_data['交易信号'].copy()

        # 初始状态
        position = 0
        avg_cost = 0
        position_history = []

        for date, signal in signals.items():
            price = stock_data.loc[date, '收盘']

            # 买入信号
            if signal == 1 and position == 0:
                # 根据风险水平确定仓位
                position_size = min(self.capital * self.risk_level, self.capital * 0.1)
                shares = int(position_size // price)

                if shares > 0:
                    position = shares
                    avg_cost = price
                    self.transaction_log.append({
                        'date': date,
                        'stock': stock_code,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price
                    })

            # 卖出信号
            elif signal == -1 and position > 0:
                self.transaction_log.append({
                    'date': date,
                    'stock': stock_code,
                    'action': 'SELL',
                    'shares': position,
                    'price': price
                })
                position = 0
                avg_cost = 0

            # 记录每日持仓
            position_history.append({
                'date': date,
                'position': position,
                'value': position * price,
                'avg_cost': avg_cost
            })

        return pd.DataFrame(position_history).set_index('date')

    def portfolio_optimization(self):
        """投资组合优化"""
        # 筛选目标股票
        # target_stocks = self.screen_companies()

        # 行业分析以后再搞，优先分析股票
        target_stocks = ["002050"]

        # 为每只股票生成信号和仓位
        for stock in target_stocks:

            stock_data = self.generate_trading_signals(stock)
            position_df = self.position_management(stock_data, stock)

            # 存储到投资组合
            self.portfolio[stock] = {
                'data': stock_data,
                'position': position_df
            }

        return self.portfolio

    def backtest_portfolio(self):
        """投资组合回测"""
        portfolio_value = pd.DataFrame(index=pd.date_range(start='2020-01-01', end=datetime.today()))
        portfolio_value['现金'] = self.capital
        portfolio_value['股票价值'] = 0
        portfolio_value['总资产'] = self.capital

        # 初始化每只股票的价值
        for stock, data in self.portfolio.items():
            position = data['position']
            stock_value = position['position'] * data['data']['收盘']
            portfolio_value = portfolio_value.merge(
                stock_value.rename(f"{stock}_价值"),
                left_index=True, right_index=True, how='left'
            )
            portfolio_value[f"{stock}_价值"] = portfolio_value[f"{stock}_价值"].fillna(0)

        # 计算每日总资产
        stock_columns = [col for col in portfolio_value.columns if '_价值' in col]
        portfolio_value['股票价值'] = portfolio_value[stock_columns].sum(axis=1)
        portfolio_value['总资产'] = portfolio_value['现金'] + portfolio_value['股票价值']

        # 计算基准
        benchmark = ak.stock_zh_index_hist_csindex(symbol="H30374")  # 中证科技指数
        benchmark['日期'] = pd.to_datetime(benchmark['日期'])
        benchmark = benchmark.set_index('日期')['收盘']
        portfolio_value = portfolio_value.merge(benchmark.rename('科技指数'),
                                                left_index=True, right_index=True, how='left')

        # 计算收益率
        portfolio_value['策略收益'] = portfolio_value['总资产'].pct_change(fill_method=None)
        portfolio_value['基准收益'] = portfolio_value['科技指数'].pct_change(fill_method=None)

        return portfolio_value

    def visualize_performance(self, portfolio_value):
        """可视化表现"""
        plt.figure(figsize=(14, 10))

        # 资产走势
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_value['总资产'], label='策略总资产', linewidth=2)
        plt.plot(portfolio_value['科技指数'] * (self.capital / portfolio_value['科技指数'].iloc[0]),
                 label='科技指数', linestyle='--')
        plt.title('投资组合表现')
        plt.legend()
        plt.grid(True)

        # 年度收益对比
        plt.subplot(2, 1, 2)
        annual_return = portfolio_value['总资产'].resample('YE').last().pct_change(fill_method=None)
        benchmark_return = portfolio_value['科技指数'].resample('YE').last().pct_change(fill_method=None)
        years = annual_return.index.year[1:]

        bar_width = 0.35
        plt.bar(years - bar_width / 2, annual_return[1:] * 100, bar_width, label='策略收益')
        plt.bar(years + bar_width / 2, benchmark_return[1:] * 100, bar_width, label='基准收益')

        plt.title('年度收益对比(%)')
        plt.legend()
        plt.grid(axis='y')

        plt.tight_layout()
        plt.savefig('tmp/10/portfolio_performance.png')
        # plt.show()

    def run(self):
        """运行投资系统"""
        print("步骤1: 筛选目标公司...")
        self.portfolio_optimization()

        print("步骤2: 执行回测...")
        portfolio_value = self.backtest_portfolio()

        print("步骤3: 可视化结果...")
        self.visualize_performance(portfolio_value)

        print("投资系统运行完成!")
        print(f'投资记录：\n{self.transaction_log}')


if __name__ == "__main__":
    # 初始化投资系统
    investment_system = QuantumInvestmentSystem(capital=1000000, risk_level=0.3)

    # 运行系统
    investment_system.run()
