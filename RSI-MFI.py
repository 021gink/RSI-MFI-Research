import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import re
import json
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.font_manager as fm
from scipy.stats import linregress
import warnings


# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 配置参数
PERIOD = 14  # RSI/MFI计算周期
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
}

# 设置中文字体
try:
    # 尝试使用系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果失败，尝试安装并使用思源黑体
    try:
        font_path = fm.findfont(fm.FontProperties(family=['Source Han Sans CN']))
        plt.rcParams['font.sans-serif'] = ['Source Han Sans CN', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("无法设置中文字体，图表可能显示异常")

def get_stock_data(symbol, count=300):
    """爬取股票历史数据（终极稳定版）"""
    # 验证股票代码格式
    if not re.match(r'^(sh|sz)\d{6}$', symbol):
        st.error(f"无效的股票代码格式: {symbol}")
        return None
    
    # 构建请求URL
    url = f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,{count},qfq'
    
    try:
        with st.spinner(f"正在获取 {symbol} 的历史数据..."):
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
        
        # 高级JSON解析
        json_data = None
        json_text = r.text
        
        # JSON修复层1: 基本修复
        json_text = re.sub(r',\s*]', ']', json_text)  # 修复尾部逗号
        json_text = json_text.replace("'", '"')        # 单引号转双引号
        
        try:
            json_data = json.loads(json_text)
        except json.JSONDecodeError:
            # JSON修复层2: 键名加引号
            json_text = re.sub(r'(\w+):', r'"\1":', json_text)
            try:
                json_data = json.loads(json_text)
            except:
                # JSON修复层3: 终极修复
                json_text = json_text.replace("None", "null")
                json_text = re.sub(r'(\d{4}-\d{2}-\d{2})', r'"\1"', json_text)
                json_text = re.sub(r'(\d+\.\d+)', r'"\1"', json_text)
                json_data = json.loads(json_text)
        
        # 深度数据提取
        if not json_data or 'data' not in json_data:
            st.error("API返回数据格式异常")
            return None
            
        stock_data = json_data['data'].get(symbol)
        if not stock_data:
            st.error(f"未找到 {symbol} 的股票数据")
            return None
            
        # 多种方式获取K线数据
        data = stock_data.get('qfqday') or stock_data.get('day') or stock_data.get('data')
        
        if not data or not isinstance(data, list) or len(data) == 0:
            st.error("未找到有效的K线数据")
            return None
        
        st.info(f"获取到原始数据 {len(data)} 条")
        
        # 智能数据结构处理
        processed_data = []
        
        for idx, item in enumerate(data):
            # 跳过无效行
            if not isinstance(item, list) or len(item) < 5:
                continue
                
            # 创建新行
            row = {}
            
            # 日期总是第一列
            try:
                row['date'] = str(item[0])
            except:
                continue
                
            # 提取数值列
            numeric_values = []
            for i in range(1, len(item)):
                value = item[i]
                
                # 转换数字类型
                if isinstance(value, (int, float)):
                    numeric_values.append(value)
                elif isinstance(value, str):
                    # 清理字符串
                    clean_value = value.replace(',', '').strip()
                    
                    # 检查是否为数字
                    if clean_value.replace('.', '', 1).isdigit():
                        numeric_values.append(float(clean_value))
                    else:
                        # 非数字值，跳过
                        continue
                else:
                    # 非数字类型，跳过
                    continue
            
            # 确保有足够的数值列
            if len(numeric_values) < 5:
                continue
                
            # 分配OHLCV
            row['open'] = numeric_values[0]
            row['close'] = numeric_values[1]
            row['high'] = numeric_values[2]
            row['low'] = numeric_values[3]
            row['volume'] = numeric_values[4]
            
            processed_data.append(row)
        
        # 创建DataFrame
        if not processed_data:
            st.error("数据清洗后无有效记录")
            return None
            
        df = pd.DataFrame(processed_data)
        st.info(f"成功创建DataFrame，包含 {len(df)} 条记录")
        
        # 专业数据清洗
        df = df.iloc[::-1].reset_index(drop=True)  # 时间升序排列
        
        # 类型转换（安全处理）
        numeric_cols = ['open', 'close', 'high', 'low']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 成交量特殊处理（转为整数）
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        
        # 智能日期解析
        df['date'] = pd.to_datetime(
            df['date'].astype(str).str.extract(r'(\d{4}-\d{1,2}-\d{1,2})')[0],
            errors='coerce'
        )
        
        # 删除无效日期
        df = df.dropna(subset=['date'])
        
        # 最终数据验证
        if df.empty:
            st.error("清洗后无有效数据")
            return None
            
        # 排序并重置索引
        df = df.sort_values('date').reset_index(drop=True)
        
        st.success(f"成功处理 {len(df)} 条有效K线数据")
        st.info(f"时间范围: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
        
        return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"网络请求失败: {str(e)}")
    except Exception as e:
        st.error(f"数据处理错误: {str(e)}")
    
    return None

def calculate_rsi(df, period):
    """计算相对强弱指数(RSI) - 优化版"""
    delta = df['close'].diff()
    
    # 处理初始NaN值
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 使用指数移动平均
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # 避免除以零
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 填充初始NaN值 - 避免inplace操作
    df = df.copy()
    df['RSI'] = df['RSI'].fillna(50)
    return df

def calculate_mfi(df, period):
    """计算资金流量指标(MFI) - 优化版"""
    # 计算典型价格和原始资金流
    tp = (df['high'] + df['low'] + df['close']) / 3
    raw_mf = tp * df['volume']
    
    # 资金流向方向
    flow_direction = np.where(tp > tp.shift(1), 1, np.where(tp < tp.shift(1), -1, 0))
    
    # 计算正负资金流
    pos_mf = np.where(flow_direction > 0, raw_mf, 0)
    neg_mf = np.where(flow_direction < 0, raw_mf, 0)
    
    # 平滑计算
    pos_flow = pd.Series(pos_mf).rolling(period, min_periods=1).sum()
    neg_flow = pd.Series(neg_mf).rolling(period, min_periods=1).sum()
    
    # 计算MFI，避免除以零
    money_ratio = pos_flow / (pos_flow + neg_flow).replace(0, np.nan)
    df['MFI'] = 100 * money_ratio
    
    # 填充NaN值 - 避免inplace操作
    df = df.copy()
    df['MFI'] = df['MFI'].fillna(50)
    return df

def generate_hybrid_indicator(df, rsi_weight, mfi_weight, 
                             strong_buy_threshold, buy_threshold, 
                             strong_sell_threshold, sell_threshold):
    """生成RSI-MFI混合指标 - 增强版"""
    # 标准化指标
    df['Norm_RSI'] = (df['RSI'] - 30) / (70 - 30) * 100
    df['Norm_MFI'] = (df['MFI'] - 30) / (70 - 30) * 100
    
    # 使用传入的权重生成混合指标
    df['Hybrid'] = (df['Norm_RSI'] * rsi_weight + 
                   df['Norm_MFI'] * mfi_weight)
    
    # 限制在0-100范围内
    df['Hybrid'] = df['Hybrid'].clip(0, 100)
    
    # 使用传入的阈值生成信号
    df['Signal'] = np.select(
        [
            (df['Hybrid'] > strong_buy_threshold) & (df['close'] > df['close'].shift(5)),
            (df['Hybrid'] > buy_threshold) & (df['close'] > df['close'].shift(5)),
            (df['Hybrid'] < strong_sell_threshold) & (df['close'] < df['close'].shift(5)),
            (df['Hybrid'] < sell_threshold) & (df['close'] < df['close'].shift(5))
        ],
        [2, 1, -2, -1],  # 2:强买, 1:买, -1:卖, -2:强卖
        default=0
    )
    
    # 信号平滑
    df['Signal'] = df['Signal'].rolling(5, min_periods=1).mean().round()
    return df

def calculate_confidence(df):
    """计算指标可信度 - 终极稳健版"""
    # 使用最近10天数据，确保有足够样本
    recent = df.tail(10)
    if len(recent) < 5:  # 最少需要5个数据点
        return 50.0  # 返回中性可信度
    
    # 计算价格变化百分比
    price_changes = recent['close'].pct_change().fillna(0) * 100
    
    # 计算信号变化
    signal_changes = recent['Signal'].diff().fillna(0)
    
    # 计算信号稳定性（信号变化的方差）
    signal_variance = signal_changes.var()
    
    # 情况1：信号完全稳定（无变化）
    if signal_variance == 0:
        return calculate_stability_confidence(recent, price_changes)
    
    # 情况2：信号有变化，但值相同（线性回归会失败）
    if signal_changes.nunique() == 1:
        return calculate_uniform_signal_confidence(recent, price_changes)
    
    # 情况3：正常信号变化
    return calculate_normal_confidence(recent, signal_changes, price_changes)

def calculate_stability_confidence(recent, price_changes):
    """计算信号稳定时的可信度"""
    # 计算价格波动率
    price_volatility = price_changes.abs().mean()
    
    # 获取最新信号
    current_signal = recent['Signal'].iloc[-1]
    
    # 计算价格趋势方向
    price_trend = "up" if recent['close'].iloc[-1] > recent['close'].iloc[0] else "down"
    
    # 判断信号与价格趋势是否一致
    signal_match = False
    if (current_signal > 0 and price_trend == "up") or (current_signal < 0 and price_trend == "down"):
        signal_match = True
    
    # 可信度计算逻辑
    if price_volatility < 0.5:  # 极低波动市场
        return 60.0 if signal_match else 40.0
    elif price_volatility < 1.5:  # 低波动市场
        return 70.0 if signal_match else 50.0
    elif price_volatility < 3.0:  # 中等波动市场
        return 65.0 if signal_match else 45.0
    else:  # 高波动市场
        return 55.0 if signal_match else 35.0

def calculate_uniform_signal_confidence(recent, price_changes):
    """计算信号值相同但非零变化时的可信度"""
    # 计算价格与信号的相关性
    price_corr = recent['close'].corr(recent['Signal'])
    
    # 计算信号方向准确性
    correct_direction = 0
    for i in range(1, len(recent)):
        signal_direction = np.sign(recent['Signal'].iloc[i] - recent['Signal'].iloc[i-1])
        price_direction = np.sign(price_changes.iloc[i])
        
        if signal_direction != 0 and signal_direction == price_direction:
            correct_direction += 1
    
    accuracy = correct_direction / (len(recent) - 1) * 100 if len(recent) > 1 else 50
    
    # 综合可信度
    confidence = 40 + min(30, abs(price_corr) * 30) + min(30, accuracy * 0.3)
    return min(95, max(5, confidence))

def calculate_normal_confidence(recent, signal_changes, price_changes):
    """计算正常信号变化时的可信度"""
    try:
        # 计算线性回归
        slope, intercept, r_value, p_value, std_err = linregress(
            signal_changes, price_changes
        )
    except ValueError:  # 处理所有x值相同的情况（理论上不应发生）
        return calculate_uniform_signal_confidence(recent, price_changes)
    
    # 计算信号准确性
    correct_signals = 0
    total_signals = 0
    
    for i in range(1, len(recent)):
        signal = recent['Signal'].iloc[i]
        prev_signal = recent['Signal'].iloc[i-1]
        price_change = price_changes.iloc[i]
        
        # 只考虑有变化的信号
        if signal != prev_signal:
            total_signals += 1
            if (signal > prev_signal and price_change > 0) or (signal < prev_signal and price_change < 0):
                correct_signals += 1
    
    # 计算准确率（避免除以零）
    accuracy = (correct_signals / total_signals * 100) if total_signals > 0 else 50
    
    # 综合可信度评分
    confidence = min(100, max(0, (abs(r_value) * 70 + accuracy * 0.3)))
    
    # 考虑趋势强度因子
    price_volatility = price_changes.abs().mean()
    if price_volatility < 0.5:  # 低波动市场
        confidence *= 0.9
    elif price_volatility > 2.0:  # 高波动市场
        confidence = min(100, confidence * 1.05)
    
    return round(confidence, 1)

def visualize_results_plotly(df, symbol):
    """使用Plotly可视化分析结果 - 交互式图表"""
    # 创建子图
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=(f'{symbol} 价格走势与交易信号', 
                                       '动量指标对比', 
                                       '成交量与资金流向'),
                        row_heights=[0.5, 0.3, 0.2])
    
    # 价格走势
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], 
                            mode='lines', name='收盘价',
                            line=dict(color='#1f77b4', width=2)),
                 row=1, col=1)
    
    # 标记交易信号
    buy_signals = df[df['Signal'] > 0]
    strong_buy = df[df['Signal'] == 2]
    sell_signals = df[df['Signal'] < 0]
    strong_sell = df[df['Signal'] == -2]
    
    fig.add_trace(go.Scatter(x=strong_buy['date'], y=strong_buy['close'],
                            mode='markers', name='强烈买入',
                            marker=dict(symbol='triangle-up', size=10, color='darkgreen')),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=buy_signals[buy_signals['Signal'] == 1]['date'], 
                            y=buy_signals[buy_signals['Signal'] == 1]['close'],
                            mode='markers', name='买入',
                            marker=dict(symbol='triangle-up', size=8, color='limegreen')),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=sell_signals[sell_signals['Signal'] == -1]['date'], 
                            y=sell_signals[sell_signals['Signal'] == -1]['close'],
                            mode='markers', name='卖出',
                            marker=dict(symbol='triangle-down', size=8, color='salmon')),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=strong_sell['date'], y=strong_sell['close'],
                            mode='markers', name='强烈卖出',
                            marker=dict(symbol='triangle-down', size=10, color='darkred')),
                 row=1, col=1)
    
    # 指标对比
    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], 
                            mode='lines', name='RSI',
                            line=dict(color='#ff7f0e', width=1.5)),
                 row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df['date'], y=df['MFI'], 
                            mode='lines', name='MFI',
                            line=dict(color='#2ca02c', width=1.5)),
                 row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df['date'], y=df['Hybrid'], 
                            mode='lines', name='混合指标',
                            line=dict(color='#d62728', width=2.5)),
                 row=2, col=1)
    
    # 阈值线
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=50, line_color="grey", opacity=0.7, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    # 成交量
    colors = np.where(df['close'] >= df['close'].shift(1), 'green', 'red')
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], 
                         name='成交量', marker_color=colors, opacity=0.7),
                 row=3, col=1)
    
    # 资金流分析
    money_flow = df['volume'] * (df['close'] - df['open'])
    pos_flow = money_flow.copy()
    neg_flow = money_flow.copy()
    pos_flow[pos_flow < 0] = 0
    neg_flow[neg_flow > 0] = 0
    
    fig.add_trace(go.Scatter(x=df['date'], y=pos_flow.cumsum(), 
                            mode='lines', name='资金流入累积',
                            line=dict(color='green', width=2)),
                 row=3, col=1)
    
    fig.add_trace(go.Scatter(x=df['date'], y=neg_flow.cumsum(), 
                            mode='lines', name='资金流出累积',
                            line=dict(color='red', width=2)),
                 row=3, col=1)
    
    # 更新布局
    fig.update_layout(
        height=800,
        title_text=f'{symbol} 技术分析 - RSI-MFI混合指标',
        hovermode='x unified',
        showlegend=True
    )
    
    # 更新y轴范围
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="指标值", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="成交量/资金流", row=3, col=1)
    
    return fig

def determine_trend(df, window=5):
    """
    专业趋势判断函数
    返回: (趋势方向, 趋势强度, 趋势描述)
    """
    # 获取最近n天的收盘价
    recent = df['close'].tail(window)
    
    # 1. 简单价格比较
    simple_up = recent.iloc[-1] > recent.iloc[0]
    
    # 2. 移动平均趋势
    ma_5 = recent.rolling(3).mean()
    ma_trend = "上涨" if ma_5.iloc[-1] > ma_5.iloc[-2] else "下跌"
    
    # 3. 线性回归趋势
    x = np.arange(len(recent))
    slope, _, _, _, _ = linregress(x, recent)
    regression_trend = "上涨" if slope > 0 else "下跌"
    
    # 4. 趋势强度指标
    max_price = recent.max()
    min_price = recent.min()
    range_pct = (max_price - min_price) / min_price * 100
    
    # 5. 趋势一致性
    up_days = sum(recent.diff().dropna() > 0)
    down_days = sum(recent.diff().dropna() < 0)
    consistency = "上涨主导" if up_days > down_days else "下跌主导"
    
    # 6. 波动率分析
    volatility = "高波动" if range_pct > 3 else "低波动"
    
    # 综合判断趋势方向
    if simple_up and ma_trend == "上涨" and regression_trend == "上涨":
        trend_direction = "上涨"
    elif not simple_up and ma_trend == "下跌" and regression_trend == "下跌":
        trend_direction = "下跌"
    else:
        trend_direction = "震荡"
    
    # 判断趋势强度
    if trend_direction == "上涨":
        if range_pct > 3:  # 显著波动
            if up_days >= window * 0.6:  # 多数天数上涨
                trend_strength = "强势"
            else:
                trend_strength = "震荡"
        else:
            trend_strength = "微弱"
    elif trend_direction == "下跌":
        if range_pct > 3:
            if down_days >= window * 0.6:
                trend_strength = "强势"
            else:
                trend_strength = "震荡"
        else:
            trend_strength = "微弱"
    else:
        trend_strength = "无方向"
    
    # 生成趋势描述
    if trend_strength == "强势" and trend_direction == "上涨":
        trend_desc = "强劲上涨趋势"
    elif trend_strength == "震荡" and trend_direction == "上涨":
        trend_desc = "震荡上行"
    elif trend_strength == "微弱" and trend_direction == "上涨":
        trend_desc = "微幅上涨"
    elif trend_strength == "强势" and trend_direction == "下跌":
        trend_desc = "大幅下跌趋势"
    elif trend_strength == "震荡" and trend_direction == "下跌":
        trend_desc = "震荡下行"
    elif trend_strength == "微弱" and trend_direction == "下跌":
        trend_desc = "微幅下跌"
    else:
        trend_desc = "横盘整理"
    
    # 添加波动率信息
    trend_desc += f" ({volatility})"
    
    return trend_direction, trend_strength, trend_desc

def generate_report(df, symbol):
    """生成专业分析报告 - 终极稳健版"""
    latest = df.iloc[-1]
    
    # 计算指标可信度 - 三重保护
    try:
        confidence = calculate_confidence(df)
    except Exception as e:
        st.warning(f"可信度计算遇到意外错误，使用保守估计: {str(e)}")
        confidence = 50.0
    
    # 专业趋势分析
    try:
        # 获取不同时间框架的趋势分析
        trend_results = []
        for days in [5, 10, 20]:
            if len(df) >= days:
                result = determine_trend(df.tail(days), window=days)
                trend_results.append(result)
            else:
                trend_results.append(("未知", "未知", f"{days}日数据不足"))
        
        trend_5d_dir, trend_5d_str, trend_5d_desc = trend_results[0]
        trend_10d_dir, trend_10d_str, trend_10d_desc = trend_results[1]
        trend_20d_dir, trend_20d_str, trend_20d_desc = trend_results[2]
        
        # 趋势一致性分析
        if all(r[0] == trend_5d_dir for r in trend_results if r[0] != "未知"):
            trend_consistency = "多周期趋势一致"
            trend_emoji = "✅"
        elif trend_5d_dir != trend_20d_dir and trend_20d_dir != "未知":
            trend_consistency = "短期与长期趋势背离"
            trend_emoji = "⚠️"
        else:
            trend_consistency = "趋势分化"
            trend_emoji = "➖"
    except Exception as e:
        st.error(f"趋势分析错误: {str(e)}")
        trend_5d_desc = "分析失败"
        trend_10d_desc = "分析失败"
        trend_20d_desc = "分析失败"
        trend_consistency = "分析失败"
        trend_emoji = "❌"

    # 信号分析
    if latest['Signal'] == 2:
        signal_analysis = "强烈买入信号：价格强劲且资金持续流入"
        signal_emoji = "🚀"
    elif latest['Signal'] == 1:
        signal_analysis = "买入信号：价格和资金流表现积极"
        signal_emoji = "✅"
    elif latest['Signal'] == -1:
        signal_analysis = "卖出信号：价格疲软且资金流出"
        signal_emoji = "⚠️"
    elif latest['Signal'] == -2:
        signal_analysis = "强烈卖出信号：价格下跌加速且资金大幅流出"
        signal_emoji = "🔥"
    else:
        signal_analysis = "中性信号：市场处于盘整阶段"
        signal_emoji = "➖"
    
    # 背离检测 - 添加边界检查
    try:
        divergence = ""
        if len(df) > 10:
            if (df['close'].iloc[-1] > df['close'].iloc[-10] and 
                df['Hybrid'].iloc[-1] < df['Hybrid'].iloc[-10] and
                df['Hybrid'].iloc[-1] < 70):
                divergence = "顶背离警告：价格创新高但指标走弱，可能反转 ⚠️"
            
            if (df['close'].iloc[-1] < df['close'].iloc[-10] and 
                df['Hybrid'].iloc[-1] > df['Hybrid'].iloc[-10] and
                df['Hybrid'].iloc[-1] > 30):
                divergence = "底背离信号：价格创新低但指标走强，可能反弹 ⬆️"
    except IndexError:
        divergence = "背离检测失败（数据不足）"
    
    # 价格位置分析 - 添加边界检查
    try:
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_to_high = (recent_high - latest['close']) / recent_high * 100
        current_to_low = (latest['close'] - recent_low) / recent_low * 100
        
        if current_to_high < 2:
            position = "接近近期高点"
        elif current_to_low < 2:
            position = "接近近期低点"
        else:
            position = "中间区域"
    except:
        position = "未知"
        current_to_high = 0
        current_to_low = 0
    
    # 涨跌幅计算 - 添加边界检查
    try:
        change_1d = (latest['close']/df['close'].iloc[-2]-1)*100 if len(df) > 1 else 0
        change_5d = (latest['close']/df['close'].iloc[-5]-1)*100 if len(df) > 5 else 0
        change_20d = (latest['close']/df['close'].iloc[-20]-1)*100 if len(df) > 20 else 0
    except IndexError:
        change_1d = change_5d = change_20d = 0
    
    # 生成专业报告
    report = f"""
    ## {symbol} 技术分析报告
    **生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
    **数据范围**: {df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {df['date'].iloc[-1].strftime('%Y-%m-%d')}  
    **分析周期**: {PERIOD}日  
    **可信度计算模型**: 三重保护机制（稳定/变化/正常信号处理）
    **指标可信度**: {confidence:.1f}% ({'高可信度' if confidence > 70 else '中可信度' if confidence > 50 else '低可信度'}) 告
    
    ### 最新行情
    - **当前日期**: {latest['date'].strftime('%Y-%m-%d')}
    - **当前价**: {latest['close']:.2f}
    - **成交量**: {latest['volume']/10000:.2f}万手
    - **价格位置**: {position} (距高点: -{current_to_high:.2f}%, 距低点: +{current_to_low:.2f}%)
    - **涨跌幅(1日)**: {change_1d:.2f}%
    - **涨跌幅(5日)**: {change_5d:.2f}%
    - **涨跌幅(20日)**: {change_20d:.2f}%
    
    ### 技术指标
    - **RSI**: {latest['RSI']:.2f} ({'超买' if latest['RSI']>70 else '超卖' if latest['RSI']<30 else '中性'})
    - **MFI**: {latest['MFI']:.2f} ({'资金流入强劲' if latest['MFI']>70 else '资金流出明显' if latest['MFI']<30 else '资金流向中性'})
    - **混合指标**: {latest['Hybrid']:.2f} ({'强势区域' if latest['Hybrid']>70 else '弱势区域' if latest['Hybrid']<30 else '中性区域'})
    
    ### 专业趋势分析
    - **短期趋势 (5日)**: {trend_5d_desc}
    - **中期趋势 (10日)**: {trend_10d_desc}
    - **长期趋势 (20日)**: {trend_20d_desc}
    - **趋势一致性**: {trend_emoji} {trend_consistency}
    
    ### 交易信号
    {signal_emoji} **{signal_analysis}**
    
    ### 背离检测
    {divergence}
    
    ### 操作建议
    {generate_trading_recommendation(latest, trend_consistency, confidence)}
    """
    
    return report

def generate_trading_recommendation(latest, trend_consistency, confidence):
    """生成操作建议 - 增强版"""
    signal = latest['Signal']
    hybrid = latest['Hybrid']
    rsi = latest['RSI']
    
    # 考虑指标一致性
    indicator_agreement = "一致" if (signal > 0 and hybrid > 50 and rsi > 50) or \
                                   (signal < 0 and hybrid < 50 and rsi < 50) else "分歧"
    
    # 高可信度情况
    if confidence > 70:
        if signal == 2:
            return "建议：积极买入，高可信度强烈买入信号"
        elif signal == 1:
            if indicator_agreement == "一致":
                return "建议：分批建仓，高可信度买入信号且指标一致"
            else:
                return "建议：谨慎买入，高可信度但指标有分歧"
        elif signal == -1:
            return "建议：考虑减仓，高可信度卖出信号"
        elif signal == -2:
            return "建议：果断卖出，高可信度强烈卖出信号"
        else:
            if hybrid > 70:
                return "建议：持仓观望，指标高位但无明确卖出信号"
            elif hybrid < 30:
                return "建议：关注机会，指标低位但无明确买入信号"
    
    # 中可信度情况
    elif confidence > 50:
        if signal > 0:
            if trend_consistency == "多周期趋势一致":
                return "建议：轻仓参与，趋势一致但信号可信度中等"
            else:
                return "建议：保持观望，信号与趋势不一致"
        elif signal < 0:
            return "建议：减仓避险，中等可信度卖出信号"
        else:
            return "建议：保持观望，市场方向不明"
    
    # 低可信度情况
    else:
        if hybrid > 70:
            return "建议：不宜追高，指标高位但可信度低"
        elif hybrid < 30:
            return "建议：不宜杀跌，指标低位但可信度低"
        else:
            if rsi > 70 or hybrid > 70:
                return "建议：警惕风险，部分指标显示超买"
            elif rsi < 30 or hybrid < 30:
                return "建议：关注机会，部分指标显示超卖"
            else:
                return "建议：保持观望，市场方向不明"

def analyze_stock(symbol, period, rsi_weight, mfi_weight,
                 strong_buy_threshold, buy_threshold,
                 strong_sell_threshold, sell_threshold):
    """主分析函数"""
    # 获取数据
    df = get_stock_data(symbol)
    
    # 计算指标 - 传入所有参数
    df = calculate_rsi(df, period)
    df = calculate_mfi(df, period)
    df = generate_hybrid_indicator(
        df, rsi_weight, mfi_weight,
        strong_buy_threshold, buy_threshold,
        strong_sell_threshold, sell_threshold
    )

    
    # 生成报告
    report = generate_report(df, symbol)
    st.markdown(report)
    
    # 可视化
    fig = visualize_results_plotly(df, symbol)
    st.plotly_chart(fig, use_container_width=True)
    
    # 性能统计
    elapsed = time.time() - start_time
    st.info(f"分析完成! 耗时: {elapsed:.2f}秒")

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import re
import json
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.font_manager as fm
from scipy.stats import linregress
import warnings


# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 配置参数
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
}

# 设置中文字体
try:
    # 尝试使用系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果失败，尝试安装并使用思源黑体
    try:
        font_path = fm.findfont(fm.FontProperties(family=['Source Han Sans CN']))
        plt.rcParams['font.sans-serif'] = ['Source Han Sans CN', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("无法设置中文字体，图表可能显示异常")

def get_stock_data(symbol, count=300):
    """爬取股票历史数据（终极稳定版）"""
    # 验证股票代码格式
    if not re.match(r'^(sh|sz)\d{6}$', symbol):
        st.error(f"无效的股票代码格式: {symbol}")
        return None
    
    # 构建请求URL
    url = f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,{count},qfq'
    
    try:
        with st.spinner(f"正在获取 {symbol} 的历史数据..."):
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
        
        # 高级JSON解析
        json_data = None
        json_text = r.text
        
        # JSON修复层1: 基本修复
        json_text = re.sub(r',\s*]', ']', json_text)  # 修复尾部逗号
        json_text = json_text.replace("'", '"')        # 单引号转双引号
        
        try:
            json_data = json.loads(json_text)
        except json.JSONDecodeError:
            # JSON修复层2: 键名加引号
            json_text = re.sub(r'(\w+):', r'"\1":', json_text)
            try:
                json_data = json.loads(json_text)
            except:
                # JSON修复层3: 终极修复
                json_text = json_text.replace("None", "null")
                json_text = re.sub(r'(\d{4}-\d{2}-\d{2})', r'"\1"', json_text)
                json_text = re.sub(r'(\d+\.\d+)', r'"\1"', json_text)
                json_data = json.loads(json_text)
        
        # 深度数据提取
        if not json_data or 'data' not in json_data:
            st.error("API返回数据格式异常")
            return None
            
        stock_data = json_data['data'].get(symbol)
        if not stock_data:
            st.error(f"未找到 {symbol} 的股票数据")
            return None
            
        # 多种方式获取K线数据
        data = stock_data.get('qfqday') or stock_data.get('day') or stock_data.get('data')
        
        if not data or not isinstance(data, list) or len(data) == 0:
            st.error("未找到有效的K线数据")
            return None
        
        st.info(f"获取到原始数据 {len(data)} 条")
        
        # 智能数据结构处理
        processed_data = []
        
        for idx, item in enumerate(data):
            # 跳过无效行
            if not isinstance(item, list) or len(item) < 5:
                continue
                
            # 创建新行
            row = {}
            
            # 日期总是第一列
            try:
                row['date'] = str(item[0])
            except:
                continue
                
            # 提取数值列
            numeric_values = []
            for i in range(1, len(item)):
                value = item[i]
                
                # 转换数字类型
                if isinstance(value, (int, float)):
                    numeric_values.append(value)
                elif isinstance(value, str):
                    # 清理字符串
                    clean_value = value.replace(',', '').strip()
                    
                    # 检查是否为数字
                    if clean_value.replace('.', '', 1).isdigit():
                        numeric_values.append(float(clean_value))
                    else:
                        # 非数字值，跳过
                        continue
                else:
                    # 非数字类型，跳过
                    continue
            
            # 确保有足够的数值列
            if len(numeric_values) < 5:
                continue
                
            # 分配OHLCV
            row['open'] = numeric_values[0]
            row['close'] = numeric_values[1]
            row['high'] = numeric_values[2]
            row['low'] = numeric_values[3]
            row['volume'] = numeric_values[4]
            
            processed_data.append(row)
        
        # 创建DataFrame
        if not processed_data:
            st.error("数据清洗后无有效记录")
            return None
            
        df = pd.DataFrame(processed_data)
        st.info(f"成功创建DataFrame，包含 {len(df)} 条记录")
        
        # 专业数据清洗
        df = df.iloc[::-1].reset_index(drop=True)  # 时间升序排列
        
        # 类型转换（安全处理）
        numeric_cols = ['open', 'close', 'high', 'low']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 成交量特殊处理（转为整数）
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        
        # 智能日期解析
        df['date'] = pd.to_datetime(
            df['date'].astype(str).str.extract(r'(\d{4}-\d{1,2}-\d{1,2})')[0],
            errors='coerce'
        )
        
        # 删除无效日期
        df = df.dropna(subset=['date'])
        
        # 最终数据验证
        if df.empty:
            st.error("清洗后无有效数据")
            return None
            
        # 排序并重置索引
        df = df.sort_values('date').reset_index(drop=True)
        
        st.success(f"成功处理 {len(df)} 条有效K线数据")
        st.info(f"时间范围: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
        
        return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"网络请求失败: {str(e)}")
    except Exception as e:
        st.error(f"数据处理错误: {str(e)}")
    
    return None

def calculate_rsi(df, period):
    """计算相对强弱指数(RSI) - 优化版"""
    delta = df['close'].diff()
    
    # 处理初始NaN值
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 使用指数移动平均
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # 避免除以零
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 填充初始NaN值 - 避免inplace操作
    df = df.copy()
    df['RSI'] = df['RSI'].fillna(50)
    return df

def calculate_mfi(df, period):
    """计算资金流量指标(MFI) - 优化版"""
    # 计算典型价格和原始资金流
    tp = (df['high'] + df['low'] + df['close']) / 3
    raw_mf = tp * df['volume']
    
    # 资金流向方向
    flow_direction = np.where(tp > tp.shift(1), 1, np.where(tp < tp.shift(1), -1, 0))
    
    # 计算正负资金流
    pos_mf = np.where(flow_direction > 0, raw_mf, 0)
    neg_mf = np.where(flow_direction < 0, raw_mf, 0)
    
    # 平滑计算
    pos_flow = pd.Series(pos_mf).rolling(period, min_periods=1).sum()
    neg_flow = pd.Series(neg_mf).rolling(period, min_periods=1).sum()
    
    # 计算MFI，避免除以零
    money_ratio = pos_flow / (pos_flow + neg_flow).replace(0, np.nan)
    df['MFI'] = 100 * money_ratio
    
    # 填充NaN值 - 避免inplace操作
    df = df.copy()
    df['MFI'] = df['MFI'].fillna(50)
    return df

def generate_hybrid_indicator(df, rsi_weight, mfi_weight, 
                             strong_buy_threshold, buy_threshold, 
                             strong_sell_threshold, sell_threshold):
    """生成RSI-MFI混合指标 - 增强版"""
    # 标准化指标
    df['Norm_RSI'] = (df['RSI'] - 30) / (70 - 30) * 100
    df['Norm_MFI'] = (df['MFI'] - 30) / (70 - 30) * 100
    
    # 使用传入的权重生成混合指标
    df['Hybrid'] = (df['Norm_RSI'] * rsi_weight + 
                   df['Norm_MFI'] * mfi_weight)
    
    # 限制在0-100范围内
    df['Hybrid'] = df['Hybrid'].clip(0, 100)
    
    # 使用传入的阈值生成信号
    df['Signal'] = np.select(
        [
            (df['Hybrid'] > strong_buy_threshold) & (df['close'] > df['close'].shift(5)),
            (df['Hybrid'] > buy_threshold) & (df['close'] > df['close'].shift(5)),
            (df['Hybrid'] < strong_sell_threshold) & (df['close'] < df['close'].shift(5)),
            (df['Hybrid'] < sell_threshold) & (df['close'] < df['close'].shift(5))
        ],
        [2, 1, -2, -1],  # 2:强买, 1:买, -1:卖, -2:强卖
        default=0
    )
    
    # 信号平滑
    df['Signal'] = df['Signal'].rolling(5, min_periods=1).mean().round()
    return df

def calculate_confidence(df):
    """计算指标可信度 - 终极稳健版"""
    # 使用最近10天数据，确保有足够样本
    recent = df.tail(10)
    if len(recent) < 5:  # 最少需要5个数据点
        return 50.0  # 返回中性可信度
    
    # 计算价格变化百分比
    price_changes = recent['close'].pct_change().fillna(0) * 100
    
    # 计算信号变化
    signal_changes = recent['Signal'].diff().fillna(0)
    
    # 计算信号稳定性（信号变化的方差）
    signal_variance = signal_changes.var()
    
    # 情况1：信号完全稳定（无变化）
    if signal_variance == 0:
        return calculate_stability_confidence(recent, price_changes)
    
    # 情况2：信号有变化，但值相同（线性回归会失败）
    if signal_changes.nunique() == 1:
        return calculate_uniform_signal_confidence(recent, price_changes)
    
    # 情况3：正常信号变化
    return calculate_normal_confidence(recent, signal_changes, price_changes)

def calculate_stability_confidence(recent, price_changes):
    """计算信号稳定时的可信度"""
    # 计算价格波动率
    price_volatility = price_changes.abs().mean()
    
    # 获取最新信号
    current_signal = recent['Signal'].iloc[-1]
    
    # 计算价格趋势方向
    price_trend = "up" if recent['close'].iloc[-1] > recent['close'].iloc[0] else "down"
    
    # 判断信号与价格趋势是否一致
    signal_match = False
    if (current_signal > 0 and price_trend == "up") or (current_signal < 0 and price_trend == "down"):
        signal_match = True
    
    # 可信度计算逻辑
    if price_volatility < 0.5:  # 极低波动市场
        return 60.0 if signal_match else 40.0
    elif price_volatility < 1.5:  # 低波动市场
        return 70.0 if signal_match else 50.0
    elif price_volatility < 3.0:  # 中等波动市场
        return 65.0 if signal_match else 45.0
    else:  # 高波动市场
        return 55.0 if signal_match else 35.0

def calculate_uniform_signal_confidence(recent, price_changes):
    """计算信号值相同但非零变化时的可信度"""
    # 计算价格与信号的相关性
    price_corr = recent['close'].corr(recent['Signal'])
    
    # 计算信号方向准确性
    correct_direction = 0
    for i in range(1, len(recent)):
        signal_direction = np.sign(recent['Signal'].iloc[i] - recent['Signal'].iloc[i-1])
        price_direction = np.sign(price_changes.iloc[i])
        
        if signal_direction != 0 and signal_direction == price_direction:
            correct_direction += 1
    
    accuracy = correct_direction / (len(recent) - 1) * 100 if len(recent) > 1 else 50
    
    # 综合可信度
    confidence = 40 + min(30, abs(price_corr) * 30) + min(30, accuracy * 0.3)
    return min(95, max(5, confidence))

def calculate_normal_confidence(recent, signal_changes, price_changes):
    """计算正常信号变化时的可信度"""
    try:
        # 计算线性回归
        slope, intercept, r_value, p_value, std_err = linregress(
            signal_changes, price_changes
        )
    except ValueError:  # 处理所有x值相同的情况（理论上不应发生）
        return calculate_uniform_signal_confidence(recent, price_changes)
    
    # 计算信号准确性
    correct_signals = 0
    total_signals = 0
    
    for i in range(1, len(recent)):
        signal = recent['Signal'].iloc[i]
        prev_signal = recent['Signal'].iloc[i-1]
        price_change = price_changes.iloc[i]
        
        # 只考虑有变化的信号
        if signal != prev_signal:
            total_signals += 1
            if (signal > prev_signal and price_change > 0) or (signal < prev_signal and price_change < 0):
                correct_signals += 1
    
    # 计算准确率（避免除以零）
    accuracy = (correct_signals / total_signals * 100) if total_signals > 0 else 50
    
    # 综合可信度评分
    confidence = min(100, max(0, (abs(r_value) * 70 + accuracy * 0.3)))
    
    # 考虑趋势强度因子
    price_volatility = price_changes.abs().mean()
    if price_volatility < 0.5:  # 低波动市场
        confidence *= 0.9
    elif price_volatility > 2.0:  # 高波动市场
        confidence = min(100, confidence * 1.05)
    
    return round(confidence, 1)

def visualize_results_plotly(df, symbol, period):
    """使用Plotly可视化分析结果 - 交互式图表"""
    # 创建子图
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=(f'{symbol} 价格走势与交易信号 (分析周期: {period}天)', 
                                       '动量指标对比', 
                                       '成交量与资金流向'),
                        row_heights=[0.5, 0.3, 0.2])
    
    # 价格走势
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], 
                            mode='lines', name='收盘价',
                            line=dict(color='#1f77b4', width=2)),
                 row=1, col=1)
    
    # 标记交易信号
    buy_signals = df[df['Signal'] > 0]
    strong_buy = df[df['Signal'] == 2]
    sell_signals = df[df['Signal'] < 0]
    strong_sell = df[df['Signal'] == -2]
    
    fig.add_trace(go.Scatter(x=strong_buy['date'], y=strong_buy['close'],
                            mode='markers', name='强烈买入',
                            marker=dict(symbol='triangle-up', size=10, color='darkgreen')),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=buy_signals[buy_signals['Signal'] == 1]['date'], 
                            y=buy_signals[buy_signals['Signal'] == 1]['close'],
                            mode='markers', name='买入',
                            marker=dict(symbol='triangle-up', size=8, color='limegreen')),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=sell_signals[sell_signals['Signal'] == -1]['date'], 
                            y=sell_signals[sell_signals['Signal'] == -1]['close'],
                            mode='markers', name='卖出',
                            marker=dict(symbol='triangle-down', size=8, color='salmon')),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=strong_sell['date'], y=strong_sell['close'],
                            mode='markers', name='强烈卖出',
                            marker=dict(symbol='triangle-down', size=10, color='darkred')),
                 row=1, col=1)
    
    # 指标对比
    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], 
                            mode='lines', name='RSI',
                            line=dict(color='#ff7f0e', width=1.5)),
                 row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df['date'], y=df['MFI'], 
                            mode='lines', name='MFI',
                            line=dict(color='#2ca02c', width=1.5)),
                 row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df['date'], y=df['Hybrid'], 
                            mode='lines', name='混合指标',
                            line=dict(color='#d62728', width=2.5)),
                 row=2, col=1)
    
    # 阈值线
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=50, line_color="grey", opacity=0.7, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    # 成交量
    colors = np.where(df['close'] >= df['close'].shift(1), 'green', 'red')
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], 
                         name='成交量', marker_color=colors, opacity=0.7),
                 row=3, col=1)
    
    # 资金流分析
    money_flow = df['volume'] * (df['close'] - df['open'])
    pos_flow = money_flow.copy()
    neg_flow = money_flow.copy()
    pos_flow[pos_flow < 0] = 0
    neg_flow[neg_flow > 0] = 0
    
    fig.add_trace(go.Scatter(x=df['date'], y=pos_flow.cumsum(), 
                            mode='lines', name='资金流入累积',
                            line=dict(color='green', width=2)),
                 row=3, col=1)
    
    fig.add_trace(go.Scatter(x=df['date'], y=neg_flow.cumsum(), 
                            mode='lines', name='资金流出累积',
                            line=dict(color='red', width=2)),
                 row=3, col=1)
    
    # 更新布局
    fig.update_layout(
        height=800,
        title_text=f'{symbol} 技术分析 - RSI-MFI混合指标',
        hovermode='x unified',
        showlegend=True
    )
    
    # 更新y轴范围
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="指标值", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="成交量/资金流", row=3, col=1)
    
    return fig

def determine_trend(df, window=5):
    """
    专业趋势判断函数
    返回: (趋势方向, 趋势强度, 趋势描述)
    """
    # 获取最近n天的收盘价
    recent = df['close'].tail(window)
    
    # 1. 简单价格比较
    simple_up = recent.iloc[-1] > recent.iloc[0]
    
    # 2. 移动平均趋势
    ma_5 = recent.rolling(3).mean()
    ma_trend = "上涨" if ma_5.iloc[-1] > ma_5.iloc[-2] else "下跌"
    
    # 3. 线性回归趋势
    x = np.arange(len(recent))
    slope, _, _, _, _ = linregress(x, recent)
    regression_trend = "上涨" if slope > 0 else "下跌"
    
    # 4. 趋势强度指标
    max_price = recent.max()
    min_price = recent.min()
    range_pct = (max_price - min_price) / min_price * 100
    
    # 5. 趋势一致性
    up_days = sum(recent.diff().dropna() > 0)
    down_days = sum(recent.diff().dropna() < 0)
    consistency = "上涨主导" if up_days > down_days else "下跌主导"
    
    # 6. 波动率分析
    volatility = "高波动" if range_pct > 3 else "低波动"
    
    # 综合判断趋势方向
    if simple_up and ma_trend == "上涨" and regression_trend == "上涨":
        trend_direction = "上涨"
    elif not simple_up and ma_trend == "下跌" and regression_trend == "下跌":
        trend_direction = "下跌"
    else:
        trend_direction = "震荡"
    
    # 判断趋势强度
    if trend_direction == "上涨":
        if range_pct > 3:  # 显著波动
            if up_days >= window * 0.6:  # 多数天数上涨
                trend_strength = "强势"
            else:
                trend_strength = "震荡"
        else:
            trend_strength = "微弱"
    elif trend_direction == "下跌":
        if range_pct > 3:
            if down_days >= window * 0.6:
                trend_strength = "强势"
            else:
                trend_strength = "震荡"
        else:
            trend_strength = "微弱"
    else:
        trend_strength = "无方向"
    
    # 生成趋势描述
    if trend_strength == "强势" and trend_direction == "上涨":
        trend_desc = "强劲上涨趋势"
    elif trend_strength == "震荡" and trend_direction == "上涨":
        trend_desc = "震荡上行"
    elif trend_strength == "微弱" and trend_direction == "上涨":
        trend_desc = "微幅上涨"
    elif trend_strength == "强势" and trend_direction == "下跌":
        trend_desc = "大幅下跌趋势"
    elif trend_strength == "震荡" and trend_direction == "下跌":
        trend_desc = "震荡下行"
    elif trend_strength == "微弱" and trend_direction == "下跌":
        trend_desc = "微幅下跌"
    else:
        trend_desc = "横盘整理"
    
    # 添加波动率信息
    trend_desc += f" ({volatility})"
    
    return trend_direction, trend_strength, trend_desc

def generate_report(df, symbol, period):
    """生成专业分析报告 - 终极稳健版"""
    latest = df.iloc[-1]
    
    # 计算指标可信度 - 三重保护
    try:
        confidence = calculate_confidence(df)
    except Exception as e:
        st.warning(f"可信度计算遇到意外错误，使用保守估计: {str(e)}")
        confidence = 50.0
    
    # 专业趋势分析
    try:
        # 获取不同时间框架的趋势分析
        trend_results = []
        for days in [5, 10, 20]:
            if len(df) >= days:
                result = determine_trend(df.tail(days), window=days)
                trend_results.append(result)
            else:
                trend_results.append(("未知", "未知", f"{days}日数据不足"))
        
        trend_5d_dir, trend_5d_str, trend_5d_desc = trend_results[0]
        trend_10d_dir, trend_10d_str, trend_10d_desc = trend_results[1]
        trend_20d_dir, trend_20d_str, trend_20d_desc = trend_results[2]
        
        # 趋势一致性分析
        if all(r[0] == trend_5d_dir for r in trend_results if r[0] != "未知"):
            trend_consistency = "多周期趋势一致"
            trend_emoji = "✅"
        elif trend_5d_dir != trend_20d_dir and trend_20d_dir != "未知":
            trend_consistency = "短期与长期趋势背离"
            trend_emoji = "⚠️"
        else:
            trend_consistency = "趋势分化"
            trend_emoji = "➖"
    except Exception as e:
        st.error(f"趋势分析错误: {str(e)}")
        trend_5d_desc = "分析失败"
        trend_10d_desc = "分析失败"
        trend_20d_desc = "分析失败"
        trend_consistency = "分析失败"
        trend_emoji = "❌"

    # 信号分析
    if latest['Signal'] == 2:
        signal_analysis = "强烈买入信号：价格强劲且资金持续流入"
        signal_emoji = "🚀"
    elif latest['Signal'] == 1:
        signal_analysis = "买入信号：价格和资金流表现积极"
        signal_emoji = "✅"
    elif latest['Signal'] == -1:
        signal_analysis = "卖出信号：价格疲软且资金流出"
        signal_emoji = "⚠️"
    elif latest['Signal'] == -2:
        signal_analysis = "强烈卖出信号：价格下跌加速且资金大幅流出"
        signal_emoji = "🔥"
    else:
        signal_analysis = "中性信号：市场处于盘整阶段"
        signal_emoji = "➖"
    
    # 背离检测 - 添加边界检查
    try:
        divergence = ""
        if len(df) > 10:
            if (df['close'].iloc[-1] > df['close'].iloc[-10] and 
                df['Hybrid'].iloc[-1] < df['Hybrid'].iloc[-10] and
                df['Hybrid'].iloc[-1] < 70):
                divergence = "顶背离警告：价格创新高但指标走弱，可能反转 ⚠️"
            
            if (df['close'].iloc[-1] < df['close'].iloc[-10] and 
                df['Hybrid'].iloc[-1] > df['Hybrid'].iloc[-10] and
                df['Hybrid'].iloc[-1] > 30):
                divergence = "底背离信号：价格创新低但指标走强，可能反弹 ⬆️"
    except IndexError:
        divergence = "背离检测失败（数据不足）"
    
    # 价格位置分析 - 添加边界检查
    try:
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_to_high = (recent_high - latest['close']) / recent_high * 100
        current_to_low = (latest['close'] - recent_low) / recent_low * 100
        
        if current_to_high < 2:
            position = "接近近期高点"
        elif current_to_low < 2:
            position = "接近近期低点"
        else:
            position = "中间区域"
    except:
        position = "未知"
        current_to_high = 0
        current_to_low = 0
    
    # 涨跌幅计算 - 添加边界检查
    try:
        change_1d = (latest['close']/df['close'].iloc[-2]-1)*100 if len(df) > 1 else 0
        change_5d = (latest['close']/df['close'].iloc[-5]-1)*100 if len(df) > 5 else 0
        change_20d = (latest['close']/df['close'].iloc[-20]-1)*100 if len(df) > 20 else 0
    except IndexError:
        change_1d = change_5d = change_20d = 0
    
    # 生成专业报告
    report = f"""
    ## {symbol} 技术分析报告
    **生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
    **数据范围**: {df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {df['date'].iloc[-1].strftime('%Y-%m-%d')}  
    **分析周期**: {period}日  
    **指标可信度**: {confidence:.1f}% ({'高可信度' if confidence > 70 else '中可信度' if confidence > 50 else '低可信度'})  
    
    ### 最新行情
    - **当前日期**: {latest['date'].strftime('%Y-%m-%d')}
    - **上一日收盘价**: {latest['close']:.2f}
    - **成交量**: {latest['volume']/10000:.2f}万手
    - **价格位置**: {position} (距高点: -{current_to_high:.2f}%, 距低点: +{current_to_low:.2f}%)
    - **涨跌幅(1日)**: {change_1d:.2f}%
    - **涨跌幅(5日)**: {change_5d:.2f}%
    - **涨跌幅(20日)**: {change_20d:.2f}%
    
    ### 技术指标
    - **RSI**: {latest['RSI']:.2f} ({'超买' if latest['RSI']>70 else '超卖' if latest['RSI']<30 else '中性'})
    - **MFI**: {latest['MFI']:.2f} ({'资金流入强劲' if latest['MFI']>70 else '资金流出明显' if latest['MFI']<30 else '资金流向中性'})
    - **混合指标**: {latest['Hybrid']:.2f} ({'强势区域' if latest['Hybrid']>70 else '弱势区域' if latest['Hybrid']<30 else '中性区域'})
    
    ### 专业趋势分析
    - **短期趋势 (5日)**: {trend_5d_desc}
    - **中期趋势 (10日)**: {trend_10d_desc}
    - **长期趋势 (20日)**: {trend_20d_desc}
    - **趋势一致性**: {trend_emoji} {trend_consistency}
    
    ### 交易信号
    {signal_emoji} **{signal_analysis}**
    
    ### 背离检测
    {divergence}
    
    ### 操作建议
    {generate_trading_recommendation(latest, trend_consistency, confidence)}
    """
    
    return report

def generate_trading_recommendation(latest, trend_consistency, confidence):
    """生成操作建议 - 增强版"""
    signal = latest['Signal']
    hybrid = latest['Hybrid']
    rsi = latest['RSI']
    
    # 考虑指标一致性
    indicator_agreement = "一致" if (signal > 0 and hybrid > 50 and rsi > 50) or \
                                   (signal < 0 and hybrid < 50 and rsi < 50) else "分歧"
    
    # 高可信度情况
    if confidence > 70:
        if signal == 2:
            return "建议：积极买入，高可信度强烈买入信号"
        elif signal == 1:
            if indicator_agreement == "一致":
                return "建议：分批建仓，高可信度买入信号且指标一致"
            else:
                return "建议：谨慎买入，高可信度但指标有分歧"
        elif signal == -1:
            return "建议：考虑减仓，高可信度卖出信号"
        elif signal == -2:
            return "建议：果断卖出，高可信度强烈卖出信号"
        else:
            if hybrid > 70:
                return "建议：持仓观望，指标高位但无明确卖出信号"
            elif hybrid < 30:
                return "建议：关注机会，指标低位但无明确买入信号"
    
    # 中可信度情况
    elif confidence > 50:
        if signal > 0:
            if trend_consistency == "多周期趋势一致":
                return "建议：轻仓参与，趋势一致但信号可信度中等"
            else:
                return "建议：保持观望，信号与趋势不一致"
        elif signal < 0:
            return "建议：减仓避险，中等可信度卖出信号"
        else:
            return "建议：保持观望，市场方向不明"
    
    # 低可信度情况
    else:
        if hybrid > 70:
            return "建议：不宜追高，指标高位但可信度低"
        elif hybrid < 30:
            return "建议：不宜杀跌，指标低位但可信度低"
        else:
            if rsi > 70 or hybrid > 70:
                return "建议：警惕风险，部分指标显示超买"
            elif rsi < 30 or hybrid < 30:
                return "建议：关注机会，部分指标显示超卖"
            else:
                return "建议：保持观望，市场方向不明"

def analyze_stock(symbol, period, rsi_weight, mfi_weight,
                 strong_buy_threshold, buy_threshold,
                 strong_sell_threshold, sell_threshold):
    """主分析函数"""
    start_time = time.time()  # 记录开始时间
    
    # 获取数据
    df = get_stock_data(symbol)
    
    # 计算指标 - 传入所有参数
    df = calculate_rsi(df, period)
    df = calculate_mfi(df, period)
    df = generate_hybrid_indicator(
        df, rsi_weight, mfi_weight,
        strong_buy_threshold, buy_threshold,
        strong_sell_threshold, sell_threshold
    )

    # 生成报告
    report = generate_report(df, symbol, period)  # 传入分析周期
    st.markdown(report)
    
    # 可视化
    fig = visualize_results_plotly(df, symbol, period)  # 传入分析周期
    st.plotly_chart(fig, use_container_width=True)
    
    # 性能统计
    elapsed = time.time() - start_time
    st.info(f"分析完成! 耗时: {elapsed:.2f}秒")

def main():
    """主界面"""
    st.set_page_config(
        page_title="股票技术分析系统",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 股票技术分析系统 - RSI-MFI混合指标")
    st.markdown("""
    ### 系统介绍
    本系统结合RSI动量指标和资金流量指标(MFI)，构建了一个混合技术分析指标，用于判断股票走势。
    - **RSI (相对强弱指数)**: 衡量价格变化速度和幅度
    - **MFI (资金流量指数)**: 结合价格和成交量，衡量资金流入流出
    - **混合指标**: 综合两者优势，提供更可靠的交易信号
    
    ### 重要说明
    **数据范围**：显示的是获取到的历史数据的时间范围  
    **分析周期**：用户设置的指标计算周期（RSI/MFI计算使用的天数）
    """)
    
    st.sidebar.header("分析参数设置")
    
    # 股票代码输入
    symbol = st.sidebar.text_input(
        "股票代码", 
        "sh600519", 
        key="symbol_input"
    ).strip()
    
    st.sidebar.markdown("示例: `sh600519` (贵州茅台), `sz300750` (宁德时代)")
    
    # 分析周期设置
    period = st.sidebar.slider(
        "指标计算周期", 
        5, 30, 14, 
        key="period_slider"
    )
    
    # 指标权重设置
    st.sidebar.subheader("指标权重设置")
    
    # RSI权重
    rsi_weight = st.sidebar.slider(
        "RSI权重", 
        0.0, 1.0, 0.6, 
        key="rsi_weight_slider"
    )
    
    # MFI权重
    mfi_weight = st.sidebar.slider(
        "MFI权重", 
        0.0, 1.0, 0.4, 
        key="mfi_weight_slider"
    )
    
    # 信号阈值设置
    st.sidebar.subheader("信号阈值设置")
    
    # 强烈买入阈值
    strong_buy_threshold = st.sidebar.slider(
        "强烈买入阈值", 
        70, 90, 75, 
        key="strong_buy_slider"
    )
    
    # 买入阈值
    buy_threshold = st.sidebar.slider(
        "买入阈值", 
        50, 80, 60, 
        key="buy_threshold_slider"
    )
    
    # 卖出阈值
    sell_threshold = st.sidebar.slider(
        "卖出阈值", 
        20, 50, 40, 
        key="sell_threshold_slider"
    )
    
    # 强烈卖出阈值
    strong_sell_threshold = st.sidebar.slider(
        "强烈卖出阈值", 
        10, 40, 25, 
        key="strong_sell_slider"
    )
    
    # 开始分析按钮
    if st.sidebar.button("开始分析", type="primary", key="analyze_button"):
        with st.spinner("分析中，请稍候..."):
            analyze_stock(
                symbol, 
                period, 
                rsi_weight, 
                mfi_weight,
                strong_buy_threshold, 
                buy_threshold,
                strong_sell_threshold, 
                sell_threshold
            )
    
    # 可信度说明
    st.sidebar.markdown("""
    ---
    ### 指标可信度说明
    - **>70%**: 高可信度 - 信号与近期价格变动高度相关
    - **50-70%**: 中可信度 - 信号与价格变动有一定相关性
    - **<50%**: 低可信度 - 信号与价格变动相关性弱，需谨慎参考
    """)
    
    # 免责声明
    st.sidebar.markdown("""
    ---
    ### 免责声明
    本分析结果仅供参考，不构成任何投资建议。股市有风险，投资需谨慎。
    """)

if __name__ == "__main__":
    main()