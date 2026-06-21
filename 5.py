import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import warnings

# 忽略警告以保持界面整洁
warnings.filterwarnings('ignore')

# ==========================================
# 1. 页面配置与标题
# ==========================================
st.set_page_config(page_title="企业财务风险预警系统", layout="wide")
st.title("📊 基于机器学习的上市公司财务风险预警系统")
st.markdown("""
> **项目说明：** 本系统基于公开数据集（Company Bankruptcy Prediction），利用随机森林算法对企业财务状况进行分析，预测破产风险，并通过可视化大屏展示关键指标。
""")

# ==========================================
# 2. 数据加载与处理模块 (使用缓存加速)
# ==========================================
@st.cache_data
def load_data():
    """
    加载并预处理数据。
    这里模拟从Kaggle下载的 'Company Bankruptcy Prediction' 数据集结构。
    为了演示方便，如果本地没有csv，我们生成一份模拟数据供测试。
    """
    try:
        # 尝试读取本地CSV文件（如果你有下载好的 data.csv）
        df = pd.read_csv('data.csv')
        st.success("✅ 成功加载本地数据文件 data.csv")
    except FileNotFoundError:
        st.warning("⚠️ 未找到本地 data.csv 文件，正在生成模拟数据进行演示...")
        # 生成模拟数据逻辑（确保代码在任何环境下都能跑通）
        np.random.seed(42)
        n_samples = 2000
        data = {
            'Working Capital': np.random.normal(0, 1, n_samples),
            'Total Assets': np.random.normal(0, 1, n_samples),
            'Retained Earnings': np.random.normal(0, 1, n_samples),
            'Cash Flow': np.random.normal(0, 1, n_samples),
            'Liabilities Ratio': np.random.normal(0, 1, n_samples),
            # 目标变量：0=健康，1=破产/高风险
            'Bankrupt?': np.concatenate([np.zeros(n_samples-200), np.ones(200)])
        }
        df = pd.DataFrame(data)

    return df

df = load_data()

# 简单的列名重命名以便显示
if 'Bankrupt?' in df.columns:
    target_col = 'Bankrupt?'
else:
    target_col = df.columns[-1] # 假设最后一列是标签

feature_cols = [c for c in df.columns if c != target_col]

# ==========================================
# 3. 侧边栏：控制面板
# ==========================================
st.sidebar.header("⚙️ 参数设置")
test_size = st.sidebar.slider("测试集比例", 0.1, 0.5, 0.2)
n_estimators = st.sidebar.slider("随机森林树数量", 50, 300, 100)

# ==========================================
# 4. 核心分析逻辑
# ==========================================
X = df[feature_cols]
y = df[target_col]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 标准化数据（提升模型效果）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练模型
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

# ==========================================
# 5. 主界面展示区域
# ==========================================

# --- 第一行：关键指标卡片 ---
col1, col2, col3 = st.columns(3)
col1.metric("模型准确率", f"{acc:.2%}")
col2.metric("样本总数", len(df))
col3.metric("高风险样本占比", f"{y.mean():.2%}")

# --- 第二行：特征重要性分析 ---
st.subheader("🔍 财务指标风险权重分析")
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig_bar = px.bar(importance, orientation='h', title="各财务指标对风险预测的贡献度", color=importance, color_continuous_scale='Reds')
st.plotly_chart(fig_bar, use_container_width=True)

# --- 第三行：交互式风险预测器 ---
st.divider()
st.subheader("🧪 实时风险预测模拟器")
st.markdown("请在下方输入企业的各项财务指标数值（建议范围 -3 到 3，基于标准化数据），系统将实时计算破产概率。")

input_cols = st.columns(len(feature_cols))
user_input = []
for i, col_name in enumerate(feature_cols):
    val = input_cols[i].number_input(col_name, value=0.0, step=0.1)
    user_input.append(val)

if st.button("开始预测"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    risk_level = "🔴 高风险 (可能破产)" if prediction == 1 else "🟢 低风险 (经营健康)"
    prob_safe = probability[0] if len(probability) > 1 else 1 - probability[0]
    prob_risk = probability[1] if len(probability) > 1 else probability[0]

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("预测结果", risk_level)
    with col_b:
        # 绘制仪表盘图
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "破产风险概率 (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}],
            }))
        st.plotly_chart(fig_gauge, use_container_width=True)

# --- 第四行：原始数据概览 ---
with st.expander("查看原始数据预览"):
    st.dataframe(df.head(10))

# ==========================================
# 6. 底部版权信息
# ==========================================
st.caption("Generated by AI Assistant | 基于 Streamlit & Scikit-Learn 构建")