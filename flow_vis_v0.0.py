import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- 页面配置 ---
st.set_page_config(page_title="Anisotropic Flow Visualizer", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stSlider { padding-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 标题与公式展示 ---
st.title("⚛️ 各向异性流观测量可视化")
st.latex(r"E \frac{d^3N}{d^3p} = \frac{1}{2\pi} \frac{d^2N}{p_T dp_T dy} \left( 1 + 2 \sum_{n=1}^{\infty} v_n \cos[n(\phi - \Psi_n)] \right)")

st.info("通过调节左侧参数，直观观察不同阶流系数 $v_n$ 及其相位 $\Psi_n$ 如何重塑粒子在动量空间的方位角分布。")

# --- 侧边栏：参数调节 ---
with st.sidebar:
    st.header("🎛️ 傅里叶参数调节")
    
    # 预设场景一键配置
    preset = st.selectbox("快速预设场景", ["手动调节", "纯椭圆流 (v2)", "三角流 (v3)", "混合涨落场景"])
    
    max_n = st.slider("最高阶数 (Max n)", 1, 6, 3)
    
    vn_values = {}
    psi_values = {}

    for n in range(1, max_n + 1):
        st.markdown(f"**第 {n} 阶谐波 (Harmonic n={n})**")
        
        # 根据预设赋初值
        default_v = 0.0
        if preset == "纯椭圆流 (v2)" and n == 2: default_v = 0.15
        elif preset == "三角流 (v3)" and n == 3: default_v = 0.12
        elif preset == "混合涨落场景": default_v = np.random.uniform(0.02, 0.1)
        
        vn_values[n] = st.slider(f"幅度 v{n}", 0.0, 0.4, default_v, step=0.01, key=f"v{n}")
        psi_values[n] = st.slider(f"相位 Ψ{n} (rad)", 0.0, float(2*np.pi/n), 0.0, step=0.1, key=f"psi{n}")
        st.divider()

# --- 核心计算 ---
phi = np.linspace(0, 2 * np.pi, 1000)
# 基础分布：1 (各项同性)
dist_total = np.ones_like(phi)
harmonics = {}

for n in range(1, max_n + 1):
    # 计算单项：2 * vn * cos(n * (phi - psi))
    h = 2 * vn_values[n] * np.cos(n * (phi - psi_values[n]))
    harmonics[n] = h
    dist_total += h

# --- 绘图区域 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 极坐标分布 (Polar Distribution)")
    fig_polar = go.Figure()
    
    # 绘制主形状
    fig_polar.add_trace(go.Scatterpolar(
        r=dist_total,
        theta=np.degrees(phi),
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.3)',
        line=dict(color='#636EFA', width=3),
        name='Total Yield'
    ))
    
    fig_polar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
        showlegend=False,
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig_polar, use_container_width=True)

with col2:
    st.subheader("📈 方位角波形 (Azimuthal Waveform)")
    fig_line = go.Figure()
    
    # 绘制总和线
    fig_line.add_trace(go.Scatter(x=phi, y=dist_total, name="Total Distribution",
                                 line=dict(color='black', width=4)))
    
    # 绘制各阶分量（可选展示）
    show_individual = st.checkbox("显示各阶独立贡献", value=True)
    if show_individual:
        for n, h in harmonics.items():
            if vn_values[n] > 0:
                fig_line.add_trace(go.Scatter(x=phi, y=1+h, name=f"n={n} contribution",
                                             line=dict(dash='dash', width=2)))

    fig_line.update_layout(
        xaxis_title="方位角 φ (rad)",
        yaxis_title="dN/dφ (Relative)",
        xaxis=dict(tickmode='array', tickvals=[0, np.pi, 2*np.pi], ticktext=['0', 'π', '2π']),
        height=500,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    st.plotly_chart(fig_line, use_container_width=True)

# --- 底部说明 ---
with st.expander("ℹ️ 物理含义快速说明"):
    st.markdown("""
    * **$v_1$ (Directed Flow):** 描述粒子分布的整体偏移。
    * **$v_2$ (Elliptic Flow):** 反映了非中心碰撞中初始重叠区域的“杏仁状”几何对称性。
    * **$v_3$ (Triangular Flow):** 起源于初始核子坐标的量子涨落，使碰撞区呈现不规则三角形。
    * **$\Psi_n$ (Event Plane):** 各阶对称平面的方位角，决定了形变的方向。
    """)