import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 页面配置 ---
st.set_page_config(page_title="QGP Flow Evolution Lab", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stSlider { margin-bottom: -10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 初始状态与预设逻辑 ---
if 'init_done' not in st.session_state:
    st.session_state.v_base = {1: 0.02, 2: 0.15, 3: 0.08, 4: 0.04, 5: 0.02}
    st.session_state.psi = {n: np.random.uniform(0, 2*np.pi/n) for n in range(1, 6)}
    st.session_state.init_done = True

def apply_preset():
    p = st.session_state.preset
    if p == "中心碰撞 (Central)":
        vals = {1:0.0, 2:0.02, 3:0.05, 4:0.01, 5:0.01}
    elif p == "非中心碰撞 (Mid-Central)":
        vals = {1:0.02, 2:0.18, 3:0.07, 4:0.04, 5:0.02}
    elif p == "强涨落场景 (Fluctuation Dominated)":
        vals = {1:0.01, 2:0.05, 3:0.12, 4:0.03, 5:0.04}
    else: return
    for n in range(1, 6): st.session_state[f"v{n}"] = vals[n]

# --- 3. 侧边栏：物理参数控制 ---
with st.sidebar:
    st.header("🔬 物理演化实验室")
    st.selectbox("快速预设", ["手动调节", "中心碰撞 (Central)", "非中心碰撞 (Mid-Central)", "强涨落场景 (Fluctuation Dominated)"], 
                 key="preset", on_change=apply_preset)
    
    st.divider()
    st.subheader("💧 介质性质 (Medium Properties)")
    # 模拟粘滞压低因子: exp(-n^2 * eta_s)
    eta_s = st.slider("剪切粘滞系数 (η/s)", 0.0, 0.5, 0.08, step=0.01, help="粘滞会显著压低高阶谐波 (v3, v4, v5)")
    
    st.divider()
    st.subheader("🎨 初始几何强度 (Initial εn)")
    v_input = {}
    psi_input = {}
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    
    for n in range(1, 6):
        col_v, col_p = st.columns([2, 1])
        with col_v:
            v_input[n] = st.slider(f"v{n} (阶数 {n})", 0.0, 0.4, st.session_state.v_base[n], key=f"v{n}")
        with col_p:
            psi_input[n] = st.number_input(f"Ψ{n}", 0.0, 6.28, st.session_state.psi[n], key=f"p{n}", label_visibility="collapsed")

    st.divider()
    num_p = st.select_slider("采样粒子数", options=[200, 500, 1000, 2000], value=500)

# --- 4. 核心物理计算 ---
# 计算经过粘滞压低后的有效 vn
def get_viscous_vn(n, v_in, eta):
    # 物理近似: 越高阶压低越快 (n^2 依赖)
    damping = np.exp(- (n**2) * eta * 0.5)
    return v_in * damping

vn_eff = {n: get_viscous_vn(n, v_input[n], eta_s) for n in range(1, 6)}

# 定义全阶分布函数
def distribution_func(phi_arr):
    res = np.ones_like(phi_arr)
    for n in range(1, 6):
        res += 2 * vn_eff[n] * np.cos(n * (phi_arr - psi_input[n]))
    return res

# --- 5. 数据生成：3D 粒子 & 分布 ---
# 采样粒子 (接受-拒绝采样)
phi_points = []
max_f = 1 + 2 * sum(vn_eff.values())
while len(phi_points) < num_p:
    t_phi = np.random.uniform(0, 2*np.pi)
    t_y = np.random.uniform(0, max_f)
    if t_y < distribution_func(t_phi):
        phi_points.append(t_phi)

costheta = np.random.uniform(-1, 1, num_p)
theta = np.arccos(costheta)
x = np.sin(theta) * np.cos(phi_points)
y = np.sin(theta) * np.sin(phi_points)
z = np.cos(theta)

# --- 6. 界面布局与可视化 ---
st.title("🌊 高能重离子碰撞：各向异性流全阶模拟器")
st.latex(r"v_n^{obs} \approx v_n^{initial} \cdot \exp\left( -n^2 \cdot \alpha \cdot \frac{\eta}{s} \right)")

col_3d, col_ana = st.columns([3, 2])

with col_3d:
    st.subheader("🪐 3D 粒子出射动量空间")
    fig3d = go.Figure()
    
    # 粒子点
    fig3d.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                marker=dict(size=3, color=phi_points, colorscale='Twilight', opacity=0.8)))
    
    # 形变包络面
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    r_surf = distribution_func(u)
    x_s = r_surf * np.sin(v) * np.cos(u)
    y_s = r_surf * np.sin(v) * np.sin(u)
    z_s = r_surf * np.cos(v)
    
    fig3d.add_trace(go.Mesh3d(x=x_s.flatten(), y=y_s.flatten(), z=z_s.flatten(), 
                             alphahull=0, opacity=0.1, color='gray'))

    fig3d.update_layout(
        scene=dict(
            xaxis=dict(title="x (Reaction Plane)", range=[-2,2], zerolinecolor="black", zerolinewidth=4),
            yaxis=dict(title="y (Out-of-Plane)", range=[-2,2], zerolinecolor="black", zerolinewidth=4),
            zaxis=dict(title="z (Beam)", range=[-2,2], zerolinecolor="black", zerolinewidth=4),
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.0))
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=700
    )
    st.plotly_chart(fig3d, use_container_width=True)

with col_ana:
    st.subheader("📊 观测量分析看板")
    
    # 1. 极坐标展示
    phi_fine = np.linspace(0, 2*np.pi, 200)
    fig_pol = go.Figure()
    fig_pol.add_trace(go.Scatterpolar(r=distribution_func(phi_fine), theta=np.degrees(phi_fine), 
                                     fill='toself', line=dict(color='indigo', width=3)))
    fig_pol.update_layout(polar=dict(radialaxis=dict(range=[0, 2.5])), height=350, margin=dict(t=30, b=30))
    st.plotly_chart(fig_pol, use_container_width=True)
    
    # 2. vn 功率谱 (Power Spectrum)
    fig_bar = go.Figure()
    # 原始值 vs 粘滞压低后的值
    fig_bar.add_trace(go.Bar(name='初始 εn (几何)', x=list(range(1,6)), y=list(v_input.values()), marker_color='lightgray'))
    fig_bar.add_trace(go.Bar(name='末态 vn (观测)', x=list(range(1,6)), y=list(vn_eff.values()), marker_color='royalblue'))
    
    fig_bar.update_layout(
        title="各阶流系数功率谱 (n=1-5)",
        xaxis_title="谐波阶数 n", yaxis_title="幅值",
        barmode='group', height=350, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # 3. 物理结论实时反馈
    st.info(f"**物理诊断：** 当前 η/s = {eta_s}。注意观察高阶项（v4, v5）的压低程度显著高于低阶项（v2）。这是判定夸克胶子等离子体（QGP）接近“近乎完美流体”的关键判据。")