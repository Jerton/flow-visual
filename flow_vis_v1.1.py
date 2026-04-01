import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 页面配置 ---
st.set_page_config(page_title="QGP Dynamics Master", layout="wide")

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
    st.session_state.avg_pt = 1.2
    st.session_state.init_done = True

def apply_preset():
    p = st.session_state.preset
    if p == "中心碰撞 (Central)":
        vals = {1:0.0, 2:0.02, 3:0.05, 4:0.01, 5:0.01}
        st.session_state.avg_pt = 1.5
    elif p == "非中心碰撞 (Mid-Central)":
        vals = {1:0.02, 2:0.18, 3:0.07, 4:0.04, 5:0.02}
        st.session_state.avg_pt = 1.2
    elif p == "强涨落场景 (Fluctuation Dominated)":
        vals = {1:0.01, 2:0.05, 3:0.12, 4:0.03, 5:0.04}
        st.session_state.avg_pt = 1.0
    else: return
    for n in range(1, 6): st.session_state[f"v{n}"] = vals[n]

# --- 3. 侧边栏：物理参数控制 ---
with st.sidebar:
    st.header("🔬 动力学演化控制")
    st.selectbox("快速预设", ["手动调节", "中心碰撞 (Central)", "非中心碰撞 (Mid-Central)", "强涨落场景 (Fluctuation Dominated)"], 
                 key="preset", on_change=apply_preset)
    
    st.divider()
    st.subheader("🔥 热力学性质 (Thermodynamics)")
    avg_pt = st.slider("平均横动量 <pT> (GeV/c)", 0.5, 3.0, st.session_state.avg_pt, step=0.1, help="代表系统的径向压力梯度")
    eta_s = st.slider("剪切粘滞系数 (η/s)", 0.0, 0.5, 0.08, step=0.01)
    
    st.divider()
    st.subheader("🎨 初始各向异性 (Initial εn)")
    v_input = {}
    psi_input = {}
    for n in range(1, 6):
        col_v, col_p = st.columns([2, 1])
        with col_v:
            v_input[n] = st.slider(f"v{n}", 0.0, 0.4, st.session_state.v_base[n], key=f"v{n}")
        with col_p:
            psi_input[n] = st.number_input(f"Ψ{n}", 0.0, 6.28, st.session_state.psi[n], key=f"p{n}", label_visibility="collapsed")

    st.divider()
    st.subheader("🧪 关联分析 (Correlations)")
    sc_23_mode = st.radio("SC(2,3) 相关性模拟", ["负相关 (常见)", "无相关", "正相关"])
    num_p = st.sidebar.slider("采样粒子数", min_value=100, max_value=2000, value=800, step=100)

# --- 4. 核心物理计算 ---
# 1. 粘滞压低有效 vn
vn_eff = {n: v_input[n] * np.exp(-(n**2) * eta_s * 0.5) for n in range(1, 6)}

# 2. 考虑 SC(2,3) 的实际 v3 (仅用于 3D 渲染表现相关性)
if sc_23_mode == "负相关 (常见)":
    render_v3 = max(0, vn_eff[3] - (vn_eff[2] * 0.3))
elif sc_23_mode == "正相关":
    render_v3 = vn_eff[3] + (vn_eff[2] * 0.2)
else:
    render_v3 = vn_eff[3]

def distribution_func(phi_arr):
    res = np.ones_like(phi_arr)
    res += 2 * vn_eff[1] * np.cos(phi_arr)
    res += 2 * vn_eff[2] * np.cos(2 * (phi_arr - psi_input[2]))
    res += 2 * render_v3 * np.cos(3 * (phi_arr - psi_input[3]))
    res += 2 * vn_eff[4] * np.cos(4 * (phi_arr - psi_input[4]))
    res += 2 * vn_eff[5] * np.cos(5 * (phi_arr - psi_input[5]))
    return res

# --- 5. 数据生成：3D 粒子 & pT 谱 ---
# 生成 pT 分布 (使用 Gamma 分布模拟热化谱)
pt_samples = np.random.gamma(shape=avg_pt*3, scale=1/3, size=num_p)

# 方位角采样
phi_points = []
max_f = 1 + 2 * sum(vn_eff.values())
while len(phi_points) < num_p:
    t_phi = np.random.uniform(0, 2*np.pi)
    if np.random.uniform(0, max_f) < distribution_func(t_phi):
        phi_points.append(t_phi)

# 坐标转换：r 受到 pT 的缩放
costheta = np.random.uniform(-1, 1, num_p)
theta = np.arccos(costheta)
x = pt_samples * np.sin(theta) * np.cos(phi_points)
y = pt_samples * np.sin(theta) * np.sin(phi_points)
z = pt_samples * np.cos(theta)

# --- 6. 界面布局 ---
st.title("⚛️ QGP 动力学与关联分析实验室")
st.latex(r"\frac{dN}{d\phi} \propto 1 + 2 \sum v_n \cos n(\phi - \Psi_n) \quad \text{与} \quad \langle p_T \rangle \text{ 径向扩张联动}")

col_3d, col_tabs = st.columns([1, 1])

with col_3d:
    st.subheader("🚀 3D 动量空间膨胀 (Radius ∝ pT)")
    fig3d = go.Figure()
    # 粒子点，颜色由 pT 决定
    fig3d.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                marker=dict(size=3, color=pt_samples, colorscale='Hot', 
                                            colorbar=dict(title="pT (GeV/c)", x=0), opacity=0.8)))
    
    # 包络面 (以平均 pT 缩放)
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    r_surf = distribution_func(u) * avg_pt
    x_s, y_s, z_s = r_surf*np.sin(v)*np.cos(u), r_surf*np.sin(v)*np.sin(u), r_surf*np.cos(v)
    fig3d.add_trace(go.Mesh3d(x=x_s.flatten(), y=y_s.flatten(), z=z_s.flatten(), alphahull=0, opacity=0.1, color='cyan'))

    fig3d.update_layout(scene=dict(aspectmode='cube', xaxis=dict(range=[-5,5]), yaxis=dict(range=[-5,5]), zaxis=dict(range=[-5,5])),
                        margin=dict(l=0,r=0,b=0,t=0), height=700)
    st.plotly_chart(fig3d, use_container_width=True)

with col_tabs:
    t1, t2, t3 = st.tabs(["📊 流系数与谱", "📈 SC(n,m) 关联分析", "📝 物理诊断"])
    
    with t1:
        # 极坐标与 pT 谱
        sub_fig = make_subplots(rows=2, cols=1, subplot_titles=("方位角分布 dN/dφ", "横动量谱 dN/dpT"), vertical_spacing=0.15)
        
        phi_fine = np.linspace(0, 2*np.pi, 200)
        dist_fine = distribution_func(phi_fine)
        # 注意：Plotly subplots 处理 polar 需要特殊配置，这里简单用常规线图代替极坐标以适应 subplot
        sub_fig.add_trace(go.Scatter(x=phi_fine, y=dist_fine, fill='tozeroy', name="Azimuthal"), row=1, col=1)
        
        pt_axis = np.linspace(0, 6, 100)
        pt_yield = pt_axis * np.exp(-pt_axis / (avg_pt/2)) # 简化的热化谱
        sub_fig.add_trace(go.Scatter(x=pt_axis, y=pt_yield, fill='tozeroy', line_color='orange', name="pT Spectrum"), row=2, col=1)
        
        sub_fig.update_layout(height=600, showlegend=False, template="plotly_white")
        st.plotly_chart(sub_fig, use_container_width=True)

    with t2:
        st.subheader("Event-by-Event $v_2$ vs $v_3$ 相关性")
        # 模拟 200 个事件
        n_ev = 250
        ev_v2 = np.random.normal(vn_eff[2], 0.04, n_ev).clip(0)
        if sc_23_mode == "负相关 (常见)":
            ev_v3 = 0.2 - 0.5 * ev_v2 + np.random.normal(0, 0.02, n_ev)
        elif sc_23_mode == "正相关":
            ev_v3 = 0.02 + 0.4 * ev_v2 + np.random.normal(0, 0.02, n_ev)
        else:
            ev_v3 = np.random.normal(vn_eff[3], 0.03, n_ev)
        
        fig_sc = go.Figure(go.Scatter(x=ev_v2, y=ev_v3.clip(0), mode='markers', marker=dict(color='rgba(255,0,0,0.5)', size=8)))
        fig_sc.update_layout(xaxis_title="v2 (Elliptic)", yaxis_title="v3 (Triangular)", height=400, template="plotly_white")
        st.plotly_chart(fig_sc, use_container_width=True)
        st.write(f"**当前 Pearson 系数 ρ:** {np.corrcoef(ev_v2, ev_v3.clip(0))[0,1]:.3f}")

    with t3:
        st.markdown(f"""
        ### 核心观测指标报告
        - **径向流强度**: $\langle p_T \rangle = {avg_pt}$ GeV/c。数值越大，3D 散点云扩张半径越广，颜色越趋向红色。
        - **粘滞阻尼**: 当前 $\eta/s = {eta_s}$。高阶项 $v_5$ 的压低比例是 $v_2$ 的 **{np.exp(-(5**2-2**2)*eta_s*0.5):.2f}** 倍。
        - **Symmetric Cumulants**: 当前选择为 **{sc_23_mode}**。
        ---
        **物理贴士**: 
        在 LHC 能量下，通常测得 $SC(2,3) < 0$ 且 $SC(2,4) > 0$。这种关联性揭示了初始几何涨落的非平凡特性。
        """)