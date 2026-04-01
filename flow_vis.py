import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 页面配置 ---
st.set_page_config(page_title="QGP Dynamics Master Pro", layout="wide")

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

# --- 3. 侧边栏控制 ---
with st.sidebar:
    st.header("🔬 动力学演化控制")
    st.selectbox("快速预设", ["手动调节", "中心碰撞 (Central)", "非中心碰撞 (Mid-Central)", "强涨落场景 (Fluctuation Dominated)"], 
                 key="preset", on_change=apply_preset)
    
    st.divider()
    st.subheader("🔥 热力学与粘滞")
    avg_pt = st.slider("平均横动量 <pT>", 0.5, 3.0, st.session_state.avg_pt, step=0.1)
    eta_s = st.slider("剪切粘滞系数 (η/s)", 0.0, 0.5, 0.08, step=0.01)
    
    st.divider()
    st.subheader("🎨 各阶流强度 (v{n})和相位Ψ{n}")
    v_input = {}
    psi_input = {}
    for n in range(1, 6):
        c1, c2 = st.columns([2, 1])
        with c1: v_input[n] = st.slider(f"v{n}", 0.0, 0.4, st.session_state.v_base[n], key=f"v{n}")
        with c2: psi_input[n] = st.number_input(f"Ψ{n}", 0.0, 6.28, st.session_state.psi[n], key=f"p{n}", label_visibility="collapsed")

    st.divider()
    st.subheader("🧪 统计关联")
    sc_23_mode = st.radio("SC(2,3) 模拟", ["负相关 (常见)", "无相关", "正相关"])
    num_p = st.select_slider("采样粒子数", options=[200, 500, 1000, 2000], value=1000)

# --- 4. 核心物理计算 ---
vn_eff = {n: v_input[n] * np.exp(-(n**2) * eta_s * 0.5) for n in range(1, 6)}

# 考虑 SC(2,3) 逻辑的渲染值
render_v3 = vn_eff[3] + (vn_eff[2] * 0.2) if sc_23_mode == "正相关" else max(0, vn_eff[3] - (vn_eff[2] * 0.3)) if sc_23_mode == "负相关 (常见)" else vn_eff[3]

def distribution_func(phi):
    res = np.ones_like(phi)
    for n, val in vn_eff.items():
        v_to_use = render_v3 if n == 3 else val
        res += 2 * v_to_use * np.cos(n * (phi - psi_input[n]))
    return res

# --- 5. 数据采样 ---
pt_samples = np.random.gamma(shape=avg_pt*3, scale=1/3, size=num_p)
phi_points = []
max_f = 1 + 2 * sum(vn_eff.values())
while len(phi_points) < num_p:
    t_phi = np.random.uniform(0, 2*np.pi)
    if np.random.uniform(0, max_f) < distribution_func(t_phi): phi_points.append(t_phi)

costheta = np.random.uniform(-1, 1, num_p)
theta = np.arccos(costheta)
x = pt_samples * np.sin(theta) * np.cos(phi_points)
y = pt_samples * np.sin(theta) * np.sin(phi_points)
z = pt_samples * np.cos(theta)

# --- 6. 界面显示 ---
st.title("⚛️ QGP 动力学全阶实验室 (Pro)")

col_3d, col_tabs = st.columns([1, 1])

with col_3d:
    st.subheader("🚀 3D 动量空间出射 (半径 ∝ pT)")
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                marker=dict(size=3, color=pt_samples, colorscale='Hot', 
                                            colorbar=dict(title="pT (GeV/c)", x=0), opacity=0.8)))
    
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    r_surf = distribution_func(u) * avg_pt
    x_s, y_s, z_s = r_surf*np.sin(v)*np.cos(u), r_surf*np.sin(v)*np.sin(u), r_surf*np.cos(v)
    fig3d.add_trace(go.Mesh3d(x=x_s.flatten(), y=y_s.flatten(), z=z_s.flatten(), alphahull=0, opacity=0.1, color='cyan'))

    fig3d.update_layout(scene=dict(aspectmode='cube', 
                                   xaxis=dict(title="x", range=[-5,5]), 
                                   yaxis=dict(title="y", range=[-5,5]), 
                                   zaxis=dict(title="z", range=[-5,5])),
                        margin=dict(l=0,r=0,b=0,t=0), height=700)
    st.plotly_chart(fig3d, use_container_width=True)

with col_tabs:
    t1, t2, t3 = st.tabs(["🎯 横截面图 & 谱分析", "📈 统计关联分析", "📜 物理诊疗"])
    
    with t1:
        # 极坐标图恢复！
        phi_fine = np.linspace(0, 2*np.pi, 360)
        dist_fine = distribution_func(phi_fine)
        
        st.write("**动量空间横截面 (Transverse Plane)**")
        fig_polar = go.Figure()
        fig_polar.add_trace(go.Scatterpolar(r=dist_fine, theta=np.degrees(phi_fine), 
                                         fill='toself', line=dict(color='firebrick', width=3)))
        fig_polar.update_layout(polar=dict(radialaxis=dict(range=[0, 2.5])), height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig_polar, use_container_width=True)
        
        # pT 谱
        st.write("**横动量产额谱 (pT Spectrum)**")
        pt_axis = np.linspace(0, 6, 100)
        pt_yield = pt_axis * np.exp(-pt_axis / (avg_pt/2))
        fig_pt = go.Figure(go.Scatter(x=pt_axis, y=pt_yield, fill='tozeroy', line_color='orange'))
        fig_pt.update_layout(xaxis_title="pT (GeV/c)", yaxis_title="Yield", height=250, margin=dict(t=20, b=20), template="plotly_white")
        st.plotly_chart(fig_pt, use_container_width=True)

    with t2:
        st.subheader("Event-by-Event v2-v3 关联")
        n_ev = 250
        ev_v2 = np.random.normal(vn_eff[2], 0.04, n_ev).clip(0)
        ev_v3 = (0.2 - 0.5 * ev_v2 + np.random.normal(0, 0.02, n_ev)) if sc_23_mode == "负相关 (常见)" else \
                (0.02 + 0.4 * ev_v2 + np.random.normal(0, 0.02, n_ev)) if sc_23_mode == "正相关" else \
                np.random.normal(vn_eff[3], 0.03, n_ev)
        
        fig_sc = go.Figure(go.Scatter(x=ev_v2, y=ev_v3.clip(0), mode='markers', marker=dict(color='rgba(255,0,0,0.5)', size=8)))
        fig_sc.update_layout(xaxis_title="v2", yaxis_title="v3", height=450, template="plotly_white")
        st.plotly_chart(fig_sc, use_container_width=True)
        st.metric("Pearson 相关系数 ρ", f"{np.corrcoef(ev_v2, ev_v3.clip(0))[0,1]:.3f}")

    with t3:
        st.info(f"**物理参数诊断：**\n- 径向膨胀强度 $\langle p_T \rangle$: {avg_pt} GeV/c\n- 有效粘滞压低: 高阶 $v_5$ 已被抑制至原始的 {np.exp(-(5**2-2**2)*eta_s*0.5)*100:.1f}%")
        st.markdown("""
        ### 为什么这个图很重要？
        1. **极坐标图**：直接展示了 $v_n$ 的几何叠加。你可以清晰地看到 $v_2$ 的花生形、$v_3$ 的三角形和 $v_4$ 的正方形是如何竞争的。
        2. **3D 视图**：补充了纵向信息。你会发现尽管横截面很花哨，但粒子在 $z$ 轴方向是均匀延展的（Bjorken Expansion 的简化表现）。
        3. **关联分析**：揭示了即使每一阶 $v_n$ 看起来很乱，它们之间依然遵循着流体动力学的内在逻辑。
        """)

        # --- 7. 页脚与免责声明 ---
st.divider() # 添加一条横线分割主内容和页脚

# 使用 Markdown 自定义样式
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9em;">
        <p>🚀 <b>Made by Yutong and Gemini</b></p>
        <p style="font-style: italic; font-size: 0.8em;">
            免责声明：本程序仅用于物理教学演示与现象学可视化。
            代码计算基于简化模型，开发者不保证物理数值的绝对精确性，
            科学研究请以正式的流体动力学模拟软件（如 VISH2+1 或 MUSIC）为准。
        </p>
    </div>
    """, unsafe_allow_html=True)