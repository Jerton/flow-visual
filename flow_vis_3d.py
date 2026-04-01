import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- 1. 页面配置 ---
st.set_page_config(page_title="HEP 3D Flow Lab", layout="wide")

# --- 2. 参数控制 (侧边栏) ---
with st.sidebar:
    st.header("🎛️ 碰撞参数设置")
    v2 = st.slider("v2 (椭圆流强度)", 0.0, 0.4, 0.15, step=0.01)
    v3 = st.slider("v3 (三角流强度)", 0.0, 0.3, 0.05, step=0.01)
    psi2 = st.slider("Ψ2 (椭圆平面角)", 0.0, np.pi, 0.0)
    
    st.divider()
    num_particles = st.slider("粒子采样数量", 100, 2000, 800)
    show_flow_lines = st.checkbox("显示流线趋势", value=True)

# --- 3. 核心数学：3D 粒子生成逻辑 ---
# 我们在 (theta, phi) 球坐标系下分布粒子
# theta: 极角 (与束流方向 z 的夹角)
# phi: 方位角 (我们在探测器横截面看到的角)

def generate_3d_flow(n_points, v2, v3, p2):
    # 随机生成 theta (均匀分布在 cos(theta) 上以保证球面上均匀)
    costheta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    
    # 使用接受-拒绝采样生成符合各向异性分布的 phi
    phi_samples = []
    max_val = 1 + 2*v2 + 2*v3
    while len(phi_samples) < n_points:
        p_test = np.random.uniform(0, 2*np.pi)
        y_test = np.random.uniform(0, max_val)
        # 核心公式
        prob = 1 + 2*v2*np.cos(2*(p_test - p2)) + 2*v3*np.cos(3*p_test)
        if y_test < prob:
            phi_samples.append(p_test)
    
    phi = np.array(phi_samples)
    
    # 转换为笛卡尔坐标用于 3D 绘图 (设半径 r=1)
    r = 1.0
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return x, y, z, phi

x, y, z, p_vals = generate_3d_flow(num_particles, v2, v3, psi2)

# --- 4. 主界面布局 ---
st.title("🌌 高能重离子碰撞 3D 末态分布")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🪐 3D 粒子出射分布 (带坐标轴)")
    
    fig_3d = go.Figure()

    # 1. 绘制采样粒子 (Scatter3d)
    fig_3d.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=p_vals, 
            colorscale='Viridis',
            opacity=0.8
        ),
        name="Detected Particles"
    ))

    # 2. 绘制理想流体外壳 (Mesh3d)
    if show_flow_lines:
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        U, V = np.meshgrid(u, v)
        R = 1 + 2*v2*np.cos(2*(U - psi2)) + 2*v3*np.cos(3*U)
        X_surf = R * np.sin(V) * np.cos(U)
        Y_surf = R * np.sin(V) * np.sin(U)
        Z_surf = R * np.cos(V)
        
        fig_3d.add_trace(go.Mesh3d(
            x=X_surf.flatten(), y=Y_surf.flatten(), z=Z_surf.flatten(),
            alphahull=0,
            opacity=0.1,
            color='cyan',
            name="Flow Surface"
        ))

    # 3. 增强型坐标轴设计 (Modern Academic Style)
    axis_config = dict(
        showbackground=False,
        showgrid=True,
        gridcolor="rgb(230, 230, 230)",
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=3,
        showticklabels=True,
        range=[-2, 2]
    )

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(**axis_config, title="x (横向)"),
            yaxis=dict(**axis_config, title="y (垂直)"),
            zaxis=dict(**axis_config, title="z (束流方向)"),
            aspectmode='cube',
            # 初始视角设置：稍微倾斜以便看清 3D 结构
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=650,
        showlegend=False
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

with col2:
    st.subheader("📐 探测器截面分布 (dN/dφ)")
    # 保持原来的极坐标图作为参考，因为它最能量化 vn
    phi_fine = np.linspace(0, 2*np.pi, 200)
    dist = 1 + 2*v2*np.cos(2*(phi_fine - psi2)) + 2*v3*np.cos(3*phi_fine)
    
    fig_polar = go.Figure()
    fig_polar.add_trace(go.Scatterpolar(
        r=dist, theta=np.degrees(phi_fine),
        fill='toself', line=dict(color='firebrick', width=3)
    ))
    fig_polar.update_layout(polar=dict(radialaxis=dict(range=[0, 2])), height=400)
    st.plotly_chart(fig_polar, use_container_width=True)
    
    st.info("""
    **3D 可视化说明：**
    * **散点**：代表探测器记录到的单个粒子。你可以观察到在特定方向（如 $v_2$ 的短轴方向）粒子点更密集。
    * **半透明外壳**：代表理想的流体膨胀前沿。当 $v_2$ 增大时，球体会沿横向被拉伸。
    * **交互**：你可以用鼠标旋转 3D 视图，查看束流方向（Z轴）前后的对称性。
    """)