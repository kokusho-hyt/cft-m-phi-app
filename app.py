import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from scipy.optimize import brentq
import math

# ==========================================
# セキュリティ設定
# ==========================================
password = st.text_input("パスワードを入力してください", type="password")
if password != "cft":
    st.warning("正しいパスワードを入力すると計算ツールが表示されます。")
    st.stop()

# ==========================================
# 1. 材料モデル関数
# ==========================================
def get_confined_concrete_props(fck, fsy, D, t, gamma_c, gamma_s, Ec):
    fcd = fck / gamma_c
    fsyd = fsy / gamma_s
    alpha = 1.0
    f1 = 2 * t * alpha * fsyd / (D - 2 * t)
    fcc = fcd * (2.254 * np.sqrt(1 + 7.94 * f1 / fck) - 2 * f1 / fck - 1.254)
    ecc = 0.002 * (1 + 5 * (fcc / fcd - 1))
    Esec = fcc / ecc
    r = Ec / (Ec - Esec)
    kc = max(0.85, 1.0 - 0.003 * fck)
    return fcc, ecc, r, Ec, kc, fcd, fsyd

def sigma_concrete(eps, fcc, ecc, r, kc_val):
    if eps <= 0: return 0.0 
    x = eps / ecc
    return kc_val * (fcc * x * r) / (r - 1 + x**r)

def sigma_steel(eps, fsyd, Es):
    eps_syd = fsyd / Es
    if abs(eps) <= eps_syd:
        return Es * eps
    else:
        sign = np.sign(eps)
        return sign * (fsyd + (abs(eps) - eps_syd) * (Es / 100.0))

# ==========================================
# 2. ファイバー断面の生成 (可視化用データ追加)
# ==========================================
def generate_fibers_polar(D, t, n_r_conc=15, n_r_steel=3, n_theta=72):
    fibers = []
    d_theta = 360.0 / n_theta
    r_in = (D - 2 * t) / 2.0
    r_out = D / 2.0
    
    # コンクリート
    dr_c = r_in / n_r_conc
    for i in range(n_r_conc):
        r_start = i * dr_c
        r_end = (i + 1) * dr_c
        r_mid = (r_start + r_end) / 2.0
        area = (r_end**2 - r_start**2) * math.pi / n_theta
        for j in range(n_theta):
            theta_deg = j * d_theta
            theta_rad = math.radians(theta_deg + d_theta/2.0)
            fibers.append({
                'x': r_mid * math.sin(theta_rad), 'y': r_mid * math.cos(theta_rad),
                'r_start': r_start, 'r_end': r_end, 'theta_start': theta_deg, 'theta_end': theta_deg + d_theta,
                'A': area, 'mat': 'concrete'
            })
    # 鋼管
    dr_s = t / n_r_steel
    for i in range(n_r_steel):
        r_start = r_in + i * dr_s
        r_end = r_in + (i + 1) * dr_s
        r_mid = (r_start + r_end) / 2.0
        area = (r_end**2 - r_start**2) * math.pi / n_theta
        for j in range(n_theta):
            theta_deg = j * d_theta
            theta_rad = math.radians(theta_deg + d_theta/2.0)
            fibers.append({
                'x': r_mid * math.sin(theta_rad), 'y': r_mid * math.cos(theta_rad),
                'r_start': r_start, 'r_end': r_end, 'theta_start': theta_deg, 'theta_end': theta_deg + d_theta,
                'A': area, 'mat': 'steel'
            })
    return fibers

# ==========================================
# 3. 解析・描画エンジン
# ==========================================
def analyze_section(phi, target_N, fibers, fsyd, fcc, ecc, r, Es):
    def calc_N_error(eps0):
        N_int = sum([ (sigma_steel(eps0 + phi * f['y'], fsyd, Es) if f['mat'] == 'steel' else sigma_concrete(eps0 + phi * f['y'], fcc, ecc, r, 1.0)) * f['A'] for f in fibers])
        return N_int - target_N
    try:
        eps0_sol = brentq(calc_N_error, -1.0, 1.0)
    except: return None, None
    M_int = sum([ (sigma_steel(eps0_sol + phi * f['y'], fsyd, Es) if f['mat'] == 'steel' else sigma_concrete(eps0_sol + phi * f['y'], fcc, ecc, r, 1.0)) * f['A'] * f['y'] * 1e-6 for f in fibers])
    return eps0_sol, M_int

def plot_section_state(ax, fibers, eps0, phi, fsyd, ecc, Es, title, D):
    ax.set_aspect('equal')
    eps_syd = fsyd / Es
    for f in fibers:
        eps_f = eps0 + phi * f['y']
        color = 'lightgray'
        if f['mat'] == 'steel':
            color = 'orange' if abs(eps_f) >= eps_syd else 'royalblue'
        else:
            if eps_f >= ecc: color = 'red'
            elif eps_f > 0: color = 'lightgreen'
            else: color = 'whitesmoke'
        
        wedge = Wedge((0, 0), f['r_end'], f['theta_start']+90, f['theta_end']+90, width=f['r_end']-f['r_start'], facecolor=color, edgecolor='none', alpha=0.8)
        ax.add_patch(wedge)
    
    # 45度位置のマーカー
    y_45 = (D/2) * math.cos(math.radians(45))
    ax.plot([y_45, -y_45], [-y_45, -y_45], 'ro', markersize=8, label='45° Yield Detect')
    ax.set_title(title); ax.set_xlim(-D/2-50, D/2+50); ax.set_ylim(-D/2-50, D/2+50); ax.axis('off')

def find_points_for_N(target_N_N, fibers, fsyd, fcc, ecc, r, D, t, Es):
    eps_syd = fsyd / Es
    y_45deg = - (D/2) * math.cos(math.radians(45))
    y_comp_edge = (D/2) - t
    phis = np.linspace(0, 0.15/D, 300)
    Y_pt, M_pt = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    found_Y, found_M = False, False
    for phi in phis:
        eps0, M = analyze_section(phi, target_N_N, fibers, fsyd, fcc, ecc, r, Es)
        if eps0 is None: continue
        if not found_Y and phi > 0 and abs(eps0 + phi * y_45deg) >= eps_syd:
            Y_pt = (phi, M, eps0); found_Y = True
        if not found_M and phi > 0 and (eps0 + phi * y_comp_edge >= ecc):
            M_pt = (phi, M, eps0); found_M = True; break
    return Y_pt, M_pt

# ==========================================
# 4. Streamlit UI
# ==========================================
st.set_page_config(page_title="CFT M-φ ES同期版", layout="wide")
st.title("CFT構造 M-φ特性 & 断面応力状態可視化")

st.sidebar.header("入力条件")
D_ui = st.sidebar.number_input("外径 D (mm)", value=1498.0)
t_ui = st.sidebar.number_input("肉厚 t (mm)", value=15.0)
fsy_ui = st.sidebar.number_input("降伏強度 (N/mm2)", value=315.0)
fck_ui = st.sidebar.number_input("コンクリート強度 (N/mm2)", value=18.0)
Es_ui = st.sidebar.number_input("鋼材 Es (N/mm2)", value=205000.0)
Ec_ui = st.sidebar.number_input("コンクリート Ec (N/mm2)", value=22000.0)
target_N_kN = st.sidebar.number_input("常時軸力 N (kN)", value=0.0)

if st.sidebar.button("解析実行"):
    fcc, ecc, r, Ec, kc, fcd, fsyd = get_confined_concrete_props(fck_ui, fsy_ui, D_ui, t_ui, 1.3, 1.05, Ec_ui)
    fibers = generate_fibers_polar(D_ui, t_ui, n_r_conc=15, n_r_steel=3, n_theta=72)
    n_tar = target_N_kN * 1000.0
    y_r, m_r = find_points_for_N(n_tar, fibers, fsyd, fcc, ecc, r, D_ui, t_ui, Es_ui)

    # 断面図描画
    st.subheader(f"断面応力分布イメージ (軸力 N = {target_N_kN} kN)")
    fig_sec, (ax_y, ax_m) = plt.subplots(1, 2, figsize=(10, 5))
    plot_section_state(ax_y, fibers, y_r[2], y_r[0], fsyd, ecc, Es_ui, f"Yield State (My={y_r[1]:,.0f})", D_ui)
    plot_section_state(ax_m, fibers, m_r[2], m_r[0], fsyd, ecc, Es_ui, f"Ultimate State (Mm={m_r[1]:,.0f})", D_ui)
    st.pyplot(fig_sec)

    # 特性表示
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**降伏点 (My)**: {y_r[1]:,.1f} kNm | **φy**: {y_r[0]*1000:.5f} 1/m")
        phis_plot = np.linspace(0, 0.1/D_ui, 100)
        m_curve = [analyze_section(p, n_tar, fibers, fsyd, fcc, ecc, r, Es_ui)[1] or 0.0 for p in phis_plot]
        fig_mphi, ax_mp = plt.subplots()
        ax_mp.plot([p*1000 for p in phis_plot], m_curve, 'k-'); ax_mp.plot(y_r[0]*1000, y_r[1], 'bo'); ax_mp.plot(m_r[0]*1000, m_r[1], 'ro')
        ax_mp.set_xlabel("Curvature (1/m)"); ax_mp.set_ylabel("Moment (kNm)"); ax_mp.grid(True); st.pyplot(fig_mphi)
    with c2:
        st.success(f"**終局点 (Mm)**: {m_r[1]:,.1f} kNm | **φm**: {m_r[0]*1000:.5f} 1/m")
        st.markdown("**凡例:**")
        st.write("🟦 鋼管(弾性) 🟧 鋼管(降伏) | 🟩 コンクリート(圧縮) 🟥 コンクリート(終局ひずみ到達)")
