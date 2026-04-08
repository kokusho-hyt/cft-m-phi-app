import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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

def sigma_concrete(eps, fcc, ecc, r, kc):
    if eps <= 0: return 0.0 
    x = eps / ecc
    return kc * (fcc * x * r) / (r - 1 + x**r)

def sigma_steel(eps, fsyd, Es):
    # ESに合わせ、降伏後に Es/100 の硬化勾配を考慮
    eps_syd = fsyd / Es
    if abs(eps) <= eps_syd:
        return Es * eps
    else:
        sign = np.sign(eps)
        # fsyd + 硬化分
        return sign * (fsyd + (abs(eps) - eps_syd) * (Es / 100.0))

# ==========================================
# 2. ファイバー断面の生成 (極座標分割)
# ==========================================
def generate_fibers_polar(D, t, n_r_conc=15, n_r_steel=4, n_theta=72):
    fibers = []
    d_theta = 2 * math.pi / n_theta
    
    # コンクリート部 (半径 0 ~ R_inner)
    r_in = (D - 2*t) / 2.0
    dr_c = r_in / n_r_conc
    for i in range(n_r_conc):
        r = (i + 0.5) * dr_c
        area = r * dr_c * d_theta
        for j in range(n_theta):
            theta = (j + 0.5) * d_theta
            fibers.append({'y': r * math.cos(theta), 'A': area, 'mat': 'concrete'})
            
    # 鋼管部 (半径 R_inner ~ R_outer)
    r_out = D / 2.0
    dr_s = t / n_r_steel
    for i in range(n_r_steel):
        r = r_in + (i + 0.5) * dr_s
        area = r * dr_s * d_theta
        for j in range(n_theta):
            theta = (j + 0.5) * d_theta
            fibers.append({'y': r * math.cos(theta), 'A': area, 'mat': 'steel'})
    return fibers

# ==========================================
# 3. 断面解析エンジン
# ==========================================
def analyze_section(phi, target_N, fibers, fsyd, fcc, ecc, r, Es):
    def calc_N_error(eps0):
        N_int = 0.0
        for f in fibers:
            eps_i = eps0 + phi * f['y']
            if f['mat'] == 'steel':
                N_int += sigma_steel(eps_i, fsyd, Es) * f['A']
            else:
                N_int += sigma_concrete(eps_i, fcc, ecc, r, 1.0) * f['A']
        return N_int - target_N

    try:
        eps0_sol = brentq(calc_N_error, -0.2, 0.2)
    except ValueError:
        return None, None

    M_int = 0.0
    for f in fibers:
        eps_i = eps0_sol + phi * f['y']
        sigma = sigma_steel(eps_i, fsyd, Es) if f['mat'] == 'steel' else sigma_concrete(eps_i, fcc, ecc, r, 1.0)
        M_int += sigma * f['A'] * f['y'] * 1e-6
    return eps0_sol, M_int

def find_points_for_N(target_N_N, fibers, fsyd, fcc, ecc, r, D, t, Es):
    eps_syd = fsyd / Es
    y_45deg = - (D/2) * math.cos(math.radians(45))
    y_comp_edge = (D/2) - t
    eps_ultimate_limit = 2.0 * ecc # ESの計算結果(0.033)に同期
    
    phi_max = 0.1 / D 
    phis = np.linspace(0, phi_max, 300)
    
    Y_pt, M_pt = None, None
    for phi in phis:
        eps0, M = analyze_section(phi, target_N_N, fibers, fsyd, fcc, ecc, r, Es)
        if eps0 is None: continue
        
        # 降伏点 (45度位置)
        if Y_pt is None and abs(eps0 + phi * y_45deg) >= eps_syd:
            Y_pt = (phi, M)
            
        # 終局点 (コンクリート縁ひずみが ecc*2.0)
        if M_pt is None and (eps0 + phi * y_comp_edge >= eps_ultimate_limit):
            M_pt = (phi, M)
            break
            
    if Y_pt is None: Y_pt = (0.0, 0.0)
    if M_pt is None: M_pt = (phis[-1], M)
    return Y_pt, M_pt

# ==========================================
# 5. Streamlit UI
# ==========================================
st.set_page_config(page_title="CFT M-φ ES同期版", layout="wide")
st.title(r"CFT構造 M-$\phi$特性 (Engineers' Studio 同期モデル)")

st.sidebar.header("入力条件")
D = st.sidebar.number_input("鋼管外径 D (mm)", value=1498.0)
t = st.sidebar.number_input("鋼管肉厚 t (mm)", value=16.0) # ESレポートPage1のt=16に同期
fsy = st.sidebar.number_input("鋼材降伏強度特性値 fsy (N/mm2)", value=315.0)
fck = st.sidebar.number_input("コンクリート設計基準強度 fck (N/mm2)", value=18.0)
Es_in = st.sidebar.number_input("鋼材の弾性係数 Es (N/mm2)", value=205000.0)
Ec_in = st.sidebar.number_input("コンクリートの弾性係数 Ec (N/mm2)", value=22000.0)
gamma_b = st.sidebar.number_input("部材係数 γb", value=1.10) # ESの1.10に同期
target_N_kN = st.sidebar.number_input("常時作用軸力 N (kN)", value=0.0)

if st.sidebar.button(r"解析実行"):
    fcc, ecc, r, Ec, kc, fcd, fsyd = get_confined_concrete_props(fck, fsy, D, t, 1.3, 1.05, Ec_in)
    # 極座標メッシュ生成 (ESのN=72分割を参考)
    fibers = generate_fibers_polar(D, t, n_r_conc=20, n_r_steel=5, n_theta=72)
    
    A_s = sum([f['A'] for f in fibers if f['mat'] == 'steel'])
    A_c = sum([f['A'] for f in fibers if f['mat'] == 'concrete'])
    Nyc_raw = (A_s * fsyd + kc * A_c * fcc) / 1000.0
    
    axf_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results_comp, results_tens = [], []
    for axf in axf_list:
        for n_raw, res_list in [(axf * Nyc_raw, results_comp), (axf * - (A_s * fsyd / 1000.0), results_tens)]:
            y, m = find_points_for_N(n_raw * 1000, fibers, fsyd, fcc, ecc, r, D, t, Es_in)
            res_list.append([n_raw/gamma_b, y[1]/gamma_b, y[0], m[1]/gamma_b, m[0]])

    # グラフ描画
    target_N_raw = target_N_kN * gamma_b * 1000.0
    phis_raw, m_raw = [], []
    for p in np.linspace(0, 0.1/D, 150):
        e0, mm = analyze_section(p, target_N_raw, fibers, fsyd, fcc, ecc, r, Es_in)
        if e0 is not None: phis_raw.append(p); m_raw.append(mm)
    
    y_r, m_r = find_points_for_N(target_N_raw, fibers, fsyd, fcc, ecc, r, D, t, Es_in)
    
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot([p*1000 for p in phis_raw], [m/gamma_b for m in m_raw], 'k-')
    ax1.plot(y_r[0]*1000, y_r[1]/gamma_b, 'bo', label=f'My={y_r[1]/gamma_b:.0f}')
    ax1.plot(m_r[0]*1000, m_r[1]/gamma_b, 'ro', label=f'Mm={m_r[1]/gamma_b:.0f}')
    ax1.set_xlabel("Curvature (1/m)"); ax1.set_ylabel("Moment (kN・m)"); ax1.grid(True); ax1.legend()

    st.pyplot(fig1)
    st.write(f"My: {y_r[1]/gamma_b:,.0f} | φy: {y_r[0]*1000:.5f}")
    st.write(f"Mm: {m_r[1]/gamma_b:,.0f} | φm: {m_r[0]*1000:.5f}")
