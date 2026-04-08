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
    # コンクリート強度の低減係数 kc (純軸圧縮用)
    kc = max(0.85, 1.0 - 0.003 * fck)
    return fcc, ecc, r, Ec, kc, fcd, fsyd

def sigma_concrete(eps, fcc, ecc, r, kc_val):
    if eps <= 0: return 0.0 
    x = eps / ecc
    return kc_val * (fcc * x * r) / (r - 1 + x**r)

def sigma_steel(eps, fsyd, Es):
    # バイリニア硬化 (Es/100)
    eps_syd = fsyd / Es
    if abs(eps) <= eps_syd:
        return Es * eps
    else:
        sign = np.sign(eps)
        return sign * (fsyd + (abs(eps) - eps_syd) * (Es / 100.0))

# ==========================================
# 2. ファイバー断面の生成 (極座標)
# ==========================================
def generate_fibers_polar(D, t, n_r_conc=20, n_r_steel=5, n_theta=72):
    fibers = []
    d_theta = 2 * math.pi / n_theta
    r_in = (D - 2 * t) / 2.0
    r_out = D / 2.0
    # コンクリート
    dr_c = r_in / n_r_conc
    for i in range(n_r_conc):
        r = (i + 0.5) * dr_c
        area = r * dr_c * d_theta
        for j in range(n_theta):
            theta = (j + 0.5) * d_theta
            fibers.append({'y': r * math.cos(theta), 'A': area, 'mat': 'concrete'})
    # 鋼管
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
    # 曲げ解析時は kc = 1.0 を適用
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
        eps0_sol = brentq(calc_N_error, -1.0, 1.0)
    except ValueError:
        return None, None

    M_int = 0.0
    for f in fibers:
        eps_i = eps0_sol + phi * f['y']
        sigma = sigma_steel(eps_i, fsyd, Es) if f['mat'] == 'steel' else sigma_concrete(eps_i, fcc, ecc, r, 1.0)
        M_int += sigma * f['A'] * f['y'] * 1e-6
    return eps0_sol, M_int

def find_points_for_N(target_N_N, fibers, fsyd, fcc, ecc, r, D, t, Es, is_limit=False):
    # 純圧縮/引張点 (M=0)
    if is_limit:
        return (0.0, 0.0), (0.0, 0.0)
    
    eps_syd = fsyd / Es
    y_45deg = - (D/2) * math.cos(math.radians(45))
    y_comp_edge = (D/2) - t
    
    phi_max = 0.12 / D 
    phis = np.linspace(0, phi_max, 250)
    
    Y_pt, M_pt = (0.0, 0.0), (0.0, 0.0)
    found_Y, found_M = False, False
    max_M_found, phi_at_max = 0.0, 0.0

    for phi in phis:
        eps0, M = analyze_section(phi, target_N_N, fibers, fsyd, fcc, ecc, r, Es)
        if eps0 is None: continue
        if M > max_M_found:
            max_M_found = M
            phi_at_max = phi
        
        if not found_Y and phi > 0 and abs(eps0 + phi * y_45deg) >= eps_syd:
            Y_pt = (phi, M); found_Y = True
        if not found_M and phi > 0 and (eps0 + phi * y_comp_edge >= ecc):
            M_pt = (phi, M); found_M = True; break
            
    if not found_M: M_pt = (phi_at_max, max_M_found)
    return Y_pt, M_pt

# ==========================================
# 4. FLIPフォーマット生成
# ==========================================
def to_f10(val):
    if val == 0.0: return "       0.0"
    s = f"{val:10.2f}" if abs(val) >= 100 else f"{val:10.4f}"
    return s[:10].rjust(10)

def create_flip_cards(axf_list, results_comp, results_tens):
    all_results = results_comp[::-1] + results_tens[1:]
    n_points = len(all_results)
    cards = f"c --- IAX = 2{n_points:02d} (非対称モデル {n_points}点) ---\n"
    cards += "c RNNY(N), RMMP(Mp)\n"
    for i in range(0, n_points, 4):
        row = "".join([to_f10(all_results[i+j][0]) + to_f10(all_results[i+j][3]) for j in range(4) if i+j < n_points])
        cards += row + "\n"
    cards += "c RNMY(N), RMMY(My)\n"
    for i in range(0, n_points, 4):
        row = "".join([to_f10(all_results[i+j][0]) + to_f10(all_results[i+j][1]) for j in range(4) if i+j < n_points])
        cards += row + "\n"
    cards += "c AXF (N/Ny ratio)\n"
    cards += "".join([to_f10(x) for x in (axf_list + [0.0]*8)[:8]]) + "\n"
    return cards

# ==========================================
# 5. UI & 解析実行
# ==========================================
st.set_page_config(page_title="CFT M-φ ES同期版", layout="wide")
st.title("CFT構造 M-φ特性 & N-M相関図 (修正版)")

st.sidebar.header("入力条件")
D_ui = st.sidebar.number_input("外径 D (mm)", value=1498.0)
t_ui = st.sidebar.number_input("肉厚 t (mm)", value=15.0)
fsy_ui = st.sidebar.number_input("鋼材降伏強度 (N/mm2)", value=315.0)
fck_ui = st.sidebar.number_input("コンクリート強度 (N/mm2)", value=18.0)
Es_ui = st.sidebar.number_input("鋼材 Es (N/mm2)", value=205000.0)
Ec_ui = st.sidebar.number_input("コンクリート Ec (N/mm2)", value=22000.0)
gamma_b_ui = st.sidebar.number_input("部材係数 γb", value=1.00)
target_N_kN = st.sidebar.number_input("常時軸力 N (kN)", value=0.0)

if st.sidebar.button("解析実行"):
    with st.spinner("解析中..."):
        fcc, ecc, r, Ec, kc, fcd, fsyd = get_confined_concrete_props(fck_ui, fsy_ui, D_ui, t_ui, 1.3, 1.05, Ec_ui)
        fibers = generate_fibers_polar(D_ui, t_ui)
        
        # 耐力上限の計算
        A_s = sum([f['A'] for f in fibers if f['mat'] == 'steel'])
        A_c = sum([f['A'] for f in fibers if f['mat'] == 'concrete'])
        Nyc_raw = (A_s * fsyd + kc * A_c * fcc) / 1000.0 # コンクリートのみkc適用
        Nyt_raw = - (A_s * fsyd) / 1000.0
        
        axf_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        results_comp, results_tens = [], []
        
        for axf in axf_list:
            for n_limit, res_list, sign in [(Nyc_raw, results_comp, 1), (Nyt_raw, results_tens, -1)]:
                n_t = axf * n_limit
                y, m = find_points_for_N(n_t * 1000, fibers, fsyd, fcc, ecc, r, D_ui, t_ui, Es_ui, is_limit=(axf>=1.0))
                # 水平キャップの追加 (axf=1.0の点に対して M=0 を追加)
                if axf >= 1.0:
                    res_list.append([n_limit/gamma_b_ui, 0.0, 0.0, 0.0, 0.0])
                else:
                    res_list.append([n_t/gamma_b_ui, y[1]/gamma_b_ui, y[0], m[1]/gamma_b_ui, m[0]])

        # ターゲット軸力
        n_tar = target_N_kN * gamma_b_ui * 1000.0
        phis, moments = [], []
        for p in np.linspace(0, 0.08/D_ui, 150):
            _, mm = analyze_section(p, n_tar, fibers, fsyd, fcc, ecc, r, Es_ui)
            if mm is not None: phis.append(p); moments.append(mm)
        y_r, m_r = find_points_for_N(n_tar, fibers, fsyd, fcc, ecc, r, D_ui, t_ui, Es_ui)
        
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("M-φ 曲線 (1/m)")
            fig1, ax1 = plt.subplots()
            ax1.plot([p*1000 for p in phis], [m/gamma_b_ui for m in moments], 'k-')
            if y_r[1]>0: ax1.plot(y_r[0]*1000, y_r[1]/gamma_b_ui, 'bo', label=f'My={y_r[1]/gamma_b_ui:.0f}')
            if m_r[1]>0: ax1.plot(m_r[0]*1000, m_r[1]/gamma_b_ui, 'ro', label=f'Mm={m_r[1]/gamma_b_ui:.0f}')
            ax1.set_xlabel("Curvature (1/m)"); ax1.set_ylabel("Moment (kN・m)"); ax1.grid(True); ax1.legend()
            st.pyplot(fig1)
            st.info(f"My: {y_r[1]/gamma_b_ui:,.1f} kN・m | φy: {y_r[0]*1000:.5f} / Mm: {m_r[1]/gamma_b_ui:,.1f} kN・m | φm: {m_r[0]*1000:.5f}")

            st.subheader("N-M 相関図")
            fig2, ax2 = plt.subplots()
            all_res = results_comp[::-1] + results_tens[1:]
            ax2.plot([r[1] for r in all_res], [r[0] for r in all_res], 'bo-', label='My')
            ax2.plot([r[3] for r in all_res], [r[0] for r in all_res], 'ro-', label='Mm')
            ax2.axhline(target_N_kN, color='gray', linestyle='--')
            ax2.set_xlabel("Moment (kN・m)"); ax2.set_ylabel("Axial (kN)"); ax2.grid(True); ax2.legend()
            st.pyplot(fig2)
        with c2:
            st.subheader("FLIP入力データ")
            st.text_area("FLIP Card", value=create_flip_cards(axf_list, results_comp, results_tens), height=800)
