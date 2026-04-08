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
    kc_limit = 1.0 - 0.003 * fck
    kc = max(0.85, kc_limit) if kc_limit < 0.85 else kc_limit # 1.0-0.003fck
    return fcc, ecc, r, Ec, kc, fcd, fsyd

def sigma_concrete(eps, fcc, ecc, r, kc):
    if eps <= 0: return 0.0 
    x = eps / ecc
    return kc * (fcc * x * r) / (r - 1 + x**r)

def sigma_steel(eps, fsyd, Es):
    sigma = Es * eps
    if sigma > fsyd: return fsyd
    elif sigma < -fsyd: return -fsyd
    return sigma

# ==========================================
# 2. ファイバー断面の生成
# ==========================================
def generate_fibers(D, t, num_layers=100):
    fibers = []
    R_outer = D / 2.0
    R_inner = R_outer - t
    dy = D / num_layers
    y_coords = np.linspace(-R_outer + dy/2, R_outer - dy/2, num_layers)
    for y in y_coords:
        if abs(y) >= R_outer: continue
        width_outer = 2 * np.sqrt(R_outer**2 - y**2)
        if abs(y) < R_inner:
            width_inner = 2 * np.sqrt(R_inner**2 - y**2)
            A_conc = width_inner * dy
            A_steel = (width_outer - width_inner) * dy
        else:
            A_conc = 0.0
            A_steel = width_outer * dy
        if A_steel > 0: fibers.append({'y': y, 'A': A_steel, 'mat': 'steel'})
        if A_conc > 0: fibers.append({'y': y, 'A': A_conc, 'mat': 'concrete'})
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
        eps0_sol = brentq(calc_N_error, -0.1, 0.1)
    except ValueError:
        return None, None

    M_int = 0.0
    for f in fibers:
        eps_i = eps0_sol + phi * f['y']
        if f['mat'] == 'steel':
            sigma = sigma_steel(eps_i, fsyd, Es)
        else:
            sigma = sigma_concrete(eps_i, fcc, ecc, r, 1.0) # バグ修正済
        M_int += sigma * f['A'] * f['y'] * 1e-6
    return eps0_sol, M_int

def find_points_for_N(target_N_N, fibers, fsyd, fcc, ecc, r, D, t, Es):
    eps_syd = fsyd / Es
    y_45deg_tension = - (D/2) * math.cos(math.radians(45))
    y_conc_comp = (D/2) - t
    eps_ultimate_limit = 2.0 * ecc # ESの挙動に合わせる
    
    phi_max = 0.1 / D 
    phis = np.linspace(0, phi_max, 250)
    
    Y_pt, M_pt = None, None
    for phi in phis:
        eps0, M = analyze_section(phi, target_N_N, fibers, fsyd, fcc, ecc, r, Es)
        if eps0 is None: continue
        
        # 降伏点判定 (45度位置)
        eps_45deg = eps0 + phi * y_45deg_tension
        if Y_pt is None and abs(eps_45deg) >= eps_syd:
            Y_pt = (phi, M)
            
        # 終局点判定 (圧縮縁)
        eps_conc_comp = eps0 + phi * y_conc_comp
        if M_pt is None and (eps_conc_comp >= eps_ultimate_limit):
            M_pt = (phi, M)
            break
            
    # 見つからない場合のセーフティ
    if Y_pt is None: Y_pt = (0.0, 0.0)
    if M_pt is None: M_pt = (phis[-1], analyze_section(phis[-1], target_N_N, fibers, fsyd, fcc, ecc, r, Es)[1] or 0.0)
    return Y_pt, M_pt

def calc_m_phi_curve(target_N_N, fibers, fsyd, fcc, ecc, r, D, t, Es):
    phi_max = 0.1 / D
    phis = np.linspace(0, phi_max, 150)
    moments, valid_phis = [], []
    for phi in phis:
        eps0, M = analyze_section(phi, target_N_N, fibers, fsyd, fcc, ecc, r, Es)
        if eps0 is not None:
            moments.append(M)
            valid_phis.append(phi)
    return valid_phis, moments

# ==========================================
# 4. FLIPフォーマット生成
# ==========================================
def to_f10(val):
    if val == 0.0: return "       0.0"
    s = f"{val:10.2f}" if abs(val) >= 100 else f"{val:10.4f}"
    if len(s) > 10: s = f"{val:10.1e}"
    return s[:10].rjust(10)

def create_flip_cards(axf_list, results_comp, results_tens):
    all_results = results_comp[::-1] + results_tens[1:] 
    n_points = len(all_results)
    cards = f"c --- IAX = 2{n_points:02d} (非対称モデル {n_points}点) ---\n"
    cards += "c RNNY(N), RMMP(Mp)\n"
    for i in range(0, n_points, 4):
        row = ""
        for j in range(4):
            if i+j < n_points:
                row += to_f10(all_results[i+j][0]) + to_f10(all_results[i+j][3])
        cards += row + "\n"
    cards += "c RNMY(N), RMMY(My)\n"
    for i in range(0, n_points, 4):
        row = ""
        for j in range(4):
            if i+j < n_points:
                row += to_f10(all_results[i+j][0]) + to_f10(all_results[i+j][1])
        cards += row + "\n"
    phi_y_0, phi_m_0 = results_comp[0][2], results_comp[0][4]
    def safe_ratio(phi, phi_0):
        return max(phi / phi_0 if phi_0 > 0 else 1.0, 0.01)
    cards += "c AXF (N/Ny ratio)\n"
    cards += "".join([to_f10(x) for x in (axf_list + [0.0]*8)[:8]]) + "\n"
    cards += "c CyRFC / CyRFT / CpRFC / CpRFT ... (省略可)\n"
    return cards

# ==========================================
# 5. Streamlit UI
# ==========================================
st.set_page_config(page_title="CFT M-φ FLIP", layout="wide")
st.title(r"CFT構造 M-$\phi$特性 & FLIP入力データ自動生成")

st.sidebar.header("入力条件")
D = st.sidebar.number_input("鋼管外径 D (mm)", value=1498.0, step=1.0)
t = st.sidebar.number_input("鋼管肉厚 t (mm)", value=15.0, step=1.0)
fsy = st.sidebar.number_input("鋼材降伏強度特性値 fsy (N/mm2)", value=315.0, step=5.0)
fck = st.sidebar.number_input("コンクリート設計基準強度 fck (N/mm2)", value=18.0, step=1.0)
Es_in = st.sidebar.number_input("鋼材の弾性係数 Es (N/mm2)", value=205000.0)
Ec_in = st.sidebar.number_input("コンクリートの弾性係数 Ec (N/mm2)", value=22000.0)
gamma_s, gamma_c, gamma_b = 1.05, 1.30, 1.10
target_N_kN = st.sidebar.number_input("常時作用軸力 N (kN)", value=0.0, step=100.0)

if st.sidebar.button(r"解析実行"):
    with st.spinner("計算中..."):
        fcc, ecc, r, Ec, kc, fcd, fsyd = get_confined_concrete_props(fck, fsy, D, t, gamma_c, gamma_s, Ec_in)
        fibers = generate_fibers(D, t, num_layers=100)
        
        A_s = sum([f['A'] for f in fibers if f['mat'] == 'steel'])
        A_c = sum([f['A'] for f in fibers if f['mat'] == 'concrete'])
        Nyc_raw = (A_s * fsyd + kc * A_c * fcc) / 1000.0
        Nyt_raw = - (A_s * fsyd) / 1000.0
        
        axf_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        results_comp, results_tens = [], []
        
        for axf in axf_list:
            for n_raw, res_list in [(axf * Nyc_raw, results_comp), (axf * Nyt_raw, results_tens)]:
                y, m = find_points_for_N(n_raw * 1000, fibers, fsyd, fcc, ecc, r, D, t, Es_in)
                res_list.append([n_raw/gamma_b, y[1]/gamma_b, y[0], m[1]/gamma_b, m[0]])

        # グラフ描画
        target_N_raw = target_N_kN * gamma_b * 1000.0
        phis, m_raw = calc_m_phi_curve(target_N_raw, fibers, fsyd, fcc, ecc, r, D, t, Es_in)
        y_r, m_r = find_points_for_N(target_N_raw, fibers, fsyd, fcc, ecc, r, D, t, Es_in)
        
        # M-phiグラフ (fig1)
        fig1, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot([p*1000 for p in phis], [m/gamma_b for m in m_raw], 'k-', label=f'N={target_N_kN}kN')
        if y_r[1]>0: ax1.plot(y_r[0]*1000, y_r[1]/gamma_b, 'bo', label=f'My={y_r[1]/gamma_b:.0f}')
        if m_r[1]>0: ax1.plot(m_r[0]*1000, m_r[1]/gamma_b, 'ro', label=f'Mm={m_r[1]/gamma_b:.0f}')
        ax1.set_xlabel("Curvature (1/m)"); ax1.set_ylabel("Moment (kN・m)"); ax1.grid(True); ax1.legend()

        # N-M相関図 (fig2)
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        all_res = results_comp[::-1] + results_tens[1:]
        ax2.plot([r[1] for r in all_res], [r[0] for r in all_res], 'bo-', label='My (Yield)')
        ax2.plot([r[3] for r in all_res], [r[0] for r in all_res], 'ro-', label='Mm (Ultimate)')
        ax2.set_xlabel("Moment (kN・m)"); ax2.set_ylabel("Axial Force (kN)"); ax2.grid(True); ax2.legend()

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("M-φ 曲線 (1/m単位)")
            st.pyplot(fig1)
            st.subheader("N-M 相関図")
            st.pyplot(fig2)
            st.write(f"My: {y_r[1]/gamma_b:,.0f} kN・m | φy: {y_r[0]*1000:.5f} 1/m")
            st.write(f"Mm: {m_r[1]/gamma_b:,.0f} kN・m | φm: {m_r[0]*1000:.5f} 1/m")
        with col2:
            st.subheader("FLIP入力データ")
            st.text_area("Copy this", value=create_flip_cards(axf_list, results_comp, results_tens), height=800)
