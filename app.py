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
    
    kc = 1.0 - 0.003 * fck
    if kc > 0.85:
        kc = 0.85
        
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
            sigma = sigma_concrete(eps_i, fcc, ecc, r, 1.0)
        M_int += sigma * f['A'] * f['y'] * 1e-6
    return eps0_sol, M_int

def find_points_for_N(target_N_N, fibers, fsyd, fcc, ecc, r, D, t, Es):
    eps_syd = fsyd / Es
    y_45deg_tension = - (D/2) * math.cos(math.radians(45))
    y_conc_comp = (D/2) - t
    y_steel_tens_edge = - (D/2)
    
    # 終局判定のひずみをESの挙動に合わせてピークの2.0倍に変更
    eps_ultimate_limit = 2.0 * ecc
    
    phi_max = 0.1 / D 
    phis = np.linspace(0, phi_max, 200)
    
    Y_pt, M_pt = None, None
    for phi in phis:
        eps0, M = analyze_section(phi, target_N_N, fibers, fsyd, fcc, ecc, r, Es)
        if eps0 is None: continue
        
        eps_45deg = eps0 + phi * y_45deg_tension
        if Y_pt is None and abs(eps_45deg) >= eps_syd:
            Y_pt = (phi, M)
            
        eps_conc_comp = eps0 + phi * y_conc_comp
        if M_pt is None and (eps_conc_comp >= eps_ultimate_limit):
            M_pt = (phi, M)
            
        if Y_pt and M_pt: break
        
    if Y_pt is None: Y_pt = (0.0, 0.0)
    if M_pt is None: M_pt = (0.0, 0.0)
    return Y_pt, M_pt

def calc_m_phi_curve(target_N_N, fibers, fsyd, fcc, ecc, r, D, t, Es):
    phi_max = 0.1 / D
    phis = np.linspace(0, phi_max, 150)
    moments = []
    valid_phis = []
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
    phi_y_0 = results_comp[0][2]
    phi_m_0 = results_comp[0][4]
    def safe_ratio(phi, phi_0):
        ratio = phi / phi_0 if phi_0 > 0 else 1.0
        return max(ratio, 0.01) 
    cards += "c AXF (N/Ny ratio)\n"
    axf_8 = (axf_list + [0.0]*8)[:8]
    cards += "".join([to_f10(x) for x in axf_8]) + "\n"
    cards += "c CyRFC (phi_y / phi_y0 Comp)\n"
    cards += "".join([to_f10(safe_ratio(res[2], phi_y_0)) for res in results_comp] + [to_f10(0.0)]*(8-len(results_comp))) + "\n"
    cards += "c CyRFT (phi_y / phi_y0 Tens)\n"
    cards += "".join([to_f10(safe_ratio(res[2], phi_y_0)) for res in results_tens] + [to_f10(0.0)]*(8-len(results_tens))) + "\n"
    cards += "c CpRFC (phi_p / phi_p0 Comp)\n"
    cards += "".join([to_f10(safe_ratio(res[4], phi_m_0)) for res in results_comp] + [to_f10(0.0)]*(8-len(results_comp))) + "\n"
    cards += "c CpRFT (phi_p / phi_p0 Tens)\n"
    cards += "".join([to_f10(safe_ratio(res[4], phi_m_0)) for res in results_tens] + [to_f10(0.0)]*(8-len(results_tens))) + "\n"
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
Es_input = st.sidebar.number_input("鋼材の弾性係数 Es (N/mm2)", value=205000.0, step=1000.0)
Ec_input = st.sidebar.number_input("コンクリートの弾性係数 Ec (N/mm2)", value=22000.0, step=1000.0)
gamma_s = st.sidebar.number_input("鋼材の材料係数 γs", value=1.05, step=0.01)
gamma_c = st.sidebar.number_input("コンクリートの材料係数 γc", value=1.30, step=0.01)
gamma_b = st.sidebar.number_input("部材係数 γb", value=1.10, step=0.01)
target_N_kN = st.sidebar.number_input("常時作用軸力 N (kN) [圧縮:+ / 引張:-]", value=0.0, step=100.0)

if st.sidebar.button(r"全軸力でM-$\phi$解析実行"):
    with st.spinner("反復解析中..."):
        Es, Ec = Es_input, Ec_input
        fcc, ecc, r, Ec_val, kc, fcd, fsyd = get_confined_concrete_props(fck, fsy, D, t, gamma_c, gamma_s, Ec)
        fibers = generate_fibers(D, t, num_layers=100)
        
        Nyc_kN_raw = (sum([f['A'] for f in fibers if f['mat'] == 'steel']) * fsyd + kc * sum([f['A'] for f in fibers if f['mat'] == 'concrete']) * fcc) / 1000.0
        Nyt_kN_raw = - (sum([f['A'] for f in fibers if f['mat'] == 'steel']) * fsyd) / 1000.0
        
        Nyc_kN, Nyt_kN = Nyc_kN_raw / gamma_b, Nyt_kN_raw / gamma_b
        axf_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        results_comp, results_tens = [], []
        for axf in axf_list:
            for n_raw, n_out, res_list in [(axf * Nyc_kN_raw, axf * Nyc_kN, results_comp), (axf * Nyt_kN_raw, axf * Nyt_kN, results_tens)]:
                y, m = find_points_for_N(n_raw * 1000, fibers, fsyd, fcc, ecc, r, D, t, Es)
                res_list.append([n_out, y[1]/gamma_b, y[0], m[1]/gamma_b, m[0]])

        # ターゲット軸力での描画
        target_N_raw = target_N_kN * gamma_b * 1000.0
        phis, m_raw = calc_m_phi_curve(target_N_raw, fibers, fsyd, fcc, ecc, r, D, t, Es)
        y_r, m_r = find_points_for_N(target_N_raw, fibers, fsyd, fcc, ecc, r, D, t, Es)
        
        m_plot = [m / gamma_b for m in m_raw]
        y_p, m_p = (y_r[0], y_r[1]/gamma_b), (m_r[0], m_r[1]/gamma_b)
        
        ei1 = (y_p[1]/(y_p[0]*1000)) if (y_p[1]>0 and y_p[0]>1e-9) else 0
        ei2 = ((m_p[1]-y_p[1])/((m_p[0]-y_p[0])*1000)) if (m_p[1]>0 and (m_p[0]-y_p[0])>1e-6) else 0
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([p*1000 for p in phis], m_plot, 'k-', label=f'N = {target_N_kN:.0f} kN')
        if y_p[1]>0: ax.plot(y_p[0]*1000, y_p[1], 'bo', label=f'My={y_p[1]:.0f}')
        if m_p[1]>0: ax.plot(m_p[0]*1000, m_p[1], 'ro', label=f'Mm={m_p[1]:.0f}')
        ax.set_xlabel(r"Curvature $\phi$ (1/m)"); ax.set_ylabel("Moment (kN・m)"); ax.grid(True); ax.legend()
        
        c1, c2 = st.columns([1.2, 1])
        with c1: st.pyplot(fig); st.markdown(f"**EI1**: {ei1:,.0f} | **EI2**: {ei2:,.0f} kN・m²")
        with c2: st.text_area("FLIP Data", value=create_flip_cards(axf_list, results_comp, results_tens), height=800)
