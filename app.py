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
    # 設計強度の算定
    fcd = fck / gamma_c
    fsyd = fsy / gamma_s
    
    alpha = 1.0
    # 鋼管による拘束応力度（設計強度を使用）
    f1 = 2 * t * alpha * fsyd / (D - 2 * t)
    
    # 拘束コンクリートの最大圧縮強度
    fcc = fcd * (2.254 * np.sqrt(1 + 7.94 * f1 / fck) - 2 * f1 / fck - 1.254)
    
    # 最大圧縮強度時のコンクリート圧縮ひずみ
    ecc = 0.002 * (1 + 5 * (fcc / fcd - 1))
    
    # 割線弾性係数および弾性係数に関する比
    Esec = fcc / ecc
    r = Ec / (Ec - Esec)
    
    # コンクリート強度の低減係数 kc (純圧縮耐力用)
    kc = 1.0 - 0.003 * fck
    if kc > 0.85:
        kc = 0.85
        
    return fcc, ecc, r, Ec, kc, fcd, fsyd

def sigma_concrete(eps, fcc, ecc, r, kc):
    if eps <= 0: return 0.0 
    x = eps / ecc
    return kc * (fcc * x * r) / (r - 1 + x**r)

def sigma_steel(eps, fsyd, Es):
    # 鋼管の応力上限値に設計降伏強度(fsyd)を使用
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
    # 曲げ強度算出時は kc を強制的に 1.0 とする
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
            sigma = sigma_concrete(eps_i, fcc, ecc, r, 1.0) # バグ修正箇所
        M_int += sigma * f['A'] * f['y'] * 1e-6
    return eps0_sol, M_int

def find_points_for_N(target_N_N, fibers, fsyd, fcc, ecc, r, D, t, Es):
    eps_syd = fsyd / Es
    y_45deg_tension = - (D/2) * math.cos(math.radians(45))
    y_conc_comp = (D/2) - t
    y_steel_tens_edge = - (D/2)
    
    eps_su = max(5.0 * eps_syd, 0.01)
    
    phi_max = 0.05 / D 
    phis = np.linspace(0, phi_max, 150)
    
    Y_pt, M_pt = None, None
    for phi in phis:
        eps0, M = analyze_section(phi, target_N_N, fibers, fsyd, fcc, ecc, r, Es)
        if eps0 is None: continue
        
        eps_45deg = eps0 + phi * y_45deg_tension
        if Y_pt is None and abs(eps_45deg) >= eps_syd:
            Y_pt = (phi, M)
            
        eps_conc_comp = eps0 + phi * y_conc_comp
        eps_tens_edge = eps0 + phi * y_steel_tens_edge
        
        if M_pt is None and (eps_conc_comp >= ecc or abs(eps_tens_edge) >= eps_su):
            M_pt = (phi, M)
            
        if Y_pt and M_pt: break
        
    if Y_pt is None: Y_pt = (0.00001, 0.0)
    if M_pt is None: M_pt = (0.00001, 0.0)
    return Y_pt, M_pt

def calc_m_phi_curve(target_N_N, fibers, fsyd, fcc, ecc, r, D, t, Es):
    phi_max = 0.05 / D
    phis = np.linspace(0, phi_max, 100)
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
    
    cards += "c CyRFC (phi_y / phi_y0 for Comp)\n"
    cards += "".join([to_f10(safe_ratio(res[2], phi_y_0)) for res in results_comp] + [to_f10(0.0)]*(8-len(results_comp))) + "\n"
    
    cards += "c CyRFT (phi_y / phi_y0 for Tens)\n"
    cards += "".join([to_f10(safe_ratio(res[2], phi_y_0)) for res in results_tens] + [to_f10(0.0)]*(8-len(results_tens))) + "\n"
    
    cards += "c CpRFC (phi_p / phi_p0 for Comp)\n"
    cards += "".join([to_f10(safe_ratio(res[4], phi_m_0)) for res in results_comp] + [to_f10(0.0)]*(8-len(results_comp))) + "\n"
    
    cards += "c CpRFT (phi_p / phi_p0 for Tens)\n"
    cards += "".join([to_f10(safe_ratio(res[4], phi_m_0)) for res in results_tens] + [to_f10(0.0)]*(8-len(results_tens))) + "\n"
    
    return cards

# ==========================================
# 5. Streamlit UI
# ==========================================
st.set_page_config(page_title="CFT M-φ FLIP", layout="wide")
st.title(r"CFT構造 M-$\phi$特性 & FLIP入力データ自動生成")

st.sidebar.header("入力条件")
D = st.sidebar.number_input("鋼管外径 D (mm)", value=1500.0, step=10.0)
t = st.sidebar.number_input("鋼管肉厚 t (mm)", value=15.0, step=1.0)
fsy = st.sidebar.number_input("鋼材降伏強度特性値 fsy (N/mm2)", value=315.0, step=5.0)
fck = st.sidebar.number_input("コンクリート設計基準強度 fck (N/mm2)", value=18.0, step=1.0)
Es_input = st.sidebar.number_input("鋼材の弾性係数 Es (N/mm2)", value=205000.0, step=1000.0)
Ec_input = st.sidebar.number_input("コンクリートの弾性係数 Ec (N/mm2)", value=22000.0, step=1000.0)
gamma_s = st.sidebar.number_input("鋼材の材料係数 γs", value=1.05, step=0.01)
gamma_c = st.sidebar.number_input("コンクリートの材料係数 γc", value=1.30, step=0.01)
gamma_b = st.sidebar.number_input("部材係数 γb (耐力低減用)", value=1.30, step=0.01)
target_N_kN = st.sidebar.number_input("常時作用軸力 N (kN) [圧縮:+ / 引張:-]", value=3000.0, step=100.0)

if st.sidebar.button(r"全軸力でM-$\phi$解析実行"):
    with st.spinner("各軸力レベルで反復解析中... (数秒かかります)"):
        Es = Es_input
        Ec = Ec_input
        fcc, ecc, r, Ec_val, kc, fcd, fsyd = get_confined_concrete_props(fck, fsy, D, t, gamma_c, gamma_s, Ec)
        fibers = generate_fibers(D, t, num_layers=100)
        
        A_steel = sum([f['A'] for f in fibers if f['mat'] == 'steel'])
        A_conc = sum([f['A'] for f in fibers if f['mat'] == 'concrete'])
        
        # 内部計算用の純耐力（純圧縮耐力の算出には引き続き kc を考慮する）
        Nyc_kN_raw = (A_steel * fsyd + kc * A_conc * fcc) / 1000.0
        Nyt_kN_raw = - (A_steel * fsyd) / 1000.0
        
        # 出力用の純耐力（部材係数で低減）
        Nyc_kN = Nyc_kN_raw / gamma_b
        Nyt_kN = Nyt_kN_raw / gamma_b
        
        if target_N_kN > Nyc_kN or target_N_kN < Nyt_kN:
            st.error(f"入力された常時軸力 ({target_N_kN} kN) が、断面の純耐力範囲を超えています。計算可能な範囲に修正してください。")
            st.stop()
        
        axf_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        results_comp = []
        results_tens = []
        
        for axf in axf_list:
            N_kN_raw = axf * Nyc_kN_raw
            N_kN = N_kN_raw / gamma_b
            Y_pt_raw, M_pt_raw = find_points_for_N(N_kN_raw * 1000, fibers, fsyd, fcc, ecc, r, D, t, Es)
            results_comp.append([N_kN, Y_pt_raw[1] / gamma_b, Y_pt_raw[0], M_pt_raw[1] / gamma_b, M_pt_raw[0]])
            
        for axf in axf_list:
            N_kN_raw = axf * Nyt_kN_raw
            N_kN = N_kN_raw / gamma_b
            Y_pt_raw, M_pt_raw = find_points_for_N(N_kN_raw * 1000, fibers, fsyd, fcc, ecc, r, D, t, Es)
            results_tens.append([N_kN, Y_pt_raw[1] / gamma_b, Y_pt_raw[0], M_pt_raw[1] / gamma_b, M_pt_raw[0]])

        # ------------------------------------
        # グラフ1: 入力された常時軸力でのM-φ曲線
        # ------------------------------------
        target_N_N_raw = target_N_kN * gamma_b * 1000.0
        phis_target, M_target_raw = calc_m_phi_curve(target_N_N_raw, fibers, fsyd, fcc, ecc, r, D, t, Es)
        Y_target_raw, M_point_target_raw = find_points_for_N(target_N_N_raw, fibers, fsyd, fcc, ecc, r, D, t, Es)
        
        M_target = [m / gamma_b for m in M_target_raw]
        Y_target = (Y_target_raw[0], Y_target_raw[1] / gamma_b)
        M_point_target = (M_point_target_raw[0], M_point_target_raw[1] / gamma_b)
        
        EI1 = (Y_target[1] / (Y_target[0] * 1000.0)) if (Y_target[1] > 0.0 and Y_target[0] > 1e-9) else 0.0
        d_phi = (M_point_target[0] - Y_target[0]) * 1000.0
        EI2 = ((M_point_target[1] - Y_target[1]) / d_phi) if (M_point_target[1] > 0.0 and d_phi > 1e-5) else 0.0
        
        # グラフ描画用の曲率を 1/m に変換
        phis_target_m = [p * 1000.0 for p in phis_target]
        Y_target_phi_m = Y_target[0] * 1000.0
        M_point_target_phi_m = M_point_target[0] * 1000.0
        
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(phis_target_m, M_target, 'k-', label=f'N = {target_N_kN:.0f} kN')
        
        if Y_target[1] > 0.0:
            ax1.plot(Y_target_phi_m, Y_target[1], 'bo', markersize=8, label=f'Y Point ($M_y$={Y_target[1]:.0f})')
        if M_point_target[1] > 0.0:
            ax1.plot(M_point_target_phi_m, M_point_target[1], 'ro', markersize=8, label=f'M Point ($M_m$={M_point_target[1]:.0f})')
            
        ax1.set_xlabel(r"Curvature $\phi$ (1/m)")
        ax1.set_ylabel(r"Bending Moment $M$ (kN・m)")
        ax1.set_title(f"Moment-Curvature Curve at Constant Axial Force (N = {target_N_kN:.0f} kN)")
        ax1.grid(True)
        ax1.legend()

        # ------------------------------------
        # グラフ2: N-M相関図の描画
        # ------------------------------------
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        all_results = results_comp[::-1] + results_tens[1:]
        N_vals = [r[0] for r in all_results]
        My_vals = [r[1] for r in all_results]
        Mm_vals = [r[3] for r in all_results]
        
        ax2.plot(My_vals, N_vals, 'bo-', label='Yield Moment ($M_y$)')
        ax2.plot(Mm_vals, N_vals, 'ro-', label='Max Moment ($M_m/M_p$)')
        ax2.plot([-x for x in My_vals], N_vals, 'bo-')
        ax2.plot([-x for x in Mm_vals], N_vals, 'ro-')
        
        ax2.axhline(target_N_kN, color='gray', linestyle='--', label='Target N')
        
        ax2.set_xlabel(r"Bending Moment $M$ (kN・m)")
        ax2.set_ylabel(r"Axial Force $N$ (kN) [Comp:+ / Tens:-]")
        ax2.set_title("N-M Interaction Diagram for FLIP")
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.grid(True)
        ax2.legend()
        
        # FLIPカード生成
        flip_cards = create_flip_cards(axf_list, results_comp, results_tens)
        
        # UIレイアウト
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader(fr"M-$\phi$ 曲線 (常時軸力 N={target_N_kN:.0f}kN)")
            st.pyplot(fig1)
            
            # 剛性値の表示
            st.markdown(f"**第1勾配 ($EI_1$)**: {EI1:,.0f} kN・m²")
            st.markdown(f"**第2勾配 ($EI_2$)**: {EI2:,.0f} kN・m²")
            
            st.subheader("N-M 相関図")
            st.pyplot(fig2)
        with col2:
            st.subheader("FLIP 入力用カード (コピーして使用)")
            st.text_area("IHT=4 (10カラム固定) 用", value=flip_cards, height=850)
