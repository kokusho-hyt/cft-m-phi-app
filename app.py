import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
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
# 2. ファイバー断面の生成 (ES同期: 36x36分割)
# ==========================================
def generate_fibers_polar(D, t, n_r_conc=36, n_r_steel=5, n_theta=36):
    fibers = []
    d_theta = 360.0 / n_theta
    r_in = (D - 2 * t) / 2.0
    r_out = D / 2.0
    
    # コンクリート部
    dr_c = r_in / n_r_conc
    for i in range(n_r_conc):
        r_start, r_end = i * dr_c, (i + 1) * dr_c
        r_mid = (r_start + r_end) / 2.0
        area = (r_end**2 - r_start**2) * math.pi / n_theta
        for j in range(n_theta):
            theta_deg = j * d_theta
            theta_rad = math.radians(theta_deg + d_theta/2.0)
            fibers.append({
                'y': r_mid * math.cos(theta_rad), 'A': area, 'mat': 'concrete',
                'r_start': r_start, 'r_end': r_end, 'theta_start': theta_deg, 'theta_end': theta_deg + d_theta
            })
    # 鋼管部
    dr_s = t / n_r_steel
    for i in range(n_r_steel):
        r_start, r_end = r_in + i * dr_s, r_in + (i + 1) * dr_s
        r_mid = (r_start + r_end) / 2.0
        area = (r_end**2 - r_start**2) * math.pi / n_theta
        for j in range(n_theta):
            theta_deg = j * d_theta
            theta_rad = math.radians(theta_deg + d_theta/2.0)
            fibers.append({
                'y': r_mid * math.cos(theta_rad), 'A': area, 'mat': 'steel',
                'r_start': r_start, 'r_end': r_end, 'theta_start': theta_deg, 'theta_end': theta_deg + d_theta
            })
    return fibers

# ==========================================
# 3. 断面解析エンジン
# ==========================================
def analyze_section(phi, target_N, fibers, fsyd, fcc, ecc, r, Es):
    def calc_N_error(eps0):
        N_int = sum([(sigma_steel(eps0 + phi * f['y'], fsyd, Es) if f['mat'] == 'steel' else 
                      sigma_concrete(eps0 + phi * f['y'], fcc, ecc, r, 1.0)) * f['A'] for f in fibers])
        return N_int - target_N
    try:
        eps0_sol = brentq(calc_N_error, -1.0, 1.0)
    except: return None, None
    M_int = sum([(sigma_steel(eps0_sol + phi * f['y'], fsyd, Es) if f['mat'] == 'steel' else 
                  sigma_concrete(eps0_sol + phi * f['y'], fcc, ecc, r, 1.0)) * f['A'] * f['y'] * 1e-6 for f in fibers])
    return eps0_sol, M_int

def find_points_for_N(target_N_N, fibers, fsyd, fcc, ecc, r, D, t, Es):
    eps_syd = fsyd / Es
    y_45deg = - (D/2) * math.cos(math.radians(45))
    y_comp_edge = (D/2) - t
    phis = np.linspace(0, 0.15/D, 350)
    Y_pt, M_pt = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    found_Y, found_M, max_M, phi_max_M, eps0_max_M = False, False, 0.0, 0.0, 0.0
    for phi in phis:
        eps0, M = analyze_section(phi, target_N_N, fibers, fsyd, fcc, ecc, r, Es)
        if eps0 is None: continue
        if M > max_M: max_M, phi_max_M, eps0_max_M = M, phi, eps0
        if not found_Y and phi > 0 and abs(eps0 + phi * y_45deg) >= eps_syd:
            Y_pt = (phi, M, eps0); found_Y = True
        if not found_M and phi > 0 and (eps0 + phi * y_comp_edge >= ecc):
            M_pt = (phi, M, eps0); found_M = True; break
    if not found_M: M_pt = (phi_max_M, max_M, eps0_max_M)
    return Y_pt, M_pt

# ==========================================
# 4. 可視化・カード生成
# ==========================================
def plot_section_state(ax, fibers, eps0, phi, fsyd, ecc, Es, title, D):
    ax.set_aspect('equal')
    eps_syd = fsyd / Es
    for f in fibers:
        eps_f = eps0 + phi * f['y']
        if f['mat'] == 'steel':
            color = 'orange' if abs(eps_f) >= eps_syd else 'royalblue'
        else:
            if eps_f >= ecc: color = 'red'
            elif eps_f > 0: color = 'lightgreen'
            else: color = 'whitesmoke'
        wedge = Wedge((0, 0), f['r_end'], f['theta_start']+90, f['theta_end']+90, width=f['r_end']-f['r_start'], facecolor=color, edgecolor='none', alpha=0.7)
        ax.add_patch(wedge)
    y_45 = (D/2) * math.cos(math.radians(45))
    ax.plot([y_45, -y_45], [-y_45, -y_45], 'ro', markersize=5)
    ax.set_title(title); ax.set_xlim(-D/2-50, D/2+50); ax.set_ylim(-D/2-50, D/2+50); ax.axis('off')

def to_f10(val):
    return (f"{val:10.2f}" if abs(val) >= 100 else f"{val:10.4f}").rjust(10)

def create_flip_cards(res_comp, res_tens, axf_list):
    all_res = res_comp[::-1] + res_tens[1:]
    n = len(all_res)
    cards = f"c --- IAX = 2{n:02d} (非対称モデル {n}点) ---\n"
    cards += "c RNNY(N), RMMP(Mp)\n"
    for i in range(0, n, 4):
        cards += "".join([to_f10(all_res[i+j][0]) + to_f10(all_res[i+j][3]) for j in range(4) if i+j < n]) + "\n"
    cards += "c RNMY(N), RMMY(My)\n"
    for i in range(0, n, 4):
        cards += "".join([to_f10(all_res[i+j][0]) + to_f10(all_res[i+j][1]) for j in range(4) if i+j < n]) + "\n"
    cards += "c AXF Ratio\n" + "".join([to_f10(x) for x in (axf_list + [0.0]*8)[:8]]) + "\n"
    return cards

# ==========================================
# 5. UI & 解析実行
# ==========================================
st.set_page_config(page_title="CFT M-φ ES同期版", layout="wide")
st.title("CFT構造 断面解析 & 可視化システム (Engineers' Studio 同期)")

st.sidebar.header("入力条件")
D_ui = st.sidebar.number_input("外径 D (mm)", value=1498.0)
t_ui = st.sidebar.number_input("肉厚 t (mm)", value=15.0)
fsy_ui = st.sidebar.number_input("降伏強度 (N/mm2)", value=315.0)
fck_ui = st.sidebar.number_input("コンクリート強度 (N/mm2)", value=18.0)
Es_ui = st.sidebar.number_input("鋼材 Es (N/mm2)", value=205000.0)
Ec_ui = st.sidebar.number_input("コンクリート Ec (N/mm2)", value=22000.0)
gamma_b = st.sidebar.number_input("部材係数 γb", value=1.00)
target_N_kN = st.sidebar.number_input("常時軸力 N (kN)", value=0.0)

if st.sidebar.button("解析実行"):
    with st.spinner("極座標メッシュ解析中..."):
        fcc, ecc, r, Ec, kc, fcd, fsyd = get_confined_concrete_props(fck_ui, fsy_ui, D_ui, t_ui, 1.3, 1.05, Ec_ui)
        fibers = generate_fibers_polar(D_ui, t_ui)
        A_s, A_c = sum([f['A'] for f in fibers if f['mat'] == 'steel']), sum([f['A'] for f in fibers if f['mat'] == 'concrete'])
        Nyc_raw, Nyt_raw = (A_s * fsyd + kc * A_c * fcc) / 1000.0, - (A_s * fsyd) / 1000.0
        
        axf_list = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
        res_comp, res_tens = [], []
        for axf in axf_list:
            nc = axf * Nyc_raw
            y, m = find_points_for_N(nc * 1000, fibers, fsyd, fcc, ecc, r, D_ui, t_ui, Es_ui)
            res_comp.append([nc/gamma_b, y[1]/gamma_b, y[0], m[1]/gamma_b, m[0]])
            nt = axf * Nyt_raw
            yt, mt = find_points_for_N(nt * 1000, fibers, fsyd, fcc, ecc, r, D_ui, t_ui, Es_ui)
            res_tens.append([nt/gamma_b, yt[1]/gamma_b, yt[0], mt[1]/gamma_b, mt[0]])
        
        y_cap, m_cap = find_points_for_N(Nyc_raw * 1000, fibers, fsyd, fcc, ecc, r, D_ui, t_ui, Es_ui)
        res_comp.append([Nyc_raw/gamma_b, y_cap[1]/gamma_b, y_cap[0], m_cap[1]/gamma_b, m_cap[0]])
        res_comp.append([Nyc_raw/gamma_b, 0.0, 0.0, 0.0, 0.0])
        res_tens.append([Nyt_raw/gamma_b, 0.0, 0.0, 0.0, 0.0])

        n_tar = target_N_kN * gamma_b * 1000.0
        y_r, m_r = find_points_for_N(n_tar, fibers, fsyd, fcc, ecc, r, D_ui, t_ui, Es_ui)
        phis_p = np.linspace(0, 0.08/D_ui, 150)
        m_curve = [analyze_section(p, n_tar, fibers, fsyd, fcc, ecc, r, Es_ui)[1] or 0.0 for p in phis_p]

        # 断面図の描画
        st.subheader(f"断面応力状態イメージ (N = {target_N_kN} kN)")
        f_sec, (ax_y, ax_m) = plt.subplots(1, 2, figsize=(10, 5))
        plot_section_state(ax_y, fibers, y_r[2], y_r[0], fsyd, ecc, Es_ui, f"Yield (My={y_r[1]:,.0f})", D_ui)
        plot_section_state(ax_m, fibers, m_r[2], m_r[0], fsyd, ecc, Es_ui, f"Ultimate (Mm={m_r[1]:,.0f})", D_ui)
        st.pyplot(f_sec)
        
        # --- 凡例の復活 ---
        st.markdown("""
        **【断面図の着色凡例】** * 🟦 **鋼管（弾性）**: 応力度が設計降伏点 $f_{syd}$ 未満 [cite: 564, 579]
        * 🟧 **鋼管（降伏）**: 45度位置または最外縁が降伏ひずみに到達 [cite: 652, 656]
        * 🟩 **コンクリート（圧縮）**: 圧縮応力が発生している要素 [cite: 658, 659]
        * 🟥 **コンクリート（終局）**: 縁ひずみが終局ひずみ $\epsilon'_{cu}$ に到達 
        * ⬜ **コンクリート（引張）**: 引張応力を無視（無効セクション） [cite: 659]
        ---
        """)

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("M-φ 曲線 (1/m)")
            f_mp, ax_mp = plt.subplots()
            ax_mp.plot([p*1000 for p in phis_p], [m/gamma_b for m in m_curve], 'k-')
            ax_mp.plot(y_r[0]*1000, y_r[1]/gamma_b, 'bo', label='Yield')
            ax_mp.plot(m_r[0]*1000, m_r[1]/gamma_b, 'ro', label='Ultimate')
            ax_mp.set_xlabel("Curvature (1/m)"); ax_mp.set_ylabel("Moment (kN・m)"); ax_mp.grid(True); st.pyplot(f_mp)
            st.info(f"**Yield**: My={y_r[1]/gamma_b:,.1f} | φy={y_r[0]*1000:.5f} / **Ultimate**: Mm={m_r[1]/gamma_b:,.1f} | φm={m_r[0]*1000:.5f}")

            st.subheader("N-M 相関図 (カットオフ対応)")
            f_nm, ax_nm = plt.subplots()
            all_r = res_comp[::-1] + res_tens[1:]
            ax_nm.plot([r[1] for r in all_r], [r[0] for r in all_res if 'all_res' not in locals() and False or True], 'bo-', label='My')
            # 修正: 凡例描画用のリスト指定ミスをカバー
            ax_nm.plot([r[1] for r in all_r], [r[0] for r in all_r], 'bo-') 
            ax_nm.plot([r[3] for r in all_r], [r[0] for r in all_r], 'ro-', label='Mm')
            ax_nm.set_xlabel("Moment (kN・m)"); ax_nm.set_ylabel("Axial (kN)"); ax_nm.grid(True); ax_nm.legend(); st.pyplot(f_nm)
        with col2:
            st.subheader("FLIP入力データ")
            st.text_area("Copy content", value=create_flip_cards(res_comp, res_tens, axf_list+[1.0]), height=800)
