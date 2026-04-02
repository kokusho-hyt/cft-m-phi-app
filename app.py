import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import math

# ==========================================
# セキュリティ設定
# ==========================================
password = st.text_input("パスワードを入力してください", type="password")
if password != "1234":  # 実運用に当っては任意のパスワードに変更する必要がある
    st.warning("正しいパスワードを入力すると計算ツールが表示されます。")
    st.stop()

# ==========================================
# 1. 材料モデル関数
# ==========================================
def get_confined_concrete_props(fck, fsy, D, t):
    alpha = 1.0
    f1 = 2 * t * alpha * fsy / (D - 2 * t)
    fcc = fck * (2.254 * np.sqrt(1 + 7.94 * f1 / fck) - 2 * f1 / fck - 1.254)
    ecc = 0.002 * (1 + 5 * (fcc / fck - 1))
    Esec = fcc / ecc
    Ec = 4700 * np.sqrt(fck)
    r = Ec / (Ec - Esec)
    
    # コンクリート強度の低減係数 kc
    kc = 1.0 - 0.003 * fck
    if kc > 0.85:
        kc = 0.85
        
    return fcc, ecc, r, Ec, kc

def sigma_concrete(eps, fcc, ecc, r, kc):
    if eps <= 0: return 0.0 
    x = eps / ecc
    return kc * (fcc * x * r) / (r - 1 + x**r)

def sigma_steel(eps, fsy, Es=200000.0):
    sigma = Es * eps
    if sigma > fsy: return fsy
    elif sigma < -fsy: return -fsy
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
        if A_conc >
