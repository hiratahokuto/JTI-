import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq
import time
import matplotlib
from io import BytesIO
import base64
import pandas as pd

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['IPAGothic', 'IPAPGothic', 'Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']

# ページ設定
st.set_page_config(
    page_title="JSI,JTIシミュレーター(PPKTP:792→1584)",
    page_icon="🔬",
    layout="wide"
)

# タイトル
st.title("JSI,JTIシミュレーター(PPKTP:792→1584)")

# サイドバーでパラメータ設定
st.sidebar.header("実験パラメータ")

# 使用方法（サイドバー最上部に配置）
with st.sidebar.expander("使用方法", expanded=True):
    st.markdown("""
    1. **パラメータ設定**
       - ポンプ光設定: 中心波長とスペクトル幅
       - 結晶設定: 温度、長さ、分極反転周期
       - 波長範囲設定: 中心波長と幅、解像度
    2. **シミュレーション実行**
       - 「シミュレーション実行」ボタンをクリック
    3. **結果の確認**
       - JSI/JTI: 光子ペアの波長/時間相関
       - ポンプ光関数/位相整合関数: 折りたたみセクションで表示
       - 計算結果: FWHM、位相整合、HOM干渉のデータ
    4. **グラフのダウンロード**
       - 各グラフの下にあるリンクからPNG形式でダウンロード可能
    """)

# ポンプ光設定（折りたたみ形式に変更）
with st.sidebar.expander("ポンプ光設定", expanded=True):
    CWL = st.slider("中心波長 (nm)", 780, 810, 792)
    FWHM = st.slider("スペクトル幅 FWHM (nm)", 1.0, 20.0, 9.5)

# 結晶設定（折りたたみ形式に変更）
with st.sidebar.expander("結晶設定", expanded=True):
    T = st.slider("温度 (°C)", 25.0, 100.0, 36.5)
    L = st.slider("長さ (mm)", 1, 50, 30)
    Lambda_0 = st.slider("分極反転周期 (μm)", 45.0, 47.0, 46.1, 0.1)

# 波長範囲設定（折りたたみ形式に変更）
with st.sidebar.expander("波長範囲設定", expanded=True):
    center_wavelength = st.slider("中心波長 (nm)", 1550, 1620, 1585)
    wavelength_span = st.slider("波長幅 (nm)", 10, 100, 70)
    lambda_min = center_wavelength - wavelength_span/2
    lambda_max = center_wavelength + wavelength_span/2
    num_pixels = st.slider("解像度 (ピクセル)", 100, 1600, 1600, 100)
    plot_range_ps = st.slider("JTI表示範囲 (ps)", 1, 15, 9)

# パラメータをdictにまとめる
params = {
    'CWL': CWL,      # nm (ポンプ光の中心波長)
    'FWHM': FWHM,    # nm (ポンプ光のFWHM)
    'T': T,          # °C (結晶温度) 
    'L': L,          # mm (結晶長)
    'Lambda_0': Lambda_0  # μm (分極反転周期)
}

# 進捗バー
progress_bar = st.progress(0)

# グラフをダウンロード可能にする関数
def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# 計算条件を表示する関数
def show_calculation_conditions(params):
    st.markdown("### 計算条件")
    conditions_data = {
        "項目": ["ポンプ光中心波長", "ポンプ光スペクトル幅", "結晶温度", "結晶長", "分極反転周期", "信号光波長範囲", "アイドラー光波長範囲"],
        "値": [
            f"{params['CWL']:.2f} nm",
            f"{params['FWHM']:.2f} nm",
            f"{params['T']:.1f} °C",
            f"{params['L']} mm",
            f"{params['Lambda_0']:.2f} μm",
            f"{lambda_min:.2f} - {lambda_max:.2f} nm",
            f"{lambda_min:.2f} - {lambda_max:.2f} nm"
        ]
    }
    st.table(conditions_data)

# --- 物理定数 ---
C_LIGHT = 299792458  # 光速 [m/s]
NM_TO_M = 1e-9
UM_TO_M = 1e-6
MM_TO_M = 1e-3
HZ_TO_THZ = 1e-12
S_TO_PS = 1e12

# --- 屈折率関数 ---
def n_p_0(p_wavelength):
    WL_p_um = p_wavelength / 1000  # nm -> μm
    return np.sqrt(2.09930 + (0.922683 / (1 - 0.0467695 / (WL_p_um**2))) - 0.0138408*(WL_p_um**2))

def n_s_0(X):
    X_um = X / 1000  # nm -> μm
    return np.sqrt(2.09930 + (0.922683 / (1 - 0.0467695 / (X_um**2))) - 0.0138408*(X_um**2))

def n_i_0(Y):
    Y_um = Y / 1000  # nm -> μm
    return np.sqrt(2.12725 + (1.18431 / (1 - 0.0514852 / (Y_um**2))) + 
                  (0.6603 / (1 - 100.00507 / (Y_um**2))) - 0.00968956 * (Y_um**2))

# --- 温度補正項 ---
def delta_n_p(p_wavelength, T):
    p_um = p_wavelength / 1000  # nm -> μm
    return ((6.2897e-6)/p_um**0 + (6.3061e-6)/p_um**1 + (-6.0629e-6)/p_um**2 + (2.6486e-6)/p_um**3) * (T-25) \
           + ((-0.14445e-8)/p_um**0 + (2.2244e-8)/p_um**1 + (-3.5770e-8)/p_um**2 + (1.3470e-8)/p_um**3) * (T-25)**2

def delta_n_s(X, T):
    X_um = X / 1000  # nm -> μm
    return ((6.2897e-6)/X_um**0 + (6.3061e-6)/X_um**1 + (-6.0629e-6)/X_um**2 + (2.6486e-6)/X_um**3) * (T-25) \
           + ((-0.14445e-8)/X_um**0 + (2.2244e-8)/X_um**1 + (-3.5770e-8)/X_um**2 + (1.3470e-8)/X_um**3) * (T-25)**2

def delta_n_i(Y, T):
    Y_um = Y / 1000  # nm -> μm
    return ((9.9587e-6)/Y_um**0 + (9.9228e-6)/Y_um**1 + (-8.9603e-6)/Y_um**2 + (4.1010e-6)/Y_um**3) * (T-25) \
           + ((-1.1882e-8)/Y_um**0 + (10.459e-8)/Y_um**1 + (-9.8136e-8)/Y_um**2 + (3.1481e-8)/Y_um**3) * (T-25)**2

# --- ポンプ光スペクトル関数 ---
def gaussian_2d(X, Y, CWL, FWHM):
    """ポンプ光のスペクトル形状をガウス関数でモデル化する (逆波長空間)"""
    sigma_inv_wl = (FWHM / CWL**2) / (2 * np.sqrt(2 * np.log(2)))
    inv_wl_p = (1/X) + (1/Y)
    inv_wl_center = 1 / CWL
    exponent = -((inv_wl_p - inv_wl_center)**2) / (2 * sigma_inv_wl**2)
    return np.exp(exponent)

# --- 位相整合関数 ---
def sinc_function(X, Y, L, Lambda, T):
    """位相整合条件を表すsinc関数を計算する"""
    L_nm = L * 1e6  # mm -> nm
    Lambda_nm = Lambda * 1e3  # μm -> nm

    # 屈折率に温度補正を適用
    n_s = n_s_0(X) + delta_n_s(X, T)
    n_i = n_i_0(Y) + delta_n_i(Y, T)

    # ポンプ波長の計算 (1/λ_p = 1/λ_s + 1/λ_i)
    P_WL = 1 / (1/X + 1/Y)
    n_p = n_p_0(P_WL) + delta_n_p(P_WL, T)

    # 波数の計算（k = 2πn/λ）
    k_s = 2 * np.pi * n_s / X
    k_i = 2 * np.pi * n_i / Y
    k_p = 2 * np.pi * n_p / P_WL

    # 位相不整合量 Δk の計算 (準位相整合 QPM)
    delta_k = k_p - k_s - k_i + 2 * 1 * np.pi / Lambda_nm

    # sinc関数の計算 (ゼロ割回避)
    delta_k_safe = delta_k + 1e-15
    arg = delta_k_safe * L_nm / 2
    return np.sin(arg) / arg

# --- JSA関数 (Joint Spectral Amplitude) ---
def JSA(lambda_s, lambda_i, params):
    """波長空間でのJoint Spectral Amplitude (JSA)を計算する"""
    return gaussian_2d(lambda_s, lambda_i, params['CWL'], params['FWHM']) * \
           sinc_function(lambda_s, lambda_i, params['L'], params['Lambda_0'], params['T'])

# --- 周波数空間JSA関数 ---
def JSA_freq(omega_s, omega_i, params):
    """周波数空間でのJoint Spectral Amplitude (JSA)を計算する"""
    lambda_s_nm = (C_LIGHT / (omega_s + 1e-15)) / NM_TO_M
    lambda_i_nm = (C_LIGHT / (omega_i + 1e-15)) / NM_TO_M
    return JSA(lambda_s_nm, lambda_i_nm, params)

# --- 計算ボタン ---
if st.button('シミュレーション実行'):
    start_time = time.time()
    
    # 波長グリッド (nm)
    progress_bar.progress(10)
    lambda_s_vals = np.linspace(lambda_min, lambda_max, num_pixels)
    lambda_i_vals = np.linspace(lambda_min, lambda_max, num_pixels)
    Lambda_s, Lambda_i = np.meshgrid(lambda_s_vals, lambda_i_vals, indexing="ij")
    
    # 周波数グリッド (Hz)
    progress_bar.progress(20)
    freq_s_max = C_LIGHT / (lambda_min * NM_TO_M)
    freq_s_min = C_LIGHT / (lambda_max * NM_TO_M)
    freq_i_max = C_LIGHT / (lambda_min * NM_TO_M)
    freq_i_min = C_LIGHT / (lambda_max * NM_TO_M)
    
    omega_s_vals = np.linspace(freq_s_min, freq_s_max, num_pixels)
    omega_i_vals = np.linspace(freq_i_min, freq_i_max, num_pixels)
    Omega_s, Omega_i = np.meshgrid(omega_s_vals, omega_i_vals, indexing="ij")
    
    # ポンプ光分布関数の計算
    progress_bar.progress(25)
    Z_pump = gaussian_2d(Lambda_s, Lambda_i, params['CWL'], params['FWHM'])
    Z_pump_normalized = Z_pump / np.max(Z_pump)
    
    # 位相整合関数の計算
    progress_bar.progress(28)
    Z_phase = sinc_function(Lambda_s, Lambda_i, params['L'], params['Lambda_0'], params['T'])
    Z_phase_normalized = np.abs(Z_phase) / np.max(np.abs(Z_phase))
    
    # JSI (波長空間)
    progress_bar.progress(30)
    JSA_wl = JSA(Lambda_s, Lambda_i, params)
    JSI_wl = np.abs(JSA_wl)**2
    if np.all(np.isnan(JSI_wl)) or np.max(JSI_wl) == 0:
        st.warning("警告: 波長空間のJSI計算結果が不正です。")
        JSI_wl_normalized = np.zeros_like(JSI_wl)
    else:
        JSI_wl_normalized = JSI_wl / np.max(JSI_wl)
    
    # JSI (周波数空間)
    progress_bar.progress(50)
    JSA_fr = JSA_freq(Omega_s, Omega_i, params)
    JSI_fr = np.abs(JSA_fr)**2
    if np.all(np.isnan(JSI_fr)) or np.max(JSI_fr) == 0:
        st.warning("警告: 周波数空間のJSI計算結果が不正です。")
        JSI_fr_normalized = np.zeros_like(JSI_fr)
    else:
        JSI_fr_normalized = JSI_fr / np.max(JSI_fr)
    
    # JTI (時間領域)
    progress_bar.progress(70)
    if np.any(np.isnan(JSA_fr)):
        st.warning("警告: JSA_frにNaNが含まれています。FFT計算をスキップします。")
        JTI_normalized = np.zeros_like(JSA_fr)
    else:
        JSA_time = fftshift(fft2(fftshift(JSA_fr)))
        JTI = np.abs(JSA_time)**2
        if np.all(np.isnan(JTI)) or np.max(JTI) == 0:
            st.warning("警告: JTI計算結果が不正です。")
            JTI_normalized = np.zeros_like(JTI)
        else:
            JTI_normalized = JTI / np.max(JTI)
    
    # 時間軸の計算
    progress_bar.progress(80)
    df_s = omega_s_vals[1] - omega_s_vals[0]
    df_i = omega_i_vals[1] - omega_i_vals[0]
    time_s_vals = fftshift(fftfreq(num_pixels, d=df_s))
    time_i_vals = fftshift(fftfreq(num_pixels, d=df_i))
    time_s_ps = time_s_vals * S_TO_PS
    time_i_ps = time_i_vals * S_TO_PS
    center_t = 0
    
    # JSIの射影を計算
    x_projection = np.sum(JSI_wl_normalized, axis=1)
    y_projection = np.sum(JSI_wl_normalized, axis=0)
    
    # FWHMを計算する関数
    def calculate_fwhm(wavelength, intensity):
        max_intensity = np.max(intensity)
        half_max_intensity = max_intensity / 2.0
        
        # 半値以上のインデックスを取得
        above_half_max_indices = np.where(intensity >= half_max_intensity)[0]
        if not above_half_max_indices.size:
            return None
            
        # 半値以上の範囲の開始と終了インデックス
        start_index = above_half_max_indices[0]
        end_index = above_half_max_indices[-1]
        
        # FWHMを計算
        fwhm = wavelength[end_index] - wavelength[start_index]
        return fwhm
    
    # FWHMを計算
    fwhm_x = calculate_fwhm(lambda_s_vals, x_projection)
    fwhm_y = calculate_fwhm(lambda_i_vals, y_projection)
    
    # プロット
    progress_bar.progress(90)
    
    # 計算条件を最初に表示
    show_calculation_conditions(params)
    
    # JSIとJTIの表示
    with st.expander("JSIとJTI", expanded=True):
        col_jsi, col_jti = st.columns(2)
        
        with col_jsi:
            st.markdown("### JSI")
            fig_jsi, ax_jsi = plt.subplots(figsize=(8, 8))
            im_jsi = ax_jsi.imshow(JSI_wl_normalized.T,
                        extent=[lambda_s_vals.min(), lambda_s_vals.max(), 
                               lambda_i_vals.min(), lambda_i_vals.max()],
                               origin='lower', cmap='jet', aspect='equal')
            ax_jsi.set_xlabel(r'$\lambda_s$ [nm]')
            ax_jsi.set_ylabel(r'$\lambda_i$ [nm]')
            ax_jsi.set_title('Joint Spectral Intensity')
            plt.colorbar(im_jsi, ax=ax_jsi)
            st.pyplot(fig_jsi)
            # JSIダウンロードリンク
            st.markdown(get_image_download_link(fig_jsi, "jsi.png", "Download JSI"), unsafe_allow_html=True)
            
        with col_jti:
            st.markdown("### JTI")
            fig_jti, ax_jti = plt.subplots(figsize=(8, 8))
            im_jti = ax_jti.imshow(JTI_normalized.T,
                        extent=[time_s_ps.min(), time_s_ps.max(), 
                               time_i_ps.min(), time_i_ps.max()],
                                origin='lower', cmap='jet', aspect='equal')
            ax_jti.set_xlabel(r'$\tau_s$ [ps]')
            ax_jti.set_ylabel(r'$\tau_i$ [ps]')
            ax_jti.set_title('Joint Temporal Intensity')
            # JTI表示範囲を中央を中心に設定
            ax_jti.set_xlim(center_t - plot_range_ps / 2, center_t + plot_range_ps / 2)
            ax_jti.set_ylim(center_t - plot_range_ps / 2, center_t + plot_range_ps / 2)
            plt.colorbar(im_jti, ax=ax_jti)
            st.pyplot(fig_jti)
            # JTIダウンロードリンク
            st.markdown(get_image_download_link(fig_jti, "jti.png", "Download JTI"), unsafe_allow_html=True)
    
    # ポンプ光分布関数と位相整合関数の表示
    with st.expander("ポンプ光分布関数と位相整合関数", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            fig_pump, ax_pump = plt.subplots(figsize=(8, 8))
            im_pump = ax_pump.imshow(Z_pump_normalized.T,
                            extent=[lambda_s_vals.min(), lambda_s_vals.max(), 
                                   lambda_i_vals.min(), lambda_i_vals.max()],
                            origin='lower', cmap='jet', aspect='equal')
            ax_pump.set_xlabel(r'$\lambda_s$ [nm]')
            ax_pump.set_ylabel(r'$\lambda_i$ [nm]')
            ax_pump.set_title('Pump Distribution Function')
            plt.colorbar(im_pump, ax=ax_pump)
            st.pyplot(fig_pump)
            # ポンプ光分布関数ダウンロードリンク
            st.markdown(get_image_download_link(fig_pump, "pump_distribution.png", "Download Pump Distribution"), unsafe_allow_html=True)
            
        with col2:
            fig_phase, ax_phase = plt.subplots(figsize=(8, 8))
            im_phase = ax_phase.imshow(Z_phase_normalized.T,
                             extent=[lambda_s_vals.min(), lambda_s_vals.max(), 
                                    lambda_i_vals.min(), lambda_i_vals.max()],
                             origin='lower', cmap='jet', aspect='equal')
            ax_phase.set_xlabel(r'$\lambda_s$ [nm]')
            ax_phase.set_ylabel(r'$\lambda_i$ [nm]')
            ax_phase.set_title('Phase Matching Function')
            plt.colorbar(im_phase, ax=ax_phase)
            st.pyplot(fig_phase)
            # 位相整合関数ダウンロードリンク
            st.markdown(get_image_download_link(fig_phase, "phase_matching.png", "Download Phase Matching"), unsafe_allow_html=True)
            
    # JSI射影の表示
    with st.expander("JSI射影", expanded=True):
        fig_proj, ax_proj = plt.subplots(figsize=(12, 6))
        ax_proj.plot(lambda_s_vals, x_projection, label='Signal Photon Projection')
        ax_proj.plot(lambda_i_vals, y_projection, label='Idler Photon Projection')
        ax_proj.set_xlabel('Wavelength [nm]')
        ax_proj.set_ylabel('Intensity')
        ax_proj.set_title('JSI Projections')
        ax_proj.legend()
        st.pyplot(fig_proj)
        # 射影グラフダウンロードリンク
        st.markdown(get_image_download_link(fig_proj, "jsi_projections.png", "Download JSI Projections"), unsafe_allow_html=True)
    
    # FWHMとピーク計算
    if fwhm_x is not None and fwhm_y is not None:
        peak_x_index = np.argmax(x_projection)
        peak_y_index = np.argmax(y_projection)
        peak_x_wavelength = lambda_s_vals[peak_x_index]
        peak_y_wavelength = lambda_i_vals[peak_y_index]
    
    # 位相整合パラメータ計算
    P_WL_specific = params['CWL']
    X_specific = 2 * params['CWL']  # 縮退の場合
    Y_specific = 2 * params['CWL']  # 縮退の場合
    
    n_s_specific = n_s_0(X_specific) + delta_n_s(X_specific, params['T'])
    n_i_specific = n_i_0(Y_specific) + delta_n_i(Y_specific, params['T'])
    n_p_specific = n_p_0(P_WL_specific) + delta_n_p(P_WL_specific, params['T'])
    
    k_s_specific = 2 * np.pi * n_s_specific / X_specific
    k_i_specific = 2 * np.pi * n_i_specific / Y_specific
    k_p_specific = 2 * np.pi * n_p_specific / P_WL_specific
    
    Lambda_nm = params['Lambda_0'] * 1e3  # μm -> nm
    
    delta_k_specific = k_p_specific - k_s_specific - k_i_specific + 2 * np.pi / Lambda_nm
    delta_k_no_grating_specific = k_p_specific - k_s_specific - k_i_specific
    
    # HOM干渉の計算
    c = 2.99792458e17  # nm/s (光速)
    
    # 中心波長での角周波数を計算
    omega_s_center = 2 * np.pi * c / (params['CWL'] * 2)
    omega_i_center = 2 * np.pi * c / (params['CWL'] * 2)
    
    # 数値微分を用いて k_s' と k_i' を計算 (中心差分)
    delta_omega = 1e10  # 微小変化
    
    # シグナル光の中心波長付近での屈折率の微分
    n_s_center_plus = n_s_0((2 * np.pi * c) / (omega_s_center + delta_omega)) + \
                     delta_n_s((2 * np.pi * c) / (omega_s_center + delta_omega), params['T'])
    n_s_center_minus = n_s_0((2 * np.pi * c) / (omega_s_center - delta_omega)) + \
                      delta_n_s((2 * np.pi * c) / (omega_s_center - delta_omega), params['T'])
    d_n_s_d_omega = (n_s_center_plus - n_s_center_minus) / (2 * delta_omega)
    
    # アイドラ光の中心波長付近での屈折率の微分
    n_i_center_plus = n_i_0((2 * np.pi * c) / (omega_i_center + delta_omega)) + \
                     delta_n_i((2 * np.pi * c) / (omega_i_center + delta_omega), params['T'])
    n_i_center_minus = n_i_0((2 * np.pi * c) / (omega_i_center - delta_omega)) + \
                      delta_n_i((2 * np.pi * c) / (omega_i_center - delta_omega), params['T'])
    d_n_i_d_omega = (n_i_center_plus - n_i_center_minus) / (2 * delta_omega)
    
    # k_s' と k_i' の計算
    k_s_prime = (n_s_center_plus + omega_s_center * d_n_s_d_omega) / c
    k_i_prime = (n_i_center_plus + omega_i_center * d_n_i_d_omega) / c
    
    # |k_s' - k_i'| * L の計算
    L_nm = params['L'] * 1e6  # mm -> nm
    abs_k_prime_diff_L = abs(k_s_prime - k_i_prime) * L_nm
    
    # 計算結果を表形式で表示（スクロールなしで全体表示）
    st.subheader("計算結果")
    
    # 表データの準備 - インデックスを表示しないようにする
    jsi_data = {
        "項目": ["Signal光子のFWHM", "Signal光子のピーク波長", "Idler光子のFWHM", "Idler光子のピーク波長"],
        "値": [
            f"{fwhm_x:.2f} nm" if fwhm_x is not None else "計算不可",
            f"{peak_x_wavelength:.2f} nm" if fwhm_x is not None else "計算不可",
            f"{fwhm_y:.2f} nm" if fwhm_y is not None else "計算不可",
            f"{peak_y_wavelength:.2f} nm" if fwhm_y is not None else "計算不可"
        ]
    }
    
    phase_data = {
        "項目": ["中心波長におけるdelta_k", "分極反転のないdelta_k"],
        "値": [
            f"{delta_k_specific:.10f} nm^-1",
            f"{delta_k_no_grating_specific:.10f} nm^-1"
        ]
    }
    
    hom_data = {
        "項目": ["HOM干渉の三角形幅 (コヒーレンス時間)", "HOM干渉の三角形幅 (コヒーレンス長)"],
        "値": [
            f"{abs_k_prime_diff_L*1e12:.4f} ps",
            f"{c*abs_k_prime_diff_L*1e-3:.4f} μm"
        ]
    }
    
    # 3つの表を表示
    st.markdown("**JSI分析**")
    df_jsi = pd.DataFrame(jsi_data).set_index("項目").T
    st.write(df_jsi.style.hide(axis="index").set_properties(**{'width': '100%'}))
    
    st.markdown("**HOM干渉分析**")
    df_hom = pd.DataFrame(hom_data).set_index("項目").T
    st.write(df_hom.style.hide(axis="index").set_properties(**{'width': '100%'}))
    
    st.markdown("**位相整合分析**")
    df_phase = pd.DataFrame(phase_data).set_index("項目").T
    st.write(df_phase.style.hide(axis="index").set_properties(**{'width': '100%'}))
    
    # 完了
    progress_bar.progress(100)

else:
    st.info("「シミュレーション実行」ボタンをクリックして、計算を開始してください。") 