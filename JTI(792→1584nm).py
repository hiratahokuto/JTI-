import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq


# --- 実験パラメータ ---
params = {
    'CWL': 792,      # nm (ポンプ光の中心波長)
    'FWHM': 9.5,     # nm (ポンプ光のFWHM)
    'T': 36.5,       # °C (結晶温度) 
    'L': 30,         # mm (結晶長)
    'Lambda_0': 46.1 # μm (分極反転周期)
}

# --- グラフ設定(JSI) ---
num_pixels = 1600 #実際の実験データも1600*1600ピクセル
lambda_s_min, lambda_s_max = 1550, 1620 # nm
lambda_i_min, lambda_i_max = 1550, 1620 # nm

# --- グラフ設定(JTI) ---
plot_range_ps = 9 #時間差0かを中心として何psの幅を見るかを指定。これによって拡大を行うことで、ピクセル数が大きいJTIが表示されてしまっているということになる。

# --- 物理定数 ---
C_LIGHT = 299792458  # 光速 [m/s]
NM_TO_M = 1e-9
UM_TO_M = 1e-6
MM_TO_M = 1e-3
HZ_TO_THZ = 1e-12
S_TO_PS = 1e12

# --- 屈折率 ---
# セルマイヤー方程式
def n_p_0(p_wavelength):
    WL_p_um = p_wavelength / 1000  # nm -> μm
    # KTP o-ray (ポンプ光用と仮定、必要なら確認・修正)
    return np.sqrt(2.09930 + (0.922683 / (1 - 0.0467695 / (WL_p_um**2))) - 0.0138408*(WL_p_um**2))

def n_s_0(X):
    X_um = X / 1000  # nm -> μm
    # KTP o-ray (シグナル光用と仮定)
    return np.sqrt(2.09930 + (0.922683 / (1 - 0.0467695 / (X_um**2))) - 0.0138408*(X_um**2))

def n_i_0(Y):
    Y_um = Y / 1000  # nm -> μm
    # KTP e-ray (アイドラ光用と仮定) - 元のコードの式を使用
    return np.sqrt(2.12725 + (1.18431 / (1 - 0.0514852 / (Y_um**2))) +
                   (0.6603 / (1 - 100.00507 / (Y_um**2))) - 0.00968956 * (Y_um**2))

# セルマイヤー方程式の温度補正項
def delta_n_p(p_wavelength, T): # 引数 T を追加
    p_um = p_wavelength / 1000  # nm -> μm
    # KTP o-ray (ポンプ光用と仮定)
    return ((6.2897e-6)/p_um**0 + (6.3061e-6)/p_um**1 + (-6.0629e-6)/p_um**2 + (2.6486e-6)/p_um**3) * (T-25) \
           + ((-0.14445e-8)/p_um**0 + (2.2244e-8)/p_um**1 + (-3.5770e-8)/p_um**2 + (1.3470e-8)/p_um**3) * (T-25)**2

def delta_n_s(X, T): # 引数 T を追加
    X_um = X / 1000  # nm -> μm
    # KTP o-ray (シグナル光用と仮定)
    return ((6.2897e-6)/X_um**0 + (6.3061e-6)/X_um**1 + (-6.0629e-6)/X_um**2 + (2.6486e-6)/X_um**3) * (T-25) \
           + ((-0.14445e-8)/X_um**0 + (2.2244e-8)/X_um**1 + (-3.5770e-8)/X_um**2 + (1.3470e-8)/X_um**3) * (T-25)**2

def delta_n_i(Y, T): # 引数 T を追加
    Y_um = Y / 1000  # nm -> μm
    # KTP e-ray (アイドラ光用と仮定)
    return ((9.9587e-6)/Y_um**0 + (9.9228e-6)/Y_um**1 + (-8.9603e-6)/Y_um**2 + (4.1010e-6)/Y_um**3) * (T-25) \
           + ((-1.1882e-8)/Y_um**0 + (10.459e-8)/Y_um**1 + (-9.8136e-8)/Y_um**2 + (3.1481e-8)/Y_um**3) * (T-25)**2

# --- ポンプ光スペクトル関数 ---
def gaussian_2d(X, Y, CWL, FWHM):
    """
    ポンプ光のスペクトル形状をガウス関数でモデル化する (逆波長空間)。
    エネルギー保存則 (1/λ_p = 1/λ_s + 1/λ_i) を考慮する。

    Args:
        X (np.ndarray): シグナル光子の波長グリッド (nm)
        Y (np.ndarray): アイドラ光子の波長グリッド (nm)
        CWL (float): ポンプ光の中心波長 (nm)
        FWHM (float): ポンプ光のFWHM (nm)

    Returns:
        np.ndarray: ポンプ光スペクトル形状
    """
    # 元のコードの sigma 定義 (逆波長空間での標準偏差)
    sigma_inv_wl = (FWHM / CWL**2) / (2 * np.sqrt(2 * np.log(2))) # 1/nm 単位
    # エネルギー保存からポンプの逆波長を計算 1/λ_p = 1/λ_s + 1/λ_i
    inv_wl_p = (1/X) + (1/Y)
    inv_wl_center = 1 / CWL
    exponent = -((inv_wl_p - inv_wl_center)**2) / (2 * sigma_inv_wl**2)
    return np.exp(exponent)

# --- 位相整合関数  ---
def sinc_function(X, Y, L, Lambda, T):
    """
    位相整合条件を表すsinc関数を計算する。

    Args:
        X (np.ndarray): シグナル光子の波長グリッド (nm)
        Y (np.ndarray): アイドラ光子の波長グリッド (nm)
        L (float): 結晶長 (mm)
        Lambda (float): 分極反転周期 (μm)
        T (float): 結晶温度 (°C)

    Returns:
        np.ndarray: 位相整合関数値
    """
    L_nm = L * 1e6  # mm -> nm
    Lambda_nm = Lambda * 1e3  # μm -> nm

    # 屈折率に温度補正を適用 (温度 T を使用)
    n_s = n_s_0(X) + delta_n_s(X, T)
    n_i = n_i_0(Y) + delta_n_i(Y, T)

    # ポンプ波長の計算 (1/λ_p = 1/λ_s + 1/λ_i)
    P_WL = 1 / (1/X + 1/Y)
    n_p = n_p_0(P_WL) + delta_n_p(P_WL, T)

    # 波数の計算（k = 2πn/λ）
    k_s = 2 * np.pi * n_s / X
    k_i = 2 * np.pi * n_i / Y
    k_p = 2 * np.pi * n_p / P_WL

    # 位相不整合量 Δk の計算 (準位相整合 QPM) - 元のコードの符号に戻す
    # Δk = k_p - k_s - k_i + K_g (K_g = 2πm / Λ)
    delta_k = k_p - k_s - k_i + 2 * 1 * np.pi / Lambda_nm # m=1

    # sinc関数の計算 sinc(x) = sin(x)/x
    # ゼロ割を回避するために微小な値を加える
    delta_k_safe = delta_k + 1e-15
    arg = delta_k_safe * L_nm / 2
    return np.sin(arg) / arg

# --- JSA関数 (Joint Spectral Amplitude) ---
def JSA(lambda_s, lambda_i, params):
    """
    波長空間でのJoint Spectral Amplitude (JSA)を計算する。
    """
    return gaussian_2d(lambda_s, lambda_i, params['CWL'], params['FWHM']) * \
           sinc_function(lambda_s, lambda_i, params['L'], params['Lambda_0'], params['T'])

# --- 周波数空間JSA関数 ---
def JSA_freq(omega_s, omega_i, params):
    """
    周波数空間でのJoint Spectral Amplitude (JSA)を計算する。
    """
    lambda_s_nm = (C_LIGHT / (omega_s + 1e-15)) / NM_TO_M
    lambda_i_nm = (C_LIGHT / (omega_i + 1e-15)) / NM_TO_M
    return JSA(lambda_s_nm, lambda_i_nm, params)


# 波長グリッド (nm)
lambda_s_vals = np.linspace(lambda_s_min, lambda_s_max, num_pixels)
lambda_i_vals = np.linspace(lambda_i_min, lambda_i_max, num_pixels)
Lambda_s, Lambda_i = np.meshgrid(lambda_s_vals, lambda_i_vals, indexing="ij")

# --- 周波数グリッド (Hz) ---
# 等間隔な周波数グリッド (FFTのため)
freq_s_max = C_LIGHT / (lambda_s_min * NM_TO_M) # Hz
freq_s_min = C_LIGHT / (lambda_s_max * NM_TO_M) # Hz
freq_i_max = C_LIGHT / (lambda_i_min * NM_TO_M) # Hz
freq_i_min = C_LIGHT / (lambda_i_max * NM_TO_M) # Hz

omega_s_vals = np.linspace(freq_s_min, freq_s_max, num_pixels) # Hz
omega_i_vals = np.linspace(freq_i_min, freq_i_max, num_pixels) # Hz
Omega_s, Omega_i = np.meshgrid(omega_s_vals, omega_i_vals, indexing="ij")

# --- プロット設定 ---
plt.rcParams['font.size'] = 12

# --- 計算 ---
# JSI (波長空間)
print("Calculating JSA in wavelength space...")
JSA_wl = JSA(Lambda_s, Lambda_i, params)
JSI_wl = np.abs(JSA_wl)**2
if np.all(np.isnan(JSI_wl)) or np.max(JSI_wl) == 0:
    print("警告: 波長空間のJSI計算結果が不正です。")
    JSI_wl_normalized = np.zeros_like(JSI_wl)
else:
    JSI_wl_normalized = JSI_wl / np.max(JSI_wl)
print("JSI (wavelength) calculation complete.")

# JSI (周波数空間)
print("Calculating JSA in frequency space...")
JSA_fr = JSA_freq(Omega_s, Omega_i, params)
JSI_fr = np.abs(JSA_fr)**2
if np.all(np.isnan(JSI_fr)) or np.max(JSI_fr) == 0:
    print("警告: 周波数空間のJSI計算結果が不正です。")
    JSI_fr_normalized = np.zeros_like(JSI_fr)
else:
    # 周波数空間の強度も規格化（表示のため）
    JSI_fr_normalized = JSI_fr / np.max(JSI_fr)
print("JSI (frequency) calculation complete.")


# JTI (時間領域)
print("Calculating JTI (FFT)...")
if np.any(np.isnan(JSA_fr)):
    print("警告: JSA_frにNaNが含まれています。FFT計算をスキップします。")
    JTI_normalized = np.zeros_like(JSA_fr)
else:
    JSA_time = fftshift(fft2(fftshift(JSA_fr))) #量子光シンセス(QOS)の実行部分
    JTI = np.abs(JSA_time)**2
    if np.all(np.isnan(JTI)) or np.max(JTI) == 0:
         print("警告: JTI計算結果が不正です。")
         JTI_normalized = np.zeros_like(JTI)
    else:
        JTI_normalized = JTI / np.max(JTI)
print("JTI calculation complete.")


# --- 時間軸の計算 ---
df_s = omega_s_vals[1] - omega_s_vals[0]
df_i = omega_i_vals[1] - omega_i_vals[0]
time_s_vals = fftshift(fftfreq(num_pixels, d=df_s))
time_i_vals = fftshift(fftfreq(num_pixels, d=df_i))
time_s_ps = time_s_vals * S_TO_PS
time_i_ps = time_i_vals * S_TO_PS
center_t = 0
num_ticks_jti = 5


# --- プロット ---
print("Generating plots...")
fig, (ax_jsi, ax_jti) = plt.subplots(1, 2, figsize=(15, 7))

# --- JSI ---
im_jsi = ax_jsi.imshow(JSI_wl_normalized.T,
                       extent=[lambda_s_vals.min(), lambda_s_vals.max(), lambda_i_vals.min(), lambda_i_vals.max()],
                       origin='lower', cmap='jet', aspect='auto',
                       interpolation='nearest')
ax_jsi.set_xlabel(r'$\lambda_s$ [nm]', fontweight='bold', fontsize=16) # 軸ラベルのフォントサイズを大きくしました
ax_jsi.set_ylabel(r'$\lambda_i$ [nm]', fontweight='bold', fontsize=16) # 軸ラベルのフォントサイズを大きくしました
ax_jsi.set_title('JSI', fontweight='bold') # 太字に変更

num_ticks_wl = 5
wl_s_ticks = np.linspace(lambda_s_vals.min(), lambda_s_vals.max(), num_ticks_wl)
wl_i_ticks = np.linspace(lambda_i_vals.min(), lambda_i_vals.max(), num_ticks_wl)
ax_jsi.set_xticks(wl_s_ticks)
ax_jsi.set_yticks(wl_i_ticks)
# 目盛りラベルを太字にする
ax_jsi.set_xticklabels(['{:.0f}'.format(wl) for wl in wl_s_ticks], fontweight='bold')
ax_jsi.set_yticklabels(['{:.0f}'.format(wl) for wl in wl_i_ticks], fontweight='bold')

ax_jsi_freq_s = ax_jsi.twiny()
ax_jsi_freq_i = ax_jsi.twinx()

num_ticks_freq = 5
omega_s_THz_vals = omega_s_vals * HZ_TO_THZ
omega_i_THz_vals = omega_i_vals * HZ_TO_THZ

ax_jsi_freq_s.set_xlim(ax_jsi.get_xlim())
freq_s_tick_positions_wl = np.linspace(lambda_s_vals.min(), lambda_s_vals.max(), num_ticks_freq)
ax_jsi_freq_s.set_xticks(freq_s_tick_positions_wl)
freq_labels_s = np.linspace(omega_s_THz_vals.max(), omega_s_THz_vals.min(), num_ticks_freq)
# 目盛りラベルを太字にする
ax_jsi_freq_s.set_xticklabels(['{:.1f}'.format(f) for f in freq_labels_s], fontweight='bold')
ax_jsi_freq_s.set_xlabel(r'$\omega_s$ [THz]', fontweight='bold', fontsize=16) # 軸ラベルのフォントサイズを大きくしました

ax_jsi_freq_i.set_ylim(ax_jsi.get_ylim())
freq_i_tick_positions_wl = np.linspace(lambda_i_vals.min(), lambda_i_vals.max(), num_ticks_freq)
ax_jsi_freq_i.set_yticks(freq_i_tick_positions_wl)
freq_labels_i = np.linspace(omega_i_THz_vals.max(), omega_i_THz_vals.min(), num_ticks_freq)
# 目盛りラベルを太字にする
ax_jsi_freq_i.set_yticklabels(['{:.1f}'.format(f) for f in freq_labels_i], fontweight='bold')
ax_jsi_freq_i.set_ylabel(r'$\omega_i$ [THz]', fontweight='bold', fontsize=16) # 軸ラベルのフォントサイズを大きくしました


# --- JTI ---
if np.any(np.isnan(JTI_normalized)) or np.max(JTI_normalized) == 0:
    vmin, vmax = 0, 1
    print("警告: JTIデータが不正なため、カラーバー範囲を[0, 1]に設定します。")
else:
    vmin = 0
    vmax = np.max(JTI_normalized)

im_jti = ax_jti.imshow(JTI_normalized.T,
                       extent=[time_s_ps.min(), time_s_ps.max(), time_i_ps.min(), time_i_ps.max()],
                       origin='lower', cmap='jet', aspect='auto',
                       vmin=vmin, vmax=vmax)
ax_jti.set_xlabel(r'$\Delta \tau_s$ [ps]', fontweight='bold', fontsize=16) # 軸ラベルのフォントサイズを大きくしました
ax_jti.set_ylabel(r'$\Delta \tau_i$ [ps]', fontweight='bold', fontsize=16) # 軸ラベルのフォントサイズを大きくしました
ax_jti.set_title('JTI', fontweight='bold') # 太字に変更

ax_jti.set_xlim(center_t - plot_range_ps / 2, center_t + plot_range_ps / 2)
ax_jti.set_ylim(center_t - plot_range_ps / 2, center_t + plot_range_ps / 2)

# JTI の目盛りを設定し、ラベルを太字にする
num_ticks_jti = 5 # JSI と同じ数にする
time_ticks = np.linspace(center_t - plot_range_ps / 2, center_t + plot_range_ps / 2, num_ticks_jti)
ax_jti.set_xticks(time_ticks)
ax_jti.set_yticks(time_ticks)
ax_jti.set_xticklabels(['{:.1f}'.format(t) for t in time_ticks], fontweight='bold')
ax_jti.set_yticklabels(['{:.1f}'.format(t) for t in time_ticks], fontweight='bold')


# --- 全体調整と表示 ---
# レイアウト調整
plt.tight_layout()
print("Displaying plots...")
plt.show()

print("Script finished.")