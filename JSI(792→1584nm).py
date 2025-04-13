import numpy as np
import matplotlib.pyplot as plt

# 実験パラメーター
CWL = 792  # nm (ポンプ光の中心波長)
FWHM = 9.5  # nm (ポンプ光のFWHM)
T = 36.5  # °C (結晶温度)(25-100度) #36.5°Cで縮退
L = 30  # mm (結晶長)
Lambda_0 = 46.1  # μm (分極反転周期)

# 波長範囲の設定
x = np.linspace(1560, 1610, 1600)  # シグナル光子の波長プロット範囲 (nm)  #実際のJSIのピクセル数も約1600らしい
y = np.linspace(1560, 1610, 1600)  # アイドラ光子の波長プロット範囲 (nm)
X, Y = np.meshgrid(x, y, indexing="ij")
inv_X, inv_Y = 1/X, 1/Y  # 逆波長 (1/nm)

# エネルギー保存則に基づく分布関数
def gaussian_2d(inv_X, inv_Y, CWL, FWHM):
    inv_CWL = 1 / CWL  # 逆波長での中心
    sigma = (FWHM / CWL**2) / (2 * np.sqrt(2 * np.log(2)))  # 逆波長空間での標準偏差
    return np.exp(- ((inv_X + inv_Y -  inv_CWL) ** 2) / (2*(sigma ** 2)))

# セルマイヤー方程式
def n_p_0(p_wavelength):
    WL_p_um = p_wavelength / 1000  # nm -> μm
    return np.sqrt(2.09930 + (0.922683 / (1 - 0.0467695 / (WL_p_um**2))) - 0.0138408*(WL_p_um**2))

def n_s_0(x):
    x_um = x / 1000  # nm -> μm
    return np.sqrt(2.09930 + (0.922683 / (1 - 0.0467695 / (x_um**2))) - 0.0138408*(x_um**2))

def n_i_0(y):
    y_um = y / 1000  # nm -> μm
    return np.sqrt(2.12725 + (1.18431 / (1 - 0.0514852 / (y_um**2))) + 
                   (0.6603 / (1 - 100.00507 / (y_um**2))) - 0.00968956 * (y_um**2))

# セルマイヤー方程式の温度補正項
def delta_n_p(p_wavelength):
    p_um = p_wavelength / 1000  # nm -> μm
    return ((6.2897e-6)/p_um**0 + (6.3061e-6)/p_um**1 + (-6.0629e-6)/p_um**2 + (2.6486e-6)/p_um**3) * (T-25) \
           + ((-0.14445e-8)/p_um**0 + (2.2244e-8)/p_um**1 + (-3.5770e-8)/p_um**2 + (1.3470e-8)/p_um**3) * (T-25)**2

def delta_n_s(x):
    x_um = x / 1000  # nm -> μm
    return ((6.2897e-6)/x_um**0 + (6.3061e-6)/x_um**1 + (-6.0629e-6)/x_um**2 + (2.6486e-6)/x_um**3) * (T-25) \
           + ((-0.14445e-8)/x_um**0 + (2.2244e-8)/x_um**1 + (-3.5770e-8)/x_um**2 + (1.3470e-8)/x_um**3) * (T-25)**2

def delta_n_i(y):
    y_um = y / 1000  # nm -> μm
    return ((9.9587e-6)/y_um**0 + (9.9228e-6)/y_um**1 + (-8.9603e-6)/y_um**2 + (4.1010e-6)/y_um**3) * (T-25) \
           + ((-1.1882e-8)/y_um**0 + (10.459e-8)/y_um**1 + (-9.8136e-8)/y_um**2 + (3.1481e-8)/y_um**3) * (T-25)**2

# 位相整合関数
def sinc_function(X, Y, L, Lambda):
    L_nm = L * 1e6  # mm -> nm
    Lambda_nm = Lambda * 1e3  # μm -> nm

    # 屈折率に温度補正を適用
    n_s = n_s_0(X) + delta_n_s(X)
    n_i = n_i_0(Y) + delta_n_i(Y)

    # ポンプ波長の計算 (1/λ_p = 1/λ_s + 1/λ_i)
    P_WL = 1 / (1/X + 1/Y)
    n_p = n_p_0(P_WL) + delta_n_p(P_WL)

    # 波数の計算（k = 2πn/λ）
    k_s = 2 * np.pi * n_s / X
    k_i = 2 * np.pi * n_i / Y
    k_p = 2 * np.pi * n_p / P_WL

    # 位相不整合量 Δk の計算
    delta_k = k_p - k_s - k_i + 2 *1* np.pi / Lambda_nm
    delta_k_no_grating = k_p - k_s - k_i

    # 中心波長における位相不整合量を計算して出力
    P_WL_specific = 792  # ポンプ光の波長 (nm)
    X_specific = 1584  # シグナル光の波長 (nm)
    Y_specific = 1584  # アイドラ光の波長 (nm)

    n_s_specific = n_s_0(X_specific) + delta_n_s(X_specific)
    n_i_specific = n_i_0(Y_specific) + delta_n_i(Y_specific)
    n_p_specific = n_p_0(P_WL_specific) + delta_n_p(P_WL_specific)

    k_s_specific = 2 * np.pi * n_s_specific / X_specific
    k_i_specific = 2 * np.pi * n_i_specific / Y_specific
    k_p_specific = 2 * np.pi * n_p_specific / P_WL_specific

    Lambda_nm = Lambda * 1e3  # μm -> nm

    delta_k_specific = k_p_specific - k_s_specific - k_i_specific + 2 * np.pi / Lambda_nm
    delta_k_no_grating_specific = k_p_specific - k_s_specific - k_i_specific

    print(f"中心波長におけるdelta_k: {delta_k_specific:.15f} nm^-1")
    print(f"中心波長におけるdelta_k_no_grating: {delta_k_no_grating_specific:.15f} nm^-1")


    # sinc関数の計算
    return np.sin(delta_k * L_nm / 2) / (delta_k * L_nm / 2)

# 計算結果の保存
Z_pump = gaussian_2d(inv_X, inv_Y, CWL, FWHM)
Z_phase = sinc_function(X, Y, L, Lambda_0)

# 屈折率の計算（中心波長付近での値）
n_s_center = n_s_0(CWL * 2) + delta_n_s(CWL * 2)  # シグナル光の中心波長での屈折率
n_i_center = n_i_0(CWL * 2) + delta_n_i(CWL * 2)  # アイドラ光の中心波長での屈折率

# |k_s - k_i| * L の計算（中心波長付近での値）
L_nm = L * 1e6  # mm -> nm
k_s_center = 2 * np.pi * n_s_center / (CWL * 2)  # シグナル光の中心波長での波数
k_i_center = 2 * np.pi * n_i_center / (CWL * 2)  # アイドラ光の中心波長での波数
abs_k_diff_L = abs(k_s_center - k_i_center) * L_nm  # |k_s - k_i| * L


# 3つのグラフを表示するためのサブプロット作成
plt.figure(figsize=(18, 5))

# 1つ目のプロット: 励起光分布関数
plt.subplot(1, 3, 1)
heatmap1 = plt.imshow(Z_pump.T, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='jet', aspect='auto')
plt.colorbar(heatmap1)
plt.xlabel('Signal wavelength (nm)', fontsize=14, fontweight='bold')
plt.ylabel('Idler wavelength (nm)', fontsize=14, fontweight='bold')
plt.title('Pump envelope', fontsize=16, fontweight='bold')

# 2つ目のプロット: 位相整合関数
plt.subplot(1, 3, 2)
heatmap2 = plt.imshow(Z_phase.T, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='jet', aspect='auto')
plt.colorbar(heatmap2)
plt.xlabel('Signal wavelength (nm)', fontsize=14, fontweight='bold')
plt.ylabel('Idler wavelength (nm)', fontsize=14, fontweight='bold')
plt.title('Phase Matching', fontsize=16, fontweight='bold')

# 3つ目のプロット: 2つの関数の積の二乗
plt.subplot(1, 3, 3)
Z_combined = (Z_pump * Z_phase)**2
heatmap3 = plt.imshow(Z_combined.T, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='jet', aspect='auto')
plt.colorbar(heatmap3)
plt.xlabel('Signal wavelength (nm)', fontsize=14, fontweight='bold')
plt.ylabel('Idler wavelength (nm)', fontsize=14, fontweight='bold')
plt.title('JSI', fontsize=16, fontweight='bold')

# レイアウト調整
plt.tight_layout()
plt.show(block=False) 

# x軸方向とy軸方向にJSIを射影したグラフを計算
x_projection = np.sum(Z_combined, axis=1)
y_projection = np.sum(Z_combined, axis=0)

# FWHMを計算する関数
def calculate_fwhm(wavelength, intensity):
    max_intensity = np.max(intensity)
    half_max_intensity = max_intensity / 2.0
    
    # 半値以上のインデックスを取得
    above_half_max_indices = np.where(intensity >= half_max_intensity)[0]
    if not above_half_max_indices.size: # 半値以上の値がない場合の処理
        return None # または適切なエラー値を返す

    # 半値以上の範囲の開始と終了インデックス
    start_index = above_half_max_indices[0]
    end_index = above_half_max_indices[-1]

    # FWHMを計算
    fwhm = wavelength[end_index] - wavelength[start_index]
    return fwhm

# x軸射影のFWHMを計算
fwhm_x = calculate_fwhm(x, x_projection)

# y軸射影のFWHMを計算
fwhm_y = calculate_fwhm(y, y_projection)

# 新しいfigureを作成
plt.figure(figsize=(12, 4)) # サイズは適宜調整

# 4つ目のプロット: x軸射影とy軸射影を同一平面にプロット
plt.subplot(1, 1, 1) # 1行1列の1番目の位置に変更
plt.plot(x, x_projection, label='Signal axis projection') # x軸射影をプロット、凡例用ラベルを追加
plt.plot(y, y_projection, label='Idler axis projection') # y軸射影をプロット、凡例用ラベルを追加
plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold') # x軸ラベルを修正
plt.ylabel('Intensity', fontsize=14, fontweight='bold')
plt.title('JSI Projections', fontsize=16, fontweight='bold') # タイトルを修正
plt.legend() # 凡例を表示

# レイアウト調整
plt.tight_layout()

# FWHMをターミナルに出力
if fwhm_x is not None and fwhm_y is not None:
    peak_x_index = np.argmax(x_projection) # x軸射影のピーク強度のインデックスを計算
    peak_y_index = np.argmax(y_projection) # y軸射影のピーク強度のインデックスを計算
    peak_x_wavelength = x[peak_x_index] # ピーク強度となるx軸の波長
    peak_y_wavelength = y[peak_y_index] # ピーク強度となるy軸の波長
    print(f"x軸射影のFWHM: {fwhm_x:.2f} nm, ピーク波長: {peak_x_wavelength:.2f} nm") # FWHMとピーク波長を両方出力
    print(f"y軸射影のFWHM: {fwhm_y:.2f} nm, ピーク波長: {peak_y_wavelength:.2f} nm") # FWHMとピーク波長を両方出力
else:
    print("FWHMを計算できませんでした。")


# k_s と k_i を周波数で微分したものを計算し、|k_s' - k_i'| * L を計算
# 中心波長付近での値を計算する
c = 2.99792458e17  # nm/s (光速)

# 中心波長での角周波数を計算
omega_s_center = 2 * np.pi * c / (CWL * 2)
omega_i_center = 2 * np.pi * c / (CWL * 2)

# 数値微分を用いて k_s' と k_i' を計算 (中心差分)
delta_omega = 1e10  # 微小変化 (適宜調整)

# シグナル光の中心波長付近での屈折率の微分
n_s_center_plus = n_s_0((2 * np.pi * c) / (omega_s_center + delta_omega)) + delta_n_s((2 * np.pi * c) / (omega_s_center + delta_omega))
n_s_center_minus = n_s_0((2 * np.pi * c) / (omega_s_center - delta_omega)) + delta_n_s((2 * np.pi * c) / (omega_s_center - delta_omega))
d_n_s_d_omega = (n_s_center_plus - n_s_center_minus) / (2 * delta_omega)

# アイドラ光の中心波長付近での屈折率の微分
n_i_center_plus = n_i_0((2 * np.pi * c) / (omega_i_center + delta_omega)) + delta_n_i((2 * np.pi * c) / (omega_i_center + delta_omega))
n_i_center_minus = n_i_0((2 * np.pi * c) / (omega_i_center - delta_omega)) + delta_n_i((2 * np.pi * c) / (omega_i_center - delta_omega))
d_n_i_d_omega = (n_i_center_plus - n_i_center_minus) / (2 * delta_omega)

# k_s' と k_i' の計算 (ω/c * n + ω/c * dω/dn = 1/c * (n + ω * dn/dω))
k_s_prime = (n_s_center + omega_s_center * d_n_s_d_omega) / c
k_i_prime = (n_i_center + omega_i_center * d_n_i_d_omega) / c

# |k_s' - k_i'| * L の計算
abs_k_prime_diff_L = abs(k_s_prime - k_i_prime) * L_nm

print(f"HOM干渉の三角形幅(コヒーレンス時間(ps)): {abs_k_prime_diff_L*1e12:.16f}")
print(f"HOM干渉の三角形幅(コヒーレンス長(μm)): {c*abs_k_prime_diff_L*1e-3:.16f}")

plt.show() 
