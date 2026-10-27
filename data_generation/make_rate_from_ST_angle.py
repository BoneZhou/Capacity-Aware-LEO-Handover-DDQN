# make_rates_from_ST_angle_filtered.py
# 讀 ST_angle.csv（time x sats 的仰角矩陣；不可視 = INVISIBLE_VALUE），
# 1. 先剃掉「全程未連線」的衛星（這裡定義為：全程 angle 都等於 INVISIBLE_VALUE）
# 2. 再依鏈路模型：仰角 -> 斜距 -> FSPL -> SNR -> Shadowed-Rician 期望
# 3. 產生 input_angle.txt ： "new_sat_id(0-based) time per_user_rate_Mbps"

import math, random
from pathlib import Path

import pandas as pd

# ======== 路徑與格式 ========
CSV_PATH = Path("ST_angle.csv")     # 你的角度矩陣
OUT_PATH = Path("input_angle.txt")  # C++ 會讀這個（已改成 0-base 的 sat_id）
ANGLES_IN_RAD = True                # True=CSV 為弧度；False=CSV 為度
INVISIBLE_VALUE = -1                # 不可視 / 未連線 的標記

# ======== 幾何常數 ========
R_E = 6371e3          # 地球半徑 (m)
H_SAT = 1300e3        # 軌道高度 (m)，若要 1000 km，改成 H_SAT = 1000e3

# ======== 鏈路收支（上行 Ka 基準組，可自行調整） ========
FREQ_HZ   = 28.0e9    # 28 GHz
B_HZ      = 5.0e8     # 500 MHz
EIRP_dBW  = 50.0      # 地面端等效全向輻射功率
G_r_dBi   = 40.0      # 衛星接收天線增益
Tsys_K    = 400.0
Lmisc_dB  = 3.0       # 大氣/極化/指向/硬體等總額外損耗

# ======== 通道模型（Shadowed-Rician） ========
USE_SHADOWED_RICIAN = True   # True=做期望；False=用簡化 log2(1+gamma)
K_dB  = 10.0                 # K 因子
m_sh  = 1.5                  # 形狀參數
Omega = 1.0                  # LoS 平均功率歸一化
MC_SAMPLES = 200             # 蒙地卡羅樣本數

# ======== 其他設定 ========
PRINT_PROGRESS_EVERY = 10    # 每處理幾顆「保留後的衛星」印一次進度

# ======== 物理常數 ========
c = 299_792_458.0
k_B = 1.380649e-23
lam = c / FREQ_HZ
N_lin = k_B * Tsys_K * B_HZ  # 雜訊功率 (W)

def db_to_lin(db):  return 10.0**(db/10.0)
def lin_to_db(x):   return 10.0*math.log10(max(x, 1e-300))

def elev_to_slant_range(elev_rad: float) -> float:
    """
    由地表仰角 -> 斜距 s（簡化幾何；地面於半徑 R_E，衛星於 R_E+H_SAT 的圓軌）
      s = -R_E*sin(e) + sqrt( (R_E+H)^2 - (R_E*cos(e))^2 )
    """
    e = elev_rad
    term = (R_E + H_SAT)**2 - (R_E * math.cos(e))**2
    if term <= 0:
        return float('inf')
    return -R_E * math.sin(e) + math.sqrt(term)

def fspl_db(slant_m: float) -> float:
    return 20.0 * math.log10(4.0 * math.pi * slant_m / lam)

# --- Shadowed-Rician 期望 ---
def sample_shadowed_rician_power(K_lin: float, m: float, Omega: float) -> float:
    # 近似生成：Rice + 乘性 Gamma 陰影（mean=1），研究/模擬足夠
    s = math.sqrt((K_lin * Omega) / (K_lin + 1.0))      # LoS 振幅均值
    sigma2 = Omega / (2.0 * (K_lin + 1.0))              # 散射分量功率/2
    x = random.gauss(0.0, math.sqrt(sigma2))
    y = random.gauss(0.0, math.sqrt(sigma2))
    I = (x + s)**2 + y**2                               # Rice 瞬時功率
    # 乘性陰影：Gamma(m, 1/m) -> mean = 1
    g = random.gammavariate(m, 1.0/m)
    return I * g

def expected_log2_1p_gamma_shadowed_rician(gamma_lin: float) -> float:
    if not USE_SHADOWED_RICIAN:
        return math.log2(1.0 + gamma_lin)
    K_lin = db_to_lin(K_dB)
    acc = 0.0
    for _ in range(MC_SAMPLES):
        g = sample_shadowed_rician_power(K_lin, m_sh, Omega)
        acc += math.log2(1.0 + gamma_lin * g)
    return acc / MC_SAMPLES

def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"找不到 {CSV_PATH}")

    # 讀角度矩陣
    df = pd.read_csv(CSV_PATH, header=0)
    # 不管原本 header 寫什麼，一律用 0..(ncol-1) 做「舊衛星 index」
    df.columns = list(range(df.shape[1]))

    T = df.shape[0]        # 時間長度
    S = df.shape[1]        # 原始衛星數

    print(f"[INFO] 原始 angle matrix: T={T}, S={S}")

    # ================================
    # 1) 剃掉「全程未連線」的衛星
    #    定義：該 column 所有值都等於 INVISIBLE_VALUE（例如 -1）
    # ================================
    mat = df.values  # shape (T, S)
    # True 表示這一顆在某個時間點有「非 INVISIBLE_VALUE」的角度
    ever_connected_mask = (mat != INVISIBLE_VALUE).any(axis=0)
    kept_old_indices = [i for i in range(S) if ever_connected_mask[i]]

    print(f"[INFO] 有曾經連線過的衛星數量 = {len(kept_old_indices)}")
    print(f"[INFO] 全程未連線、被剃除的衛星數量 = {S - len(kept_old_indices)}")

    if len(kept_old_indices) == 0:
        raise SystemExit("[ERROR] 沒有任何衛星曾連線過，無法產生輸出。")

    # 只保留有連線過的衛星
    df_kept = df[kept_old_indices]
    # 重新排成 matrix (T, S_kept)
    angle_kept = df_kept.values
    T_kept, S_kept = angle_kept.shape

    print(f"[INFO] 剔除後 matrix: T={T_kept}, S_kept={S_kept}")
    print("[INFO] 0-base 新編號 new_sat_id 會是 0 ~", S_kept - 1)

    # 若你想留一份 old_sid -> new_sid 的 map，可以加寫一個 csv（選用）
    # 這裡只是印出來給你看：
    for new_sid, old_sid in enumerate(kept_old_indices):
        print(f"[MAP] old_sid={old_sid} -> new_sid={new_sid}")

    # ================================
    # 2) 對保留後的衛星做 angle -> throughput 轉換
    #    並輸出 new_sat_id（0-based）
    # ================================
    with OUT_PATH.open("w", newline="") as out:
        w = out.write

        for new_sid, old_sid in enumerate(kept_old_indices):
            if new_sid % PRINT_PROGRESS_EVERY == 0:
                print(f"[INFO] 處理保留後衛星 new_sid={new_sid}/{S_kept-1} (old_sid={old_sid})...")

            col = df[old_sid].values  # 長度 T 的仰角序列（原始 column）
            for t in range(T):
                elev_val = col[t]

                # 不可視 / 未連線 → rate=0
                if elev_val == INVISIBLE_VALUE or pd.isna(elev_val):
                    w(f"{new_sid} {t} 0.000000\n")
                    continue

                # 單位處理：弧度 or 度
                if ANGLES_IN_RAD:
                    elev_rad = float(elev_val)
                else:
                    elev_rad = math.radians(float(elev_val))

                # 極低仰角保護
                if elev_rad <= 0:
                    w(f"{new_sid} {t} 0.000000\n")
                    continue

                # 幾何 -> 斜距 -> FSPL -> SNR
                s_m = elev_to_slant_range(elev_rad)
                if not math.isfinite(s_m) or s_m <= 0:
                    w(f"{new_sid} {t} 0.000000\n")
                    continue

                FSPL_dB = fspl_db(s_m)
                Pr_dBW = EIRP_dBW + G_r_dBi - FSPL_dB - Lmisc_dB
                gamma_lin = db_to_lin(Pr_dBW) / N_lin  # (dBW->W) / W

                # 通道期望容量（bits/s/Hz）
                ccap = expected_log2_1p_gamma_shadowed_rician(gamma_lin)

                # 轉成 bps，再轉 Mbps
                rate_bps = B_HZ * ccap
                rate_mbps = rate_bps / 1e6

                w(f"{new_sid} {t} {rate_mbps:.6f}\n")

    print("✅ 完成輸出：", OUT_PATH)
    print("   格式： new_sat_id(0-based)  time  per_user_rate_Mbps")
    print(f"   參考：原始 S={S}，保留後 S_kept={S_kept}，T={T}")


if __name__ == "__main__":
    main()
