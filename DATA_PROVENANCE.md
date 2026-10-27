# 資料來源與公開前確認事項

本 Repository 保留研究程式實際讀取的兩份衍生模擬輸入：

- `input.txt`：50 顆可見衛星的區間與每顆容量。
- `input_angle.txt`：50 顆衛星、3599 個 time slots 的 achievable-rate table。

`input_angle.txt` 可由 `data_generation/make_rate_from_ST_angle.py` 從 `ST_angle.csv` 產生；目前 Repository 沒有附 `ST_angle.csv`。原專案另有約 120 MB 的 `Telesat/ECEF` 軌跡與一支沒有授權聲明、含本機路徑的 `NTN_simulation.py`。由於無法確認其來源與再散布權，兩者沒有放入公開版。

在公開上傳前，資料擁有人仍須確認：

1. `input.txt` 與 `input_angle.txt` 是否為研究團隊自行生成並可公開再散布。
2. `ST_angle.csv`、原始 ECEF 軌跡及星座參數的來源與授權。
3. 若要完整重建軌跡到 rate table 的流程，應提供已獲授權的原始軌跡／角度資料及產生設定。

本檔不是授權聲明，也沒有替任何第三方資料授權。

