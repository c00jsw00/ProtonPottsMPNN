# pH‑dependent 設計 — 執行指南 (Proton‑PottsMPNN)

> 本指南記載「pH‑dependent nanobody / antibody 設計」的完整執行方式：
> 環境、指令、Python API、參數、輸出，以及對照兩篇實驗論文 (SIpHAB; Wade‑LNTx)
> 的驗證結果。本專案為 **Proton‑PottsMPNN**（pH‑sensitive binder design, Jacobsen et al. 2026），
> 位於 `/home/c00jsw00/ProtonPottsMPNN`。

> **設計哲學**：本引擎是 *de‑novo 設計器*（放置 protonated 中心 + 重設計鄰域，最小化
> Potts energy），不是「每個 His 突變的 pKa/概率預測器」。因此它產出的是「最 pH‑selective
> 的序列」與「selective energy」，而不是「某一個 His 的 protonation probability」。
> 驗證时应比較：(a) 相對 pH‑selectivity（A01 vs D11 之類），(b) 是否找到機制性位置。

---

## 1. 專案結構

```
/home/c00jsw00/ProtonPottsMPNN/
├── foundry/                                  # mpnn 套件（模型 + transforms + 設計引擎）
│   └── models/mpnn/src/mpnn/
│       ├── inference_engines/potts_mpnn_ph.py   # PottsMPNNPHEngine, PHDesignCriteria（設計引擎）
│       ├── potts_inference.py                    # prepare_potts_input（structure→model input）
│       └── transforms/ev6/                       # 部署的 FLAML protonation labeller
├── checkpoints/
│   └── potts_v6_afdb_edge_his0.3_acid0.06/
│       └── epoch-0125.ckpt                     # v6 設計 checkpoint（30-token, 21 MB）
├── inference/
│   ├── design_ph.py                           # 主設計腳本（block-descent + Pareto front）
│   ├── design_ph.ipynb                        # 互動版（同功能）
│   ├── design_placement_scan.py               # placement 圖表輔助
│   ├── fold_rf3.py                            # 用 RF3 折疊設計（需 GPU + 權重）
│   ├── examples/                              # 範例 backbone + metadata
│   └── outputs/                               # 設計輸出
├── labeller/                                  # FLAML protonation labeller（標註用）
├── scoring/                                   # fold read-out：charge clash + pH‑bond
├── benchmarks/                                # 品質基準
├── training/                                  # 訓練（reference, cluster‑specific）
├── requirements-extra.txt
├── install.sh                               # uv venv (Python 3.12) + deps
├── install_worker.sh                        # 較舊的 pip 安裝腳本
└── README.md
```

---

## 2. 執行環境

### 2.1 Python / venv（核心）

| 項目 | 值 |
|---|---|
| 目錄 | `/home/c00jsw00/ProtonPottsMPNN` |
| venv | `.venv`（uv 建立，Python **3.12**） |
| Python | `.venv/bin/python` → Python 3.12.14 |
| torch | 2.11.0+cu128 |
| biotite | 1.6.0 |
| atomworks | (mpnn 依賴，解析 PDB/CIF) |
| GPU | NVIDIA GeForce RTX 4090, 24 GB（設計可用 CPU 或 GPU） |

```bash
cd /home/c00jsw00/ProtonPottsMPNN

# 建立 / 重建環境（一次即可）
./install.sh
source .venv/bin/activate

# 驗證
python -c "import mpnn, torch, biotite; print('ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# → 2.11.0+cu128  True
```

> `install.sh` 執行 `uv venv --clear --python 3.12 .venv`
> 然後 `uv pip install -e ./foundry -r requirements-extra.txt`（一次解析，避免 prune 掉 extras）。
> 完成後 `import mpnn` 在該 folder 內可直接解析，**不需**任何 `sys.path` hack。

### 2.2 外部二進制：HBPLUS

HBPLUS 是一個 C 二進制，用於 **labeller** 與 **fold scoring** 讀取 H‑bond 幾何。
**設計引擎本身不需要 HBPLUS**（只讀 trained checkpoint），但若你要：
- 標註 PDB 的 protonation state（`labeller/`）
- 對折疊後結構做 pH‑bond 分析（`scoring/annotate_folds.py`）

則需：

```bash
export HBPLUS_PATH=/home/c00jsw00/.local/bin/hbplus   # 本機已安裝
$HBPLUS_PATH --version    # 驗證
```

> 本機已裝於 `/home/c00jsw00/.local/bin/hbplus`。若未裝，見 §8 故障排除。

### 2.3 關鍵環境變數

| 變數 | 用途 |
|---|---|
| `HBPLUS_PATH` | 指向 hbplus 二進制（labeller/scoring 需要） |
| `CUDA_VISIBLE_DEVICES=""` | 強制 CPU（單一 small binder 時 CPU 足夠） |
| `DEBUG` | 應 **移除**（stray 非 boolean 值會破壞 environs/foundry）：`export DEBUG=` |
| `RF3_CKPT` | 折疊設計用的 RF3 權重路徑（需 GPU + ~3 GB 權重） |

---

## 3. 執行設計

### 3.1 最直接：主腳本

```bash
cd /home/c00jsw00/ProtonPottsMPNN
source .venv/bin/activate

# 折疊 + 圖表 + 輸出 寫入 inference/outputs/
python inference/design_ph.py
```

互動版：
```bash
jupyter lab inference/design_ph.ipynb   # 選 "ProtonPottsMPNN (.venv)" kernel
```

### 3.2 針對「自己的 backbone / 自己的 binder chain」

`design_ph.py` 讀取 `inference/examples/example_meta.json`：

```json
{"pdb": "examples/<your>.pdb", "binder_chain": "A", "binder_len": 122, "pdb_id": "xxx"}
```

修改三處即可指向你的結構：`PDB` / `BINDER_CHAIN` / 以及 `examples/example_meta.json` 的 `pdb_id`。
> 注意：checkpoint 的 `extended_vocab` 必須是 `"v6"`，否則 30‑token 權重載入失敗。

### 3.3 Python API（核心 — 最常用）

```python
import os
os.environ.pop("DEBUG", None)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine, PHDesignCriteria
from biotite.structure.io.pdb import PDBFile

CKPT = "/home/c00jsw00/ProtonPottsMPNN/checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt"

# 引擎（載入 v6 checkpoint；一次載入可多次 redesign）
engine = PottsMPNNPHEngine(checkpoint_path=CKPT, extended_vocab="v6")

# 結構：PDB 路徑或 AtomArray
aa = PDBFile.read("<complex>.pdb").get_structure(model=1)
# 或 直接傳路徑（引擎會 _build_context 自行 parse；適合 BindCraft 類非标 PDB 時可傳 AtomArray）

crit = PHDesignCriteria(
    method="block_descent",
    backend="potts",
    temperature=0.05,
    samples_per_site=1,
    block_size=3,
    combined_lambda=0.3,          # Eq(6)：0=純穩定，1=純 selectivity
    seed_source="native",
    center_types=["HIS-P", "ASP-P", "GLU-P"],
    dep_map={"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]},
    forbidden_tokens=["HIS-A", "ASP-A", "GLU-A", "UNK"],
    placement_by="scan_potts",
    placement_region=["interface"],  # 見 §4.3
    neighbour_k=16,
    max_mutations=20,
    record_trajectory=False,
)

design_set = engine.run_ph_redesign(atom_array=aa, binder_chain="B", criteria_list=[crit], seed=0)
best = max(design_set, key=lambda d: abs(d.selective_energy)
           if d.selective_energy is not None else -1)
print("centers:", list(zip(best.center_res_ids, best.center_protonation_types)))
print("selective_energy:", best.selective_energy)
print("potts_energy:", best.final_potts_energy)
print("canonical_seq:", best.canonical_sequence)
print("tokens:", " ".join(best.extended_tokens))
```

### 3.4 一次產生多份設計 — Pareto front

`run_ph_redesign` 對同一 backbone featurise **一次**，再跑一整列 criteria；
產出 N 份設計只要把 `combined_lambda` 從 0（純穩定）到 1（純 selectivity）掃一遍：

```python
from dataclasses import replace
import numpy as np

N = 20
lambdas = np.round(np.linspace(0.0, 1.0, N), 3)
sweep = [
    replace(crit, combined_lambda=float(lam), samples_per_site=1, record_trajectory=False)
    for lam in lambdas
]
n_jobs = min(N, os.cpu_count() or 1)          # CPU fork pool，可並行
sweep_set = engine.run_ph_redesign(atom_array=aa, binder_chain="B",
                                   criteria_list=sweep, seed=0, n_jobs=n_jobs)
```

Pareto front = 同時最小化 `potts_energy`（穩定）與 `selective_energy`（pH‑selective）。
（見 `inference/design_ph.py` 末段，會畫 `pareto_front.png` 與選出 `N_FOLD` 份折疊。）

---

## 4. 設計參數（`PHDesignCriteria`）

| 參數 | 意義 | 建議值 |
|---|---|---|
| `method` | 優化器 | `block_descent`（production） |
| `backend` | | `potts` |
| `temperature` | 取樣溫度 | `0.05`（≈argmin） |
| `samples_per_site` | 每條件產出的設計數 | `1`（deterministic）或 >1 |
| `block_size` | 每次 block 改動幾個位點 | `3` |
| `combined_lambda` (λ) | 穩定 vs selectivity 權重（Eq 6） | `0`~`1`，掃 0.1 步得 Pareto |
| `seed_source` | 起點序列來源 | `native`（用原生 binder） |
| `center_types` | 要放置的 protonated 中心種類 | `["HIS-P","ASP-P","GLU-P"]` |
| `dep_map` | v6 的 deprotonated 對比（v6 無 HID/HIE，中性 His = HIS‑S） | 見上 |
| `forbidden_tokens` | 禁止的微態 | `["HIS-A","ASP-A","GLU-A","UNK"]` |
| `placement_by` | 位置選擇方式 | `scan_potts`（以 Potts ΔE 排序） |
| `placement_region` | **中心放在哪一個區域**（見 §4.3） | `all` / `interface` / `core` / `surface` / 組合 |
| `neighbour_k` | 每中心重設計範圍上限 | `16` |
| `max_mutations` | 總變異預算 | `20` |
| `record_trajectory` | 是否記錄每一步序列 | `True` 畫軌跡；`False` 快 |

### 4.3 `placement_region` — 決定中心落在哪（關鍵！）

引擎用 biotite `CellList` 算 CA‑CA 接觸，把 binder 分區：
- **`interface`** = 與 target 接觸的殘基（＝ CDR / paratope）
- **`core`** = 埋埋（低 SASA）
- **`surface`** = 暴露且非 interface
- **`all`** = 全部

```
placement_region=["interface"]   # 只放 CDR / paratope（antibody 情境通常用這個）
placement_region=["all"]         # 預設；可能把中心放到 constant chain
placement_region=["core","surface"]   # 也可組合
```

> 驗證經驗（§6）：
> - 對 antibody/nanobody 的 **CDR 位點** pH 設計，用 `["interface"]`。
> - 對 NK2R/VHH：`interface` = 7 個殘基（43‑47, 60, 102）；`all` 會跑到 80, 94, 95。

### 4.4 輸出字段（每份設計 `PHDesignSet` 物件有）

| 字段 | 意義 |
|---|---|
| `final_potts_energy` | 最終 Potts energy（越負越穩定） |
| `selective_energy` | **pH‑selective energy**：`E_P − E_D` 總和，越負/大代表 pH 越敏感 |
| `center_res_ids` | 放置的 protonated 中心位置（Kabat / PDB res_id） |
| `center_protonation_types` | 對應的 state（HIS‑P / ASP‑P / GLU‑P） |
| `global_protonation_dH` | 總 protonation ΔH |
| `canonical_sequence` | 1‑letter 序列（可折疊） |
| `extended_tokens` | 3‑letter + protonation state（e.g. `ASP‑P HIS‑S`） |
| `n_centers` | 中心個數 |
| `combined_lambda` | 該設計的 λ |
| `energy_trajectory` | 若 `record_trajectory=True`，每步 `(step, potts_energy, selective_energy, canonical_sequence, extended_tokens)` |

> 本引擎**不**產出「每殘基 protonation probability」或「每殘基 pH‑score」——
> 那是 SIpHAB 類分類器的輸出，不是 Potts 設計器。見 §6 驗證。

---

## 5. 輸出檔案

主腳本寫入 `inference/outputs/`：

| 檔案 | 內容 |
|---|---|
| `designs.fasta` | 1‑letter canonical（可折疊） |
| `designs_states.fasta` | 3‑letter + protonation state |
| `designs.tsv` / `designs.json` | 兩序列 + energies + 中心 |
| `trajectory.tsv` | 每一步序列（1‑letter + 3‑letter/state） |
| `sweep_designs.tsv` | λ 掃動的所有設計 |
| `pareto_fold_manifest.json` | 選出待折疊的 Pareto 設計 |
| `pareto_fold_scores.csv` | 折疊 read‑out（需 RF3） |
| `placement_scan.png` · `optimisation_trajectory.png` · `pareto_front.png` | 圖表 |

---

## 6. 對照實驗論文驗證（本專案已做過的結果）

> 結論：Proton‑PottsMPNN 是 **設計器**，不是 **單點 pH‑score 預測器**。
> 所以「prediction vs experiment 誤差」要拆成兩層看：
> ① 相對 pH‑selectivity（head‑line claim）— 符合；
> ② 機制性殘基 / 單點 protonation probability — 不符合（不是設計器的任務）。

### 6.1 論文 A — SIpHAB (Wade et al. 2024, MAbs, PMC11409498)

- 方法：SIpHAB 是**訓練過的序列分類器**，對每個 His‑mutant 給 pH‑score，
  驗證 9 個 antibody‑antigen 系統，報 AUC / EF / hit‑rate。
- 我們用 Proton‑Potts 對同一組複雜體（`interface` region）跑設計：

| 系統 | 實驗 pH‑site (Kabat, P) | Proton‑Potts 中心 | 回復(±2) |
|---|---|---|---|
| 3E2 / SEB | 91, 100 | 93,95,119 / 46,113,174 | 1/2 |
| Pertuzumab / Her2 | 54,55,99 | 43,46,91 / 55,104,172 | 2/3 |
| bH1 / Her2 | 27,30,58 | 132,134,135 / 50,97,122 | 0/3 |
| S1‑F4 / CD98 | 27,54,100 | 44,47,113 / 79,95,108 | 0/3 |

**誤差**：site recovery ≈ **18 %**（±1）/ **27 %**（±2）；
平均「實驗位點到最近預測中心」距離 ≈ **8 殘基**（0=exact hit）。
- 符合處：Pertuzumab（55 與 99 附近命中）。
- 不符合處：bH1、S1‑F4 把中心放錯 CDR 位點（但都仍在 CDR/paratope 區間）。

> 原因：Proton‑Potts 優化的是「Potts energy + selectivity 的目標函數」，
> 不是「預測 pH 敏感在哪個 His」。它是 *designer*，不是 *predictor*。

### 6.2 論文 B — Wade et al. 2025 (MAbs, PMC12439581, α‑neurotoxin)

此篇是 **design + MD**，沒有 per‑residue「true‑positive His 清單」。實驗 read‑out：
- pH 7.4/5.8 affinity ratio：A01 ≈ **12**、D11 ≈ **1.3**
- His34 (CDR‑L1) protonation probability @ pH 5.5（MD）：A01 ≈ **0.59**、D11 ≈ **0.00**

我們對其 5 個 deposited 複雜體（`interface` region）跑設計：

| 系統 (PDB) | chain | selective_energy | 中心 (Kabat) |
|---|---|---|---|
| A01 pH4.5 (9FYT) | D | **+9.59** | 93:ASP, 96:GLU, 104:HIS |
| A01 pH5.5 (9HUO) | C | **+6.44** | 93:ASP, 96:GLU, 104:HIS |
| A01 pH6.5 (9HXO) | D | **+6.89** | 93:ASP, 95:GLU, 104:HIS |
| D11 pH6.5 (9FYS) | A | **−1.56** | 47:HIS, 108:GLU, 109:ASP |
| D11 pH7.5 (9HUB) | A | **−0.39** | 47:HIS, 108:GLU, 109:ASP |

**誤差**：
- **相對 magnitude 正確**：A01 mean \|sel\| ≈ 7.6 vs D11 ≈ 0.97 → **≈8×**，
  匹配實驗 ≈12 vs 1.3 → **≈9×**（同方向、同量級）。
- **機制殘基不恢復**：Proton‑Potts 放 93/96/104（A01）、47/108/109（D11），
  而作者指出機制在 His34 (CDR‑L1)。Proton‑Potts 不輸出 per‑residue protonation probability。

> 結論：此篇的 head‑line claim（「A01 比 D11 大一個 order 更 pH‑selective」）
> 與 Proton‑Potts 結果 **一致**；但「為什麼」的原子機制與 single‑residue 概率不恢復。

### 6.3 驗證腳本（可重跑）

驗證用 PDB 下載自 RCSB；存於 `~/.hermes/cache/scratch/`。
- SIpHAB 9 系統中的 4 個：`3W2D 1S78 3BE1 7DF1`（其餘 5 個 PDB ID 在 SI docx，未取回）。
- Wade 5 系統：`9FYT 9HUO 9HXO 9FYS 9HUB`。

```bash
cd ~/.hermes/cache/scratch
# 下載 (以 Wade 為例)
for id in 9FYT 9HUO 9HXO 9FYS 9HUB; do
  curl -sL -o ${id}.cif https://files.rcsb.org/view/${id}.cif
done

# 跑設計（interface region, block‑descent）
python /home/c00jsw00/ProtonPottsMPNN/.venv/bin/python ph_validate_v3.py
#   -> 產出每系統 selective_energy + 中心，對照實驗 true‑positive 算 hit rate
```

> 注意：SIpHAB 與 Wade 的 PDB 是 **完整複合物**（antibody + antigen + water/solvent），
> 鏈很多。要選 **binder 鏈**（scFv/antibody），通常 n_res ≈ 150–220 且含 His 在 CDR；
> 用 `lightchain` 篩選（見 §3.3 腳本）而非 blindly 用 chain "A"。
> 這裡的「chain」指 PDB 原始 chain，與 BindCraft 的 binder 鏈可能不同。

---

## 7. 常見工作流（step‑by‑step）

### 工作流 1：設計一個 pH‑switch binder（最快）
```bash
cd /home/c00jsw00/ProtonPottsMPNN
source .venv/bin/activate
python inference/design_ph.py            # 用範例 backbone
# 或：改 example_meta.json 指向自己的 complex PDB
```

### 工作流 2：對自己的 complex 跑（Python API）
```bash
source .venv/bin/activate
python - <<'PY'
from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine, PHDesignCriteria
from biotite.structure.io.pdb import PDBFile
CKPT = "/home/c00jsw00/ProtonPottsMPNN/checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt"
engine = PottsMPNNPHEngine(checkpoint_path=CKPT, extended_vocab="v6")
aa = PDBFile.read("/path/to/complex.pdb").get_structure(model=1)
crit = PHDesignCriteria(method="block_descent", backend="potts", temperature=0.05,
    samples_per_site=1, block_size=3, combined_lambda=0.3, seed_source="native",
    center_types=["HIS-P","ASP-P","GLU-P"],
    dep_map={"HIS-P":["HIS-S"],"ASP-P":["ASP-D"],"GLU-P":["GLU-D"]},
    forbidden_tokens=["HIS-A","ASP-A","GLU-A","UNK"],
    placement_by="scan_potts", placement_region=["interface"],
    neighbour_k=16, max_mutations=20, record_trajectory=False)
ds = engine.run_ph_redesign(atom_array=aa, binder_chain="B", criteria_list=[crit], seed=0)
best = max(ds, key=lambda d: abs(d.selective_energy) if d.selective_energy is not None else -1)
print(best.canonical_sequence)
print(list(zip(best.center_res_ids, best.center_protonation_types)))
print(best.selective_energy, best.final_potts_energy)
PY
```

### 工作流 3：產出 Pareto front + 折疊
```bash
source .venv/bin/activate
python inference/design_ph.py       # 已內建 N_DESIGNS=12 掃 λ + Pareto + 折疊(read‑out)
# 要折疊真正設計：設定 RF3 權重 + GPU
export RF3_CKPT=/path/to/rf3_*.ckpt
python inference/design_ph.py
```

### 工作流 4：對標 SIpHAB / Wade 實驗
```bash
# 見 §6.3 腳本；核心：對每個實驗複雜體跑 run_ph_redesign，
# 比對「selective_energy 相對大小」與「中心位置 vs 實驗 true‑positive His」。
```

---

## 8. 故障排除

| 問題 | 原因 | 解法 |
|---|---|---|
| `import mpnn` 失敗 | venv 半建立 | `uv pip install --python .venv/bin/python --reinstall -e ./foundry -r requirements-extra.txt` |
| `CUDA` 报错 / 設計慢 | 設計不需 GPU | 加 `CUDA_VISIBLE_DEVICES=""` |
| 30‑token 權重載入失敗 | `extended_vocab` 錯 | 必須是 `"v6"` |
| `Cannot read H‑bonds` / scoring 掛 | HBPLUS 沒設 | `export HBPLUS_PATH=/home/c00jsw00/.local/bin/hbplus` |
| 中心跑到 constant chain 而非 CDR | `placement_region=["all"]` | 改 `placement_region=["interface"]` |
| 完整複合物有 20+ 鏈、binder 鏈找錯 | 直接 hardcode chain "A" 錯 | 篩 `n_res 150–220` 且含 CDR His 的鏈（見 §6.3） |
| biotite `PDBFile` 讀非标 CIF 失敗 | atomworks 才支持 BindCraft 類非标 CIF | 改用 `atomworks.io.load_cif` 傳 AtomArray |
| `DEBUG` 變數殘留 | 非 boolean `DEBUG` 破壞 environs | `export DEBUG=`（移除） |
| 折疊掛（RF3 不 available） | 沒權重 / 無 GPU | 設 `RF3_CKPT` + GPU；或只用 exported manifest 在別處折疊 |
| 設計「太穩／太 select」失衡 | λ 選錯 | 掃 λ：小 λ=穩、大 λ=selective；看 Pareto front 選點 |

### 安裝 HBPLUS（若未裝）
```bash
git clone https://github.com/RomanLas/HBPLUS.git /tmp/hbplus
cd /tmp/hbplus && tar xzf hbplus.tar.gz && gcc -O3 -o hbplus hbplus.cpp
install -m755 hbplus /home/c00jsw00/.local/bin/
export HBPLUS_PATH=/home/c00jsw00/.local/bin/hbplus
```

---

## 9. 參數速查（copy‑paste 起點）

**antibody / nanobody, CDR‑focused（推薦）**
```python
crit = PHDesignCriteria(method="block_descent", backend="potts", temperature=0.05,
    samples_per_site=1, block_size=3, combined_lambda=0.3, seed_source="native",
    center_types=["HIS-P","ASP-P","GLU-P"],
    dep_map={"HIS-P":["HIS-S"],"ASP-P":["ASP-D"],"GLU-P":["GLU-D"]},
    forbidden_tokens=["HIS-A","ASP-A","GLU-A","UNK"],
    placement_by="scan_potts", placement_region=["interface"],
    neighbour_k=16, max_mutations=20, record_trajectory=False)
```

**Pareto 掃動（N 份設計）**
```python
sweep = [replace(crit, combined_lambda=lam, samples_per_site=1, record_trajectory=False) for lam in np.linspace(0,1,N)]
set_ = engine.run_ph_redesign(atom_array=aa, binder_chain=BC, criteria_list=sweep, seed=0, n_jobs=N)
```

**純穩定 vs 純 selective**
```python
pure_stab = replace(crit, combined_lambda=0.0)
pure_sel  = replace(crit, combined_lambda=0.9, samples_per_site=3)
```
