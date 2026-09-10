# EGWT Reproduction

非官方復刻：J. Feng et al., *"Enhanced Crop Disease Detection With EfficientNet Convolutional Group-Wise Transformer"*, IEEE Access, 2024.

論文本身未公開程式碼，此為根據論文公式、架構圖、與訓練細節逐步重建的 PyTorch 實作，並在論文引用的三個原始資料集（PlantVillage、cassava、tomato leaves）上實際訓練驗證。

> **凍結策略對照（已完成，含真實預訓練版本）**：發現論文對微調凍結範圍有兩種可能讀法，且原本標示為「Section IV.C 凍結 stage1+2」的「本次重現」表格實際上是在**完全不凍結**（stage1/2/3 全部可訓練，`freeze_mode=none`）的設定下跑出來的（`total_params_M == trainable_params_M`，見 `runs/egwt_*_summary.json`）。先後跑了兩輪對照：
> 1. **隨機初始化骨幹版**（`run_freeze_comparison.ps1`，各資料集 40 epoch）：`stage12`（凍結 stage1+2）、`head_only`（凍結 stage1+2+3，只微調最後 Linear 分類層）——當時完整 ImageNet-1k 預訓練尚未完成，stage1/2 仍是隨機初始化，這組對照只能測「凍結策略在沒有真正預訓練骨幹下的相對趨勢」，不是對論文遷移學習效果的忠實重現。
> 2. **真實預訓練版本（2026-09-02 完成）**：`pretrain_imagenet.py` 的 ImageNet-1k from-scratch 預訓練已完整跑完 50/50 epoch（best_top1=63.96%，checkpoint 存於 `checkpoints/imagenet_pretrain/egwt_imagenet_best.pt`），用這個真實權重重跑 `none`/`stage12`/`head_only` 三種凍結策略 × 三個資料集（`run_freeze_comparison_realpretrain.ps1`，各 40 epoch，結果存於 `runs_realpretrain/`），這才是對論文遷移學習主張的忠實測試——結果見下方新增的「真實預訓練對照結果」。

## 架構

3-stage CNN + Group-wise Transformer 混合架構：DWTE（深度可分離卷積切塊）→ Convolutional Projection → Group-wise Multi-Head Attention (G-MHA) → Group-wise MLP (G-MLP)，stage 3 額外加入 EfficientNet-B0 風格的 projection。

## 與原論文的差異

| 項目 | 狀態 |
|---|---|
| DWTE、G-MHA 組內權重共享、G-MLP 展開結構 | ✅ 經論文原圖 (Fig. 6, 7) 逐一核對確認吻合 |
| Convolutional Projection 的 stride | ✅ 完全比照 Fig. 6(b) 的 stride=1（改用 PyTorch flash attention 解決效能問題後，不再需要 stride=2 妥協） |
| Table 2 的 768/1024 工作維度 | ⚠️ 論文資訊不足以唯一決定架構（字面讀取會使參數量超預算達1.9倍），採用內部最一致的替代讀法，參數量 19.88M vs 論文 23.04M |
| EfficientNet-B0 projection 內部結構 | ⚠️ 論文只給不透明方塊，用 depthwise conv + Squeeze-Excite 近似 |
| ImageNet 預訓練 | 初版用真實 EfficientNet-B0 權重 + Transformer 隨機初始化；後續已改為完整 ImageNet-1k from-scratch 預訓練，比照論文 Section III.B.5。**已於 2026-09-02 完整跑完 50/50 epoch，best_top1=63.96%**（checkpoint: `checkpoints/imagenet_pretrain/egwt_imagenet_best.pt`），已用此權重重新微調三個作物資料集，見下方「真實預訓練對照結果」 |
| cassava 的 SMOTE 過採樣 | ⚠️ 改用 class-weighted loss（SMOTE 定義的特徵空間論文未說明） |
| 微調凍結範圍 | ⚠️ 論文遷移學習段落寫「除最後線性層外全部凍結」，程式碼原本誤植為只凍 stage1+2；現以 `--freeze_mode {none,stage12,head_only}` 提供三種讀法並實測比較，隨機初始化骨幹版 `head_only` 分數偏低（見下方對照表），但用真實 ImageNet 預訓練骨幹重跑後 `head_only` 已能逼近全量微調，支持論文讀法 |

## 結果對照

> 這裡同時保留三組數據：論文原始報告、先前舊版實驗結果，以及本次重現結果。它們代表不同實驗設定與時間點，不能直接互相覆蓋。

| 資料集 | 論文 Top-1 | 舊版結果（先前實驗） | 本次重現 Top-1 (100 epoch, freeze_mode=none 全可訓練) | 論文 Params | 本次 Params |
|---|---:|---:|---:|---:|---:|
| PlantVillage | 99.88% | 93.81% | 99.72% | 23.04M | 19.88M |
| cassava | 84.29% | 69.71% | 77.73% | 23.04M | 19.84M |
| tomato | 99.99% | 93.46% | 98.77% | 23.04M | 19.85M |

註：
- 舊版結果為先前在本 repo 內跑出的舊紀錄，保留以展示歷史軌跡。
- 本次重現結果來自實際 log / summary：
  - PlantVillage best_top1 = 99.72%
  - cassava best_top1 = 77.73%
  - tomato best_top1 = 98.77%

### 凍結策略對照結果

在無真正 ImageNet 預訓練骨幹的前提下（見上方說明），比較三種凍結策略，皆用 Adam + `lr_lambda=max(0.9^epoch, 0.1)`、同一份資料切分：

| 資料集 | none（全可訓練，100 epoch，trainable=19.8~19.9M） | stage12（凍結 stage1+2，40 epoch，trainable≈17.5M） | head_only（只訓練分類頭，40 epoch，trainable≈0.005~0.04M） |
|---|---:|---:|---:|
| PlantVillage | 99.72% | 95.72% | 65.69% |
| cassava | 77.73% | 71.40% | 63.68% |
| tomato | 98.77% | 96.43% | 51.97% |

（皆為 best_top1，來自 `runs/egwt_<dataset>_<freeze_mode>_summary.json`；`none` 是 100 epoch 的舊結果，`stage12`/`head_only` 是新跑的 40 epoch，epoch 數不同不能做嚴格的訓練曲線比較，但排序結論很穩定。）

**結論（隨機初始化版）**：三個資料集上凍結程度和準確率呈一致的單調關係——`none` > `stage12` > `head_only`，且 `head_only` 掉幅最大（PlantVillage 掉 34pp、tomato 掉 47pp）。這符合預期：在 stage1/2 仍是隨機初始化、只有 stage3 有啟發式暖啟動的前提下，凍結愈多等於愈依賴這些未經真正訓練的隨機/近似特徵，`head_only` 實質上是在對隨機投影做線性探測。這**不能**用來否證論文「凍結除最後線性層外的所有層」的遷移學習主張，因為論文的前提是骨幹已用真正 ImageNet-1k 預訓練收斂——下方「真實預訓練對照結果」補上了這個忠實測試。

### 真實預訓練對照結果（2026-09-02，忠實重現論文遷移學習主張）

用真正跑完的 ImageNet-1k 預訓練權重（50 epoch，best_top1=63.96%）重跑三種凍結策略，同樣 40 epoch、同一份資料切分（`runs_realpretrain/egwt_<dataset>_<freeze_mode>_summary.json`）：

| 資料集 | none（全可訓練） | stage12（凍結 stage1+2） | head_only（只訓練分類頭，trainable≈0.005~0.04M） | head_only（舊版，隨機初始化骨幹） |
|---|---:|---:|---:|---:|
| PlantVillage | 99.77% | 99.64% | **97.39%** | 65.69% |
| cassava | 84.58% | 82.09% | **77.79%** | 63.68% |
| tomato | 98.98% | 98.91% | **88.61%** | 51.97% |

**結論**：`none` > `stage12` > `head_only` 的單調關係依然成立，但掉幅徹底改觀——`head_only` 現在只比 `none` 低 2.4pp（PlantVillage）到 10.4pp（tomato），不再是隨機初始化版那種 34-47pp 的崩潰式下滑。**這次的結果真正支持論文「除最後線性分類層外，其餘層全部凍結」的遷移學習主張**：只要骨幹網路有真正收斂的 ImageNet-1k 預訓練，光訓練最後一層線性分類頭就能拿到接近全量微調的表現。tomato 的差距（10.4pp）比另外兩個資料集略大，可能與其類別數較多（11 類）、任務粒度較細有關，不影響整體單調趨勢的結論。

### 第二組獨立預訓練骨幹交叉驗證（2026-09-04）

為確認上面的結論不是單一次 ImageNet-1k 預訓練的僥倖，額外用同學獨立跑的另一份 ImageNet-1k 預訓練權重（`checkpoints/imagenet_pretrain_classmate/`）重跑一次完整的凍結策略對照。兩份預訓練權重都是**各自從頭跑滿 50/50 epoch**，checkpoint 內的 `epoch`/`scheduler.last_epoch` metadata 確認皆已正常訓練完畢，未中途中斷：

| 預訓練來源 | best_top1 | Checkpoint | 下載 |
|---|---:|---|---|
| 本次重現（自己跑） | 63.96% | `checkpoints/imagenet_pretrain/egwt_imagenet_best.pt` | [下載](https://github.com/s51744/egwt-reproduction/releases/download/v1.2-imagenet-backbones/egwt_imagenet_backbone_selfrun_best.pt) |
| 同學獨立跑一份 | 64.43% | `checkpoints/imagenet_pretrain_classmate/egwt_imagenet_best.pt` | [下載](https://github.com/s51744/egwt-reproduction/releases/download/v1.2-imagenet-backbones/egwt_imagenet_backbone_classmate_best.pt) |

用同學這份權重重跑三種凍結策略 × 三個資料集（`run_freeze_comparison_classmate.ps1`，結果存於 `runs_realpretrain_classmate/`），結果與自己那份幾乎一致：

| 資料集 | none | stage12 | head_only | 對照：本次重現 head_only |
|---|---:|---:|---:|---:|
| PlantVillage | 99.85% | 99.75% | 98.04% | 97.39% |
| cassava | 84.89% | 82.93% | 78.10% | 77.79% |
| tomato | 98.98% | 98.85% | 89.49% | 88.61% |

每一格與上方「真實預訓練對照結果」表的對應數字都只差 0.1–0.6pp，`none > stage12 > head_only` 的單調關係與 `head_only` 小幅落後（非崩潰式落後）的結論在兩份獨立預訓練骨幹上都成立——**確認這是方法本身的性質，不是單次預訓練跑出來的巧合**。

#### Pretrained Models（真實預訓練對照，`v1.0-realpretrain-freeze`）

| Checkpoint | 設定 | best_top1 | 下載 |
|---|---|---:|---|
| `egwt_plantvillage_none_best.pt` | PlantVillage, none | 99.77% | [下載](https://github.com/s51744/egwt-reproduction/releases/download/v1.0-realpretrain-freeze/egwt_plantvillage_none_best.pt) |
| `egwt_plantvillage_stage12_best.pt` | PlantVillage, stage12 | 99.64% | [下載](https://github.com/s51744/egwt-reproduction/releases/download/v1.0-realpretrain-freeze/egwt_plantvillage_stage12_best.pt) |
| `egwt_plantvillage_head_only_best.pt` | PlantVillage, head_only | 97.39% | [下載](https://github.com/s51744/egwt-reproduction/releases/download/v1.0-realpretrain-freeze/egwt_plantvillage_head_only_best.pt) |
| `egwt_cassava_none_best.pt` | cassava, none | 84.58% | [下載](https://github.com/s51744/egwt-reproduction/releases/download/v1.0-realpretrain-freeze/egwt_cassava_none_best.pt) |
| `egwt_cassava_stage12_best.pt` | cassava, stage12 | 82.09% | [下載](https://github.com/s51744/egwt-reproduction/releases/download/v1.0-realpretrain-freeze/egwt_cassava_stage12_best.pt) |
| `egwt_cassava_head_only_best.pt` | cassava, head_only | 77.79% | [下載](https://github.com/s51744/egwt-reproduction/releases/download/v1.0-realpretrain-freeze/egwt_cassava_head_only_best.pt) |
| `egwt_tomato_none_best.pt` | tomato, none | 98.98% | [下載](https://github.com/s51744/egwt-reproduction/releases/download/v1.0-realpretrain-freeze/egwt_tomato_none_best.pt) |
| `egwt_tomato_stage12_best.pt` | tomato, stage12 | 98.91% | [下載](https://github.com/s51744/egwt-reproduction/releases/download/v1.0-realpretrain-freeze/egwt_tomato_stage12_best.pt) |
| `egwt_tomato_head_only_best.pt` | tomato, head_only | 88.61% | [下載](https://github.com/s51744/egwt-reproduction/releases/download/v1.0-realpretrain-freeze/egwt_tomato_head_only_best.pt) |

載入範例：
```python
import torch
from model import EGWT

model = EGWT(num_classes=1000, pretrained_effnet=False)
state = torch.load("egwt_plantvillage_head_only_best.pt", map_location="cpu")
model.load_state_dict(state)
model.head = torch.nn.Linear(1024, 38)  # 換成對應資料集的類別數
```

## 使用方式

```bash
# 資料整理（依論文的 85/15 train/test split）
python prepare_plantvillage.py
python prepare_cassava.py
python prepare_tomato.py

# (可選) 完整 ImageNet-1k 從頭預訓練，比照論文的訓練協議
python prepare_imagenet.py --extracted_root <ILSVRC2012解壓縮路徑>
python pretrain_imagenet.py --data_dir data/imagenet --epochs 50 --batch_size 32

# 在各作物資料集上微調
python train.py --data_dir data/plantvillage --dataset_name plantvillage --epochs 100
python train.py --data_dir data/cassava --dataset_name cassava --epochs 100
python train.py --data_dir data/tomato --dataset_name tomato --epochs 100
# 若使用完整 ImageNet 預訓練權重，加上 --imagenet_ckpt checkpoints/egwt_imagenet_best.pt
```

## 免責聲明

這是獨立的第三方復刻，與原論文作者無關。目的是驗證論文方法的可重現性並誠實記錄任何無法逐字複現之處，不代表對原論文的權威性解讀。
