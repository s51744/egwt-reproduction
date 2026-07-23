# EGWT Reproduction

非官方復刻：J. Feng et al., *"Enhanced Crop Disease Detection With EfficientNet Convolutional Group-Wise Transformer"*, IEEE Access, 2024.

論文本身未公開程式碼，此為根據論文公式、架構圖、與訓練細節逐步重建的 PyTorch 實作，並在論文引用的三個原始資料集（PlantVillage、cassava、tomato leaves）上實際訓練驗證。

## 架構

3-stage CNN + Group-wise Transformer 混合架構：DWTE（深度可分離卷積切塊）→ Convolutional Projection → Group-wise Multi-Head Attention (G-MHA) → Group-wise MLP (G-MLP)，stage 3 額外加入 EfficientNet-B0 風格的 projection。

## 與原論文的差異（摘要，完整說明見 `model.py` docstring）

| 項目 | 狀態 |
|---|---|
| DWTE、G-MHA 組內權重共享、G-MLP 展開結構 | ✅ 經論文原圖 (Fig. 6, 7) 逐一核對確認吻合 |
| Convolutional Projection 的 stride | ✅ 完全比照 Fig. 6(b) 的 stride=1（改用 PyTorch flash attention 解決效能問題後，不再需要 stride=2 妥協） |
| Table 2 的 768/1024 工作維度 | ⚠️ 論文資訊不足以唯一決定架構（字面讀取會使參數量超預算達1.9倍），採用內部最一致的替代讀法，參數量 19.88M vs 論文 23.04M |
| EfficientNet-B0 projection 內部結構 | ⚠️ 論文只給不透明方塊，用 depthwise conv + Squeeze-Excite 近似 |
| ImageNet 預訓練 | 初版用真實 EfficientNet-B0 權重 + Transformer 隨機初始化；後續已改為完整 ImageNet-1k from-scratch 預訓練，比照論文 Section III.B.5 |
| cassava 的 SMOTE 過採樣 | ⚠️ 改用 class-weighted loss（SMOTE 定義的特徵空間論文未說明） |

## 結果對照

| 資料集 | 論文 Top-1 | 本次 Top-1 (100 epoch) | 論文 Params | 本次 Params |
|---|---|---|---|---|
| PlantVillage | 99.88% | 93.81% | 23.04M | 19.88M |
| cassava | 84.29% | 69.71% | 23.04M | 19.84M |
| tomato | 99.99% | 93.46% | 23.04M | 19.85M |

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
