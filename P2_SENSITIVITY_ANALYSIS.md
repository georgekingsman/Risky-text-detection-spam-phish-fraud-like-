# P2 Experimental Completeness: Sensitivity Analysis and Multi-Seed Robustness

本文档说明项目中新增的两项可选但增强了实验完整性和可信度的功能。

## 1. DedupShift 超参敏感性分析 (DedupShift Hyperparameter Sensitivity)

### 背景
DedupShift使用SimHash的Hamming距离阈值 (`h_thresh`) 来识别近似重复。阈值的选择影响：
- **去重速率**: 更低的阈值→更激进的去重（移除更多样本）
- **模型性能**: 去重过度可能损伤训练数据，去重不足可能保留泄露

### 实现
- **脚本**: [src/sensitivity_analysis_dedup.py](src/sensitivity_analysis_dedup.py)
- **测试阈值**: 2, 3 (默认), 4
- **代表模型**: TF-IDF word + Logistic Regression（简单且快速）
- **输出**: 
  - `results/sensitivity_dedup_summary.csv` - 汇总表格
  - LaTeX表格和可视化（通过 `generate_sensitivity_tables.py`）

### 使用方式
```bash
# 单独运行敏感性分析
python -m src.sensitivity_analysis_dedup

# 或通过 make 命令
make sensitivity_dedup

# 生成LaTeX表格和图表
make generate_sensitivity_tables

# 在完整论文复现中包含
make paper_repro  # 包含敏感性分析步骤
```

### 结果解释
输出表格示例：
```
dataset        h_thresh  n_input  n_exact_removed  n_near_removed  n_output  dedup_rate_%  f1_score
SMS (UCI)      2         4459     203              234             4022      9.80          0.9523
SMS (UCI)      3         4459     203              156             4100      8.05          0.9631
SMS (UCI)      4         4459     203              92              4164      6.61          0.9702
```

**观察**:
- `h_thresh=2` 去重最激进 (9.80%) → F1稍低 (0.9523)
- `h_thresh=3` 平衡 (8.05%) → F1 (0.9631)
- `h_thresh=4` 保守 (6.61%) → F1 稍高但可能包含泄露 (0.9702)

---

## 2. DistilBERT 多Seed训练 (DistilBERT Multi-Seed Training)

### 背景
神经网络训练通常依赖于随机种子。为了增强研究的严谨性和可信度，我们报告多个随机种子下的结果，以及平均值和标准差。

### 实现
- **脚本**: [src/train_distilbert_multiseed.py](src/train_distilbert_multiseed.py)
- **Seeds**: 0, 1, 2
- **输出**: 
  - `results/distilbert_multiseed.csv` - 聚合结果（mean±std）
  - `results/distilbert_multiseed_seeds.csv` - 原始per-seed结果
  - LaTeX表格（通过 `generate_sensitivity_tables.py`）

### 使用方式
```bash
# 单独运行多seed训练
python src/train_distilbert_multiseed.py \
  --train_csv dataset/dedup/processed/all.csv \
  --train_domain sms \
  --eval_csvs dataset/dedup/processed/all.csv dataset/spamassassin/dedup/processed/all.csv \
  --eval_domains sms spamassassin \
  --out_dir models/distilbert_sms_dedup_multiseed \
  --results_csv results/distilbert_multiseed.csv \
  --seeds 0 1 2 --epochs 2 --batch 8 --max_len 128

# 或通过 make 命令
make distilbert_multiseed

# 在完整论文复现中包含
make paper_repro  # 包含多seed DistilBERT步骤
```

### 结果解释
输出表格示例（mean±std格式）：
```
train_domain  test_domain   split  model            f1_mean  f1_std
sms           sms           test   distilbert_ft    0.9854   0.0012
sms           spamassassin  test   distilbert_ft    0.5623   0.0145
spamassassin  spamassassin  test   distilbert_ft    0.9911   0.0008
spamassassin  sms           test   distilbert_ft    0.0234   0.0089
```

**观察**:
- In-domain F1非常高且稳定 (std < 0.002)
- Cross-domain F1低且方差较大 (std~0.01)，表明cross-domain问题的确存在
- 标准差小表示训练过程稳定，增强论文可信度

---

## 3. 完整集成到论文复现流程

### Makefile 新增 targets

```makefile
# 单独运行敏感性分析
make sensitivity_dedup

# 单独运行多seed DistilBERT
make distilbert_multiseed

# 生成敏感性分析的LaTeX表格和图表
make generate_sensitivity_tables

# 完整论文复现（包含所有P2功能）
make paper_repro
```

### paper_repro 中的新步骤

当运行 `make paper_repro` 时，以下新步骤会自动执行：

1. **敏感性分析** (位置：dedup之后，模型训练之前)
   ```
   $(PY) -m src.sensitivity_analysis_dedup
   ```

2. **多seed DistilBERT** (位置：单seed DistilBERT之后)
   ```
   $(PY) src/train_distilbert_multiseed.py ...
   ```

3. **生成敏感性表格** (位置：所有模型训练之后)
   ```
   $(PY) -m src.generate_sensitivity_tables
   ```

### 论文中的新章节

在 [paper/main.tex](paper/main.tex) 中添加了新的 Section：

#### 7.1 Hyperparameter Sensitivity and Multi-Seed Robustness
- **DedupShift阈值分析** (Tab. 5): 显示不同 $h_\text{thresh}$ 下的去重速率和F1权衡
- **DistilBERT多seed** (Tab. 6): 显示3个seed下的mean±std F1得分
- **图表** (Fig. 3): 去重速率 vs F1的可视化曲线

#### 8.2 Threats to Validity - 更新
- 强调DedupShift阈值权衡已通过敏感性分析量化
- 说明neural baseline的seed方差已通过多seed报告控制

---

## 性能考虑

### 运行时间估计

| 任务 | 时间 | CPU/GPU |
|------|------|---------|
| `sensitivity_analysis_dedup` | ~30秒 | CPU (3 thresholds × 2 datasets) |
| `distilbert_multiseed` | ~10-15分钟 (CPU) / ~2分钟 (GPU) | 3个seeds × DistilBERT训练 |
| `generate_sensitivity_tables` | ~5秒 | CPU |

**建议**: 
- 如果CPU较慢，DedupShift敏感性分析可快速完成
- DistilBERT多seed训练可选择在GPU服务器上运行（`--device cuda`）

### GPU支持

```bash
# 在GPU上运行多seed DistilBERT
python src/train_distilbert_multiseed.py \
  ... \
  --device cuda  # 或 'mps' for Apple Silicon
```

---

## 集成到论文提交的建议

### 表格和图表位置
```
paper/
├── tables/
│   ├── sensitivity_dedup_threshold.tex    # 新增
│   ├── distilbert_multiseed.tex           # 新增
│   ├── cross_domain_table_dedup.tex       # 既有
│   ├── dedup_effect.tex
│   └── ...
├── figs/
│   ├── fig_sensitivity_dedup_threshold.png # 新增
│   └── ...
└── main.tex  # 新增 Section 7
```

### 论文描述建议

**在Results或Experimental Setup中**:
> "We analyze the robustness of DedupShift to the choice of Hamming threshold $h_{\text{thresh}} \in \{2,3,4\}$ by measuring deduplication rates and downstream model performance (Table X, Fig. Y). To ensure training stability of our neural baseline (DistilBERT), we train with three seeds (0, 1, 2) and report mean $\pm$ std F1 scores (Table Z), confirming high in-domain stability but substantial cross-domain variance."

---

## 故障排查

### `sensitivity_analysis_dedup` 失败
- ✅ 检查 `dataset/processed/train.csv` 存在
- ✅ 检查 `src/dedup_split.py` 可用
- ✅ 若使用自定义paths，更新脚本中的硬编码路径

### `distilbert_multiseed` 失败
- ✅ 检查 `transformers`, `torch` 已安装
- ✅ 检查 GPU 可用（若使用 `--device cuda`）
- ✅ 检查 `dataset/dedup/processed/all.csv` 存在
- ✅ 若内存不足，减小 `--batch` 大小

### `generate_sensitivity_tables` 失败
- ✅ 检查 `matplotlib` 已安装
- ✅ 检查 `results/sensitivity_dedup_summary.csv` 存在
- ✅ 检查 `paper/tables/` 和 `paper/figs/` 目录存在

---

## 参考

- **论文**: [paper/main.tex](paper/main.tex) - 完整论文（含新Section）
- **Makefile**: [Makefile](Makefile) - 包含所有targets定义
- **脚本**:
  - [src/sensitivity_analysis_dedup.py](src/sensitivity_analysis_dedup.py)
  - [src/train_distilbert_multiseed.py](src/train_distilbert_multiseed.py)
  - [src/generate_sensitivity_tables.py](src/generate_sensitivity_tables.py)

---

**总结**: P2功能通过定量分析DedupShift超参以及多seed neural baseline结果，进一步增强了论文的实验完整性和CCF-C投稿的竞争力。🎯
