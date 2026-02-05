# Telegram Dataset Integration Guide

本文档总结了将 **Telegram Spam or Ham** 数据集（Kaggle）集成到现有 pipeline 的完整方案。

## 📁 新增文件清单

### 数据目录
```
dataset/telegram_spam_ham/
├── raw/                        # Kaggle 原始下载
├── processed/                  # 标准化后 (data.csv, train/val/test.csv)
└── dedup/processed/            # DedupShift 后的拆分

data/telegram_spam_ham/
└── dedup/processed/            # EAT 训练用的数据副本
```

### 脚本文件
| 文件 | 功能 |
|------|------|
| [src/prepare_telegram.py](src/prepare_telegram.py) | Kaggle CSV → 统一格式 + 初始拆分 |
| [src/domain_shift_stats_3domains.py](src/domain_shift_stats_3domains.py) | 三域 JSD 计算 |
| [data/data_card_telegram.md](data/data_card_telegram.md) | 数据卡片 |

### 修改的文件
| 文件 | 改动 |
|------|------|
| [Makefile](Makefile) | 新增 `telegram_*` targets + 更新 `paper_repro` |
| [src/build_results_dedup.py](src/build_results_dedup.py) | 支持三域评估循环 |
| [src/eval_eat_cross_domain.py](src/eval_eat_cross_domain.py) | 添加 telegram 路径 |
| [src/merge_robustness_dedup.py](src/merge_robustness_dedup.py) | 合并三域 robustness |
| [README.md](README.md) | 添加 Telegram 使用说明 |

---

## 🚀 使用流程

### 方式 A：一键完成（推荐）
```bash
# 下载数据（需配置 Kaggle API）
make telegram_download

# 完整 pipeline
make telegram_full
```

### 方式 B：分步执行
```bash
# Step 1: 下载
kaggle datasets download -d mexwell/telegram-spam-or-ham \
    -p dataset/telegram_spam_ham/raw --unzip

# Step 2: 标准化 + 初始拆分
make telegram_prepare

# Step 3: DedupShift
make telegram_dedup

# Step 4: 训练基线模型
make telegram_train

# Step 5: 同步到 data/ 目录
make telegram_sync

# Step 6: EAT 增强
make telegram_eat_augment
make telegram_eat_train

# Step 7: Robustness 评估
make telegram_robust
```

---

## 📊 产出文件

### 评估结果
| 文件 | 说明 |
|------|------|
| `results/dedup_report_telegram.csv` | DedupShift 统计 |
| `results/robustness_dedup_telegram.csv` | Robustness 评估 |
| `results/domain_shift_stats_3domains.csv` | 三域特征统计 |
| `results/domain_shift_js_3domains.csv` | 三域 JSD 矩阵 |

### 模型文件
| 文件模式 | 说明 |
|----------|------|
| `models/telegram_dedup_*.joblib` | 基线模型 |
| `models/telegram_dedup_*_eat.joblib` | EAT 模型 |

---

## 📝 论文写作要点

### 1. Domain Shift 更强
> Telegram（chat）与 SMS/email 的 n-gram 分布偏移显著（JSD 更大），解释了 cross-domain degrade 的结构性原因。

引用：`results/domain_shift_js_3domains.csv`

### 2. 结论泛化验证
> 我们的结论不仅在 "old corpora" 成立，也在现代 **chat scams** 场景下成立。

引用：`results/cross_domain_table_dedup.csv` 中的 `telegram_dedup` 行

### 3. EAT 跨域提升复现
> EAT/AttackMix 在 Telegram→SMS 和 Telegram→SpamAssassin 场景下同样有效。

引用：`results/eat_cross_domain.csv` 中的 telegram 相关行

---

## ⚠️ 注意事项

1. **Kaggle API 配置**：需要 `~/.kaggle/kaggle.json`
2. **可选性**：如果没有 Telegram 数据，`make paper_repro` 仍能正常运行（只处理 SMS + SpamAssassin）
3. **数据许可**：Kaggle 数据集遵循其原有许可证，详见 [data_card_telegram.md](data/data_card_telegram.md)

---

## 🔄 与 paper_repro 的集成

`make paper_repro` 现在会自动检测 Telegram 数据是否存在：
- **存在**：包含在三域评估中
- **不存在**：仅运行原有的 SMS + SpamAssassin 两域

这确保了向后兼容性，同时允许扩展到三域分析。
