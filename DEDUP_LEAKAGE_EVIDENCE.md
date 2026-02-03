# DedupShift Leakage Evidence

## 概述

本文档记录 **DedupShift 方法的定量证据**，用于支撑论文中关于"消除数据泄漏"的核心贡献声明。

通过这些定量指标，DedupShift 从"方法提出"升级为"经实证验证的有效方法"。

---

## 关键发现（可直接引用）

### 数据集：SMS (UCI)
- **原始样本数**: 5,574
- **去重后样本数**: 5,126 (保留率: **92.0%**)
- **移除样本**: 448 (415 精确重复 + 33 近似重复)
- **泄漏消除效果**:
  - Train/Test overlap: **69 → 0** (100% 消除)
  - Train/Val overlap: **60 → 0** (100% 消除)
  - Val/Test overlap: **10 → 0** (100% 消除)
  - **总泄漏消除**: 139 → 0

### 数据集：SpamAssassin
- **原始样本数**: 6,008
- **去重后样本数**: 2,884 (保留率: **48.0%**)
- **移除样本**: 3,124 (3,070 精确重复 + 54 近似重复)
- **泄漏消除效果**:
  - Train/Test overlap: **494 → 0** (100% 消除)
  - Train/Val overlap: **491 → 0** (100% 消除)
  - Val/Test overlap: **58 → 0** (100% 消除)
  - **总泄漏消除**: 1,043 → 0

---

## 论文引用建议

### 在 DedupShift 方法章节添加

> **定量验证**: 表 X 展示了 DedupShift 在两个数据集上的泄漏消除效果。对于 SMS 数据集，该方法在保留 92% 样本的前提下，完全消除了 139 个 train/test 交叉重叠样本（100% 泄漏消除）。对于 SpamAssassin 数据集，虽然去重率更高（保留 48%），但同样实现了 1,043 个重叠样本的完全清除。这一结果证明，DedupShift 能够在不同数据特征下（SMS 的短消息 vs SpamAssassin 的邮件模板），稳定地保证训练-测试分离的纯净性。

### LaTeX 表格引用
```latex
\begin{table}[t]
\centering
\caption{DedupShift 泄漏消除定量证据}
\label{tab:dedup_leakage}
\begin{tabular}{lrrrr}
\toprule
数据集 & 原始样本 & 保留率 & 原始泄漏 & 去重后泄漏 \\
\midrule
SMS (UCI) & 5,574 & 92.0\% & 139 & \textbf{0} \\
SpamAssassin & 6,008 & 48.0\% & 1,043 & \textbf{0} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 数据来源

所有数据来自实际运行：
- **原始报告**: `results/dedup_report_sms.csv`, `results/dedup_report_spamassassin.csv`
- **汇总表**: `results/dedup_leakage_stats.csv`
- **生成脚本**: `src/generate_leakage_table.py`

### 复现指令
```bash
# 运行去重（生成原始报告）
make dedup

# 生成汇总表
make generate_leakage_table

# 或一键复现全部
make paper_repro
```

---

## 详细统计表

完整指标见 `results/dedup_leakage_stats.csv`:

| 字段 | 说明 |
|------|------|
| `n_original` | 去重前样本总数 |
| `n_exact_dup` | 精确重复样本数 |
| `n_near_dup` | 近似重复样本数（MinHash h_thresh=3） |
| `n_deduplicated` | 去重后样本数 |
| `retention_rate_%` | 样本保留率 |
| `orig_train_test_overlap` | 原始 train/test 交叉重叠数 |
| `dedup_train_test_overlap` | 去重后 train/test 重叠（应为0） |
| `total_orig_overlap` | 原始所有分割间重叠总和 |
| `total_dedup_overlap` | 去重后总重叠（应为0） |
| `overlap_reduction_%` | 泄漏消除百分比（应为100%） |

---

## 讨论价值

### 1. **方法有效性**
- 两个数据集均实现 **100% 泄漏消除**，证明方法在不同数据分布下通用

### 2. **保留率差异**
- SMS 保留率高（92%）：短消息重复率低，近似去重温和
- SpamAssassin 保留率低（48%）：邮件模板重复严重，去重更激进
- **启示**: 数据集特征影响去重程度，但不影响泄漏消除效果

### 3. **实用价值**
- 避免"测试集见过训练样本"导致的性能高估
- 为跨域实验提供干净的基准数据
- 可作为文本分类数据集预处理的标准流程

---

## Threats to Validity 补充建议

可在论文 Threats 章节添加：

> **数据泄漏控制**: 我们通过 DedupShift 方法完全消除了 train/test 交叉污染（SMS 消除 139 样本泄漏，SpamAssassin 消除 1,043 样本泄漏），但这也带来了样本量减少（SMS 减少 8%，SpamAssassin 减少 52%）。未来研究可探索在保持更高样本保留率的前提下实现泄漏控制的方法。

---

## 参考代码

查看脚本实现：
```python
# src/generate_leakage_table.py
# 提取 dedup_report 中的关键指标，生成汇总表
```

---

**生成时间**: 2025-01-XX  
**版本**: v1.0.0  
**对应论文章节**: Section 3 (DedupShift Method)
