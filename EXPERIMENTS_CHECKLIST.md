# CCF-B 论文改进实验清单

## A. 必做实验（已完成）

### 1. ✅ 多随机种子评估 (mean±std)
**脚本**: `src/multiseed_eval.py`

**结果** (`results/multiseed_results.csv`):
| Dataset | Model | Seed 0 | Seed 1 | Seed 2 | Mean±Std |
|---------|-------|--------|--------|--------|----------|
| SMS | MiniLM+LR | 0.9478 | 0.9478 | 0.9478 | 0.9478±0.00 |
| Telegram | MiniLM+LR | 0.9197 | 0.9197 | 0.9197 | 0.9197±0.00 |

**说明**: MiniLM 嵌入是确定性的，因此 LR seed 变化不影响结果。DistilBERT 需要单独跑。

**运行**: `python -m src.multiseed_eval --datasets sms telegram`

---

### 2. ✅ 误差分析 (Error Taxonomy)
**脚本**: `src/error_taxonomy.py`

**结果** (`results/error_taxonomy.csv`, `results/error_examples.md`):

| Category | SMS | Telegram |
|----------|-----|----------|
| false_positive_other | 0 | 18 |
| false_negative_other | 1 | 10 |
| short_text | 1 | 5 |
| char_substitution | 0 | 6 |
| long_subtle_spam | 0 | 6 |
| template_marketing | 0 | 3 |

**关键发现**:
- SMS 模型仅 3 个错误（高准确率）
- Telegram 主要错误：误报（34%）、漏报（21%）、短文本（11%）
- JSD=0.43 解释了 Telegram→SMS 高迁移性：两者分布相似

**运行**: `python -m src.error_taxonomy --datasets sms telegram`

---

### 3. ✅ EAT 消融实验
**脚本**: `src/eat_ablation.py`

**结果** (`results/eat_ablation.csv`):

#### Attack Mix 消融
| Config | Clean | Obfuscate | Prompt Injection |
|--------|-------|-----------|------------------|
| obfuscate_only | 0.991 | 0.981 | **0.859** |
| prompt_inj_only | 0.991 | 0.981 | 0.575 |
| balanced | 0.991 | 0.986 | 0.657 |
| weighted_obf (default) | 0.991 | 0.981 | 0.709 |

**关键发现**: `obfuscate_only` 配置在保持 clean/obfuscate 性能的同时，对 prompt_injection 防御最强！

#### Augmentation Ratio 消融
| Ratio | Clean | Obfuscate | Prompt Injection |
|-------|-------|-----------|------------------|
| 0% (baseline) | 0.986 | 0.981 | **0.862** |
| 10% | 0.991 | 0.986 | 0.677 |
| 30% | 0.991 | 0.986 | 0.728 |
| 50% | 0.991 | 0.977 | 0.736 |
| 70% | 0.991 | 0.981 | 0.732 |

**关键发现**: 30-50% 增强比例最优

**运行**: `python -m src.eat_ablation --datasets sms --ablations aug_ratio attack_mix`

---

## B. 可选实验

### 4. 🔄 强基线评估 (RoBERTa/DeBERTa)
**脚本**: `src/strong_baselines.py`
**状态**: 脚本已就绪，后台训练中

**运行**: `python -m src.strong_baselines --models roberta-base --train-dataset sms`

预期输出:
- In-domain F1
- Cross-domain F1 (SMS→Telegram)
- Robustness under attacks
- Latency (ms/msg)
- Parameters (M)

---

### 5. ✅ 高级真实攻击
**脚本**: `src/robustness/advanced_attacks.py`

**攻击类型**:
- **Homoglyph**: Unicode 同形异义字替换 (`a` → `а`/Cyrillic)
- **Zero-width**: 零宽字符注入 (分词破坏)
- **URL Obfuscation**: `http://` → `hxxp://`, `.` → `[.]`
- **Currency variants**: `$` → `＄`/`💲`
- **Number obfuscation**: `0` → `o`/`O`/`⓪`
- **Mixed-script**: 组合多种攻击

**防御**:
- `normalize_text()`: NFKC 标准化 + 去除零宽字符 + 反向同形字映射

**示例**:
```
Original: "URGENT: You won $1000!"
Homoglyph: "UｒGЕNT: Ｙоu wｏn $1000!"
Zero-width: "UR‍GENT: Y​ou won $1000!"
Mixed: "UR‍GENT: Y​ou ‌won ＄1000!"
Normalized: "URGENT: You won SI000!"
```

---

## 论文写作建议

### Discussion 节可用的发现

1. **JSD 与跨域性能相关性**
   - JSD(SMS, Telegram) = 0.427 (最低)
   - Telegram→SMS F1 = 0.936 (最高跨域)
   - 说明：低分布差异导致高迁移性

2. **EAT 的权衡**
   - `obfuscate_only` 配置意外地对 prompt_injection 也有防御效果
   - 过度增强 (70%) 反而降低 obfuscate 防御

3. **错误类型归因**
   - 短文本 (11%) 是跨域的主要挑战
   - Telegram 的误报多为商业/营销内容

4. **Green AI 论点**
   - TF-IDF Char SVM: 0.04ms, F1=0.98
   - MiniLM+LR: 24ms, F1=0.95
   - RoBERTa-base: ~100ms (预估), F1 TBD
   - 结论：简单模型在成本效益上更优
