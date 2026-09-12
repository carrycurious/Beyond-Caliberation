# Beyond Calibration: Romanization for Closing the Quantization Gap in Multilingual LLMs

<p align="center">
  <strong>Romanization as a simple, calibration-independent strategy for improving multilingual robustness under low-bit quantization.</strong>
</p>

<p align="center">
  Sandeep Kumar · Kunal Kingkar
  <br>
  Indian Institute of Technology, Kharagpur
</p>

---

## Description

This repository accompanies research on **multilingual robustness under low-bit LLM quantization**. We study whether calibration-data mismatch fully explains performance gaps between English and Indic languages, and evaluate **Romanization (via IndicXlit)** as an input-level alternative to calibration-based fixes.

Using **Qwen 3.5 (9B)** and **Gemma 4 (12B-It)** across six languages and five Indic benchmarks, we show that:

- Multilingual quantization gaps persist even under **calibration-free** quantization.
- **Native + Romanized** prompting improves low-bit accuracy and reduces degradation versus native-script inputs alone.
- Romanization can match or exceed language-aware AWQ at 4-bit **without model modification, continual pretraining, or language-specific calibration**.

> **Status:** Experimental code and full reproduction scripts are being finalized and will be added soon.

---

## 📄 Overview

Post-training quantization is widely used to reduce the memory footprint and inference cost of large language models (LLMs). However, existing work has shown that quantization can disproportionately degrade performance on low-resource and non-Latin-script languages.

A common explanation is **calibration-data mismatch**: quantization methods that rely on calibration data may be calibrated primarily on English, causing the resulting quantized model to preserve English performance better than that of other languages.

This work asks a broader question:

> **How much of the multilingual quantization gap is actually caused by calibration?**

We investigate this question using two multilingual instruction-tuned LLMs:

- **Qwen 3.5 (9B)**
- **Gemma 4 (12B-It)**

across:

- **English**
- **Hindi**
- **Bengali**
- **Gujarati**
- **Tamil**
- **Telugu**

and five multilingual NLU and reasoning benchmarks:

- IndicXNLI
- IndicCOPA
- Indic-BoolQ
- Indic-MMLU
- ARC-Challenge

We compare native-script inputs against **Romanized inputs generated using IndicXlit**, under BF16, INT8, and NF4 quantization, together with calibration-free and language-aware AWQ settings.

The central finding is that **the multilingual quantization gap persists even without calibration**, suggesting that calibration mismatch is not its sole cause. More importantly, providing both the native and Romanized representations of an input substantially improves robustness to low-bit quantization.

---

##  Key Findings

### 1. Calibration is not the sole source of the multilingual quantization gap

Even under calibration-free quantization, Indic languages experience greater degradation than English.

For example, across the evaluated tasks:

| Model | Quantization | English degradation | Indic degradation |
|---|---:|---:|---:|
| Gemma 4 | INT8 | 0.20 pp | 0.74 pp |
| Gemma 4 | NF4 | 0.50 pp | 4.59 pp |
| Qwen 3.5 | INT8 | 0.67 pp | 1.35 pp |
| Qwen 3.5 | NF4 | 1.47 pp | 2.60 pp |

This demonstrates that multilingual quantization disparities cannot be attributed entirely to English-only calibration.

---

### 2. Native + Romanized prompting is substantially more robust

For Qwen 3.5, the Indic macro-average at NF4 is:

| Prompting strategy | NF4 accuracy |
|---|---:|
| Native | **55.39%** |
| Romanized only | **44.89%** |
| Native + Romanized | **57.09%** |

Thus, providing the Romanized representation **alongside** the original native-script input improves NF4 performance by approximately **1.70 percentage points** over native prompting.

The hybrid representation also reduces the quantization degradation slope from approximately:

- **−2.73 percentage points / quantization tier** for Native
- **−1.00 percentage point / quantization tier** for Native + Romanized

---

### 3. The effect is also observed in Gemma 4

For Gemma 4, the Indic macro-average at NF4 is:

| Prompting strategy | NF4 accuracy |
|---|---:|
| Native | **68.88%** |
| Romanized only | **47.00%** |
| Native + Romanized | **70.28%** |

The hybrid prompting strategy therefore improves over native prompting by approximately **1.40 percentage points** at NF4.

The paper reports a reduction in the degradation slope from **−2.51** for native prompting to approximately **−1.67 percentage points per quantization tier** for Native + Romanized prompting in Section 4.  
> **Note:** the abstract reports the latter slope as **1.27%**, while Section 4 reports **1.67%**.

---

### 4. Romanization can compete with calibration-based quantization

For Qwen 3.5:

- Native + IndicXlit at NF4: **57.09%**
- Native + AWQ at 4-bit: **56.20%**

Thus, the input-level Romanization strategy slightly outperforms the calibrated 4-bit native configuration in this experiment, despite requiring:

- no model modification,
- no continual pretraining,
- no language-specific calibration,
- and no additional fine-tuning.

---

##  Motivation

Quantization compresses model weights and/or activations into lower-precision representations.

While this can dramatically reduce inference cost, the resulting quantization error is not necessarily distributed uniformly across languages.

The problem is particularly important for multilingual LLMs because:

1. Different languages use different scripts.
2. Different scripts can interact differently with the model tokenizer.
3. Multilingual representations may not be equally robust to quantization error.
4. Calibration-based quantization methods can inherit biases from their calibration distributions.

Previous work has highlighted calibration-data mismatch as an important contributor to multilingual degradation.

This work therefore separates two questions:

### Question 1

> Does the multilingual quantization gap disappear when calibration is removed?

### Question 2

> Can changing only the input representation make a multilingual model more robust to quantization?

The experiments suggest:

- **No** for Question 1 — substantial multilingual degradation remains without calibration.
- **Yes** for Question 2 — Romanized representations, particularly Native + Romanized prompting, can substantially improve low-bit robustness.

---

# 🧪 Method

The experiments consist of two complementary components.

### Component 1 — Calibration-free quantization

We evaluate different input representations under:

- BF16
- INT8 using RPN
- NF4

The purpose is to determine whether multilingual degradation exists independently of language-specific calibration.

### Component 2 — Calibration-aware quantization

We additionally evaluate **Activation-aware Weight Quantization (AWQ)** using multilingual calibration data.

The calibration data contains samples from:

- English
- Hindi
- Bengali
- Gujarati
- Tamil
- Telugu

drawn from C4 and Wikipedia.

This allows us to compare:

> input-level representation changes

against

> quantization-level calibration changes.

---

#  Input Representations

Each Indic-language evaluation example is evaluated under three prompting strategies.

## 1. Native Script

The original text is provided directly to the model.

```text
Question: <native-script question>
