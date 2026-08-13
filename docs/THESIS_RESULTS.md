# Thesis Results — Adversarial Robustness of ML-Based NIDS

Auto-generated from experiment artifacts. CICIDS2017 flow CSV, 78 features, 15 classes.

## Evaluation protocol

| Report | Run ID | Test rows |
|--------|--------|-----------|
| Baselines | `20260520T192855Z` | 566,149 (full test) |
| Primary attacks (thesis table) | `20260520T220532Z` | 100,000 stratified |
| Supplement full-test attacks | `20260520T220618Z` | 566,149 |
| Adv-trained attack eval | `adv_eval_20260520T193255Z` | 100,000 stratified |
| RF transfer (unconstrained) | `20260520T231047Z` | 100,000 stratified |
| RF transfer (constrained) | `20260605T004611Z` | 100,000 stratified |
| Constrained adv train | `constrained_20260605T010438Z` | — |
| Constrained adv eval | `adv_eval_constrained_20260605T010438Z` | 100,000 stratified |
| Multi-seed headline | `headline_20260605T010708Z` | 100,000 × 3 seeds |

See `docs/EVALUATION_PROTOCOL.md`. Constrained attacks: project in **raw physical feature space**, then StandardScaler-transform for the MLP.

## 1. Clean Baseline Performance

**Run:** `20260520T192855Z` | Train rows: 500000 | Test rows: 566149

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Random Forest | 0.9965 | 0.7796 | 0.9971 |
| MLP | 0.8840 | 0.4850 | 0.9111 |

## 2. Adversarial Attacks on Baseline MLP

### Attacks run `20260520T220532Z`

**Test rows:** 100000 (full test split: 566149) | **Constraint space:** raw_physical

| Attack | Mode | ε | ASR | Robust Acc | Macro F1 |
|--------|------|---|-----|------------|----------|
| fgsm | unconstrained | 0.005 | 0.0139 | 0.8733 | 0.4258 |
| fgsm | unconstrained | 0.01 | 0.0245 | 0.8639 | 0.4130 |
| fgsm | unconstrained | 0.02 | 0.0488 | 0.8424 | 0.3868 |
| fgsm | constrained | 0.005 | 0.0030 | 0.8829 | 0.4485 |
| fgsm | constrained | 0.01 | 0.0054 | 0.8808 | 0.4442 |
| fgsm | constrained | 0.02 | 0.0089 | 0.8777 | 0.4375 |
| pgd | unconstrained | 0.005 | 0.0140 | 0.8732 | 0.4256 |
| pgd | unconstrained | 0.01 | 0.0248 | 0.8637 | 0.4130 |
| pgd | unconstrained | 0.02 | 0.0324 | 0.8569 | 0.4014 |
| pgd | constrained | 0.005 | 0.0030 | 0.8829 | 0.4485 |
| pgd | constrained | 0.01 | 0.0049 | 0.8812 | 0.4444 |
| pgd | constrained | 0.02 | 0.0069 | 0.8795 | 0.4411 |

**Finding (constrained vs unconstrained):** Physical-feasibility projection in **raw feature space** (inverse StandardScaler → project → rescale) changes the robustness evaluation versus unconstrained L_inf attacks in scaled space alone.

- Paired comparisons: **6** (FGSM/PGD × ε ∈ {0.005, 0.01, 0.02}).
- Constrained ASR **lower** than unconstrained: **6/6**.
- Constrained ASR **higher** than unconstrained: **0/6**.
- Constrained **robust accuracy** higher: **6/6**.

Do **not** claim constrained attacks always reduce ASR; report robust accuracy and ASR jointly. Unconstrained feature-space attacks can overstate evasion when implausible perturbations are allowed.

- Example: PGD ε=0.01: ASR 2.48% → 0.49%; robust acc 86.37% → 88.12%.

### Supplement: full test-set attacks

### Attacks run `20260520T220618Z`

**Test rows:** 566149 (full test split: 566149) | **Constraint space:** raw_physical

| Attack | Mode | ε | ASR | Robust Acc | Macro F1 |
|--------|------|---|-----|------------|----------|
| fgsm | unconstrained | 0.005 | 0.0138 | 0.8718 | 0.4521 |
| fgsm | unconstrained | 0.01 | 0.0246 | 0.8623 | 0.4343 |
| fgsm | unconstrained | 0.02 | 0.0490 | 0.8407 | 0.4085 |
| fgsm | constrained | 0.005 | 0.0026 | 0.8817 | 0.4769 |
| fgsm | constrained | 0.01 | 0.0051 | 0.8796 | 0.4678 |
| fgsm | constrained | 0.02 | 0.0087 | 0.8763 | 0.4632 |
| pgd | unconstrained | 0.005 | 0.0139 | 0.8717 | 0.4520 |
| pgd | unconstrained | 0.01 | 0.0249 | 0.8620 | 0.4343 |
| pgd | unconstrained | 0.02 | 0.0326 | 0.8553 | 0.4229 |
| pgd | constrained | 0.005 | 0.0026 | 0.8817 | 0.4769 |
| pgd | constrained | 0.01 | 0.0047 | 0.8799 | 0.4681 |
| pgd | constrained | 0.02 | 0.0068 | 0.8780 | 0.4666 |

## 3. Random Forest Transfer Attacks (MLP-generated)

| MLP attack mode | Run ID | Attack | ε | RF clean acc | RF acc on adv | Transfer ASR |
|-----------------|--------|--------|---|--------------|---------------|--------------|
| unconstrained | `20260520T231047Z` | fgsm | 0.01 | 0.9967 | 0.8062 | **0.1940** |
| constrained | `20260605T004611Z` | fgsm | 0.01 | 0.9967 | 0.8941 | **0.1057** |

Unconstrained transfer ASR **19.40%**; constrained transfer ASR **10.57%** (same ε, MLP FGSM → RF).

## 4. Adversarial Training (MLP)

**Run:** `20260520T193255Z` | Passes: 3

| Pass | Val Macro F1 | Checkpoint |
|------|--------------|------------|
| 1 | 0.4874 | `pass_1.pt` |
| 2 | 0.4774 | `pass_2.pt` |
| 3 | 0.5286 | `pass_3.pt` |

## 5. Attacks on Adversarially Trained MLP

**Run:** `adv_eval_20260520T193255Z` | Adv train: `20260520T193255Z` | Test: 100000

### 5.1 Full adversarial-evaluation matrix

| Pass | Attack | Mode | ε | ASR | Robust Acc | Macro F1 |
|------|--------|------|---|-----|------------|----------|
| 1 | fgsm | constrained | 0.005 | 0.0014 | 0.8706 | 0.4229 |
| 1 | fgsm | constrained | 0.01 | 0.0029 | 0.8694 | 0.4207 |
| 1 | fgsm | constrained | 0.02 | 0.0053 | 0.8675 | 0.4166 |
| 1 | fgsm | unconstrained | 0.005 | 0.0053 | 0.8672 | 0.4203 |
| 1 | fgsm | unconstrained | 0.01 | 0.0112 | 0.8620 | 0.4132 |
| 1 | fgsm | unconstrained | 0.02 | 0.0334 | 0.8427 | 0.3925 |
| 1 | pgd | constrained | 0.005 | 0.0014 | 0.8706 | 0.4229 |
| 1 | pgd | constrained | 0.01 | 0.0026 | 0.8696 | 0.4209 |
| 1 | pgd | constrained | 0.02 | 0.0038 | 0.8686 | 0.4187 |
| 1 | pgd | unconstrained | 0.005 | 0.0053 | 0.8672 | 0.4205 |
| 1 | pgd | unconstrained | 0.01 | 0.0115 | 0.8618 | 0.4131 |
| 1 | pgd | unconstrained | 0.02 | 0.0204 | 0.8541 | 0.4031 |
| 2 | fgsm | constrained | 0.005 | 0.0023 | 0.8944 | 0.4634 |
| 2 | fgsm | constrained | 0.01 | 0.0045 | 0.8924 | 0.4598 |
| 2 | fgsm | constrained | 0.02 | 0.0070 | 0.8901 | 0.4536 |
| 2 | fgsm | unconstrained | 0.005 | 0.0059 | 0.8911 | 0.4573 |
| 2 | fgsm | unconstrained | 0.01 | 0.0119 | 0.8858 | 0.4509 |
| 2 | fgsm | unconstrained | 0.02 | 0.0317 | 0.8680 | 0.3976 |
| 2 | pgd | constrained | 0.005 | 0.0022 | 0.8944 | 0.4636 |
| 2 | pgd | constrained | 0.01 | 0.0039 | 0.8930 | 0.4604 |
| 2 | pgd | constrained | 0.02 | 0.0053 | 0.8917 | 0.4565 |
| 2 | pgd | unconstrained | 0.005 | 0.0059 | 0.8911 | 0.4573 |
| 2 | pgd | unconstrained | 0.01 | 0.0123 | 0.8854 | 0.4504 |
| 2 | pgd | unconstrained | 0.02 | 0.0228 | 0.8760 | 0.4368 |
| 3 | fgsm | constrained | 0.005 | 0.0013 | 0.8867 | 0.4957 |
| 3 | fgsm | constrained | 0.01 | 0.0034 | 0.8848 | 0.4902 |
| 3 | fgsm | constrained | 0.02 | 0.0054 | 0.8831 | 0.4848 |
| 3 | fgsm | unconstrained | 0.005 | 0.0054 | 0.8830 | 0.4891 |
| 3 | fgsm | unconstrained | 0.01 | 0.0204 | 0.8697 | 0.4642 |
| 3 | fgsm | unconstrained | 0.02 | 0.0315 | 0.8599 | 0.4442 |
| 3 | pgd | constrained | 0.005 | 0.0013 | 0.8867 | 0.4956 |
| 3 | pgd | constrained | 0.01 | 0.0030 | 0.8851 | 0.4903 |
| 3 | pgd | constrained | 0.02 | 0.0041 | 0.8841 | 0.4877 |
| 3 | pgd | unconstrained | 0.005 | 0.0055 | 0.8829 | 0.4887 |
| 3 | pgd | unconstrained | 0.01 | 0.0208 | 0.8694 | 0.4638 |
| 3 | pgd | unconstrained | 0.02 | 0.0262 | 0.8646 | 0.4539 |

### 5.2 Key comparison — constrained PGD (ε=0.01)

| Adv Pass | ASR | Robust Acc |
|----------|-----|------------|
| 1 | 0.0026 | 0.8696 |
| 2 | 0.0039 | 0.8930 |
| 3 | 0.0030 | 0.8851 |

Pass 3 constrained PGD (ε=0.01) ASR **0.30%** vs baseline MLP **0.49%** on the same protocol — adversarial training improves robust accuracy (Δ robust acc **+0.0039**).

### 5.3 Baseline MLP vs adversarial pass 3 (Δ)

| Attack | Mode | ε | Baseline ASR | Pass-3 ASR | Δ ASR | Baseline Rob. Acc | Pass-3 Rob. Acc | Δ Rob. Acc |
|--------|------|---|--------------|------------|-------|-------------------|-----------------|------------|
| fgsm | constrained | 0.005 | 0.0030 | 0.0013 | -0.0017 | 0.8829 | 0.8867 | +0.0037 |
| fgsm | constrained | 0.01 | 0.0054 | 0.0034 | -0.0019 | 0.8808 | 0.8848 | +0.0039 |
| fgsm | constrained | 0.02 | 0.0089 | 0.0054 | -0.0036 | 0.8777 | 0.8831 | +0.0054 |
| fgsm | unconstrained | 0.005 | 0.0139 | 0.0054 | -0.0084 | 0.8733 | 0.8830 | +0.0097 |
| fgsm | unconstrained | 0.01 | 0.0245 | 0.0204 | -0.0041 | 0.8639 | 0.8697 | +0.0058 |
| fgsm | unconstrained | 0.02 | 0.0488 | 0.0315 | -0.0173 | 0.8424 | 0.8599 | +0.0175 |
| pgd | constrained | 0.005 | 0.0030 | 0.0013 | -0.0017 | 0.8829 | 0.8867 | +0.0038 |
| pgd | constrained | 0.01 | 0.0049 | 0.0030 | -0.0019 | 0.8812 | 0.8851 | +0.0039 |
| pgd | constrained | 0.02 | 0.0069 | 0.0041 | -0.0027 | 0.8795 | 0.8841 | +0.0046 |
| pgd | unconstrained | 0.005 | 0.0140 | 0.0055 | -0.0085 | 0.8732 | 0.8829 | +0.0097 |
| pgd | unconstrained | 0.01 | 0.0248 | 0.0208 | -0.0040 | 0.8637 | 0.8694 | +0.0057 |
| pgd | unconstrained | 0.02 | 0.0324 | 0.0262 | -0.0063 | 0.8569 | 0.8646 | +0.0077 |

Negative Δ ASR and positive Δ robust accuracy indicate improved robustness after adversarial training.

### 5.4 Per-class recall — constrained PGD (ε=0.01)

| Class | Baseline | Pass 1 | Pass 2 | Pass 3 |
|-------|------|------|------|------|
| BENIGN | 0.8533 | 0.8400 | 0.8679 | 0.8579 |
| Bot | 0.9857 | 0.9143 | 0.9857 | 0.9857 |
| DDoS | 0.9978 | 0.9867 | 0.9960 | 0.9993 |
| DoS GoldenEye | 0.9918 | 0.9698 | 0.9918 | 0.9725 |
| DoS Hulk | 0.9974 | 0.9980 | 0.9975 | 0.9987 |
| DoS Slowhttptest | 0.9021 | 0.9794 | 0.9794 | 0.9794 |
| DoS slowloris | 0.9902 | 0.9122 | 0.9756 | 0.9707 |
| FTP-Patator | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Heartbleed | 0.0000 | 0.0000 | — | — |
| Infiltration | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| PortScan | 0.9991 | 0.9963 | 0.9989 | 0.9991 |
| SSH-Patator | 0.9712 | 0.9760 | 0.9712 | 0.9712 |
| Web Attack - Brute Force | 0.8679 | 0.0943 | 0.8679 | 0.8679 |
| Web Attack - Sql Injection | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Web Attack - XSS | 0.0000 | 0.9565 | 0.0000 | 0.0000 |

**Highlights:**

- **Lowest baseline recall:** **Heartbleed** (0.00%), **Web Attack - Sql Injection** (0.00%), **Web Attack - XSS** (0.00%).
- **Largest pass-3 gains vs baseline:** **Web Attack - Sql Injection** (+100.00%), **DoS Slowhttptest** (+7.73%).
- **Zero recall after adv training (pass 3):** Web Attack - XSS.
- **Unchanged zero recall:** Heartbleed, Web Attack - XSS.

## 6. Constrained Adversarial Training (aligned threat model)

**Run:** `constrained_20260605T010438Z` | Passes: 3 | Inner PGD constrained: **True**

| Pass | Val Macro F1 | Checkpoint |
|------|--------------|------------|
| 1 | 0.5106 | `pass_1.pt` |
| 2 | 0.5097 | `pass_2.pt` |
| 3 | 0.5197 | `pass_3.pt` |

### 6.1 Constrained-AT attack eval — constrained PGD (ε=0.01)

**Run:** `adv_eval_constrained_20260605T010438Z` | Adv train: `constrained_20260605T010438Z` | Test: 100000

| Adv Pass | ASR | Robust Acc | Macro F1 |
|----------|-----|------------|----------|
| 1 | 0.0053 | 0.8669 | 0.4651 |
| 2 | 0.0032 | 0.8998 | 0.4945 |
| 3 | 0.0053 | 0.8743 | 0.4770 |

Best pass (**2**) constrained PGD (ε=0.01) ASR **0.32%** vs baseline **0.49%**; robust acc **89.98%** (Δ **+0.0185**).

## 7. Multi-seed Headline Audit

**Artifact:** `results/multi_seed/headline_20260605T010708Z.json` | Headline: pgd_constrained_eps0.01 | Seeds: [42, 43, 44]

- Mean ASR: **0.46%** ± 0.03%
- Mean robust accuracy: **88.02%** ± 0.07%

| Seed | ASR | Robust Acc | Macro F1 |
|------|-----|------------|----------|
| 42 | 0.0050 | 0.8812 | 0.4446 |
| 43 | 0.0043 | 0.8798 | 0.4729 |
| 44 | 0.0045 | 0.8798 | 0.4786 |

## 8. Methodological Notes

- **Dataset:** CICIDS2017 MachineLearningCSV (2,830,743 flows), stratified 70/10/20 split.
- **Training subsample:** 500,000 stratified train rows for RF/MLP/adv training (compute budget).
- **Attack evaluation:** 100,000-row stratified subsample for primary thesis tables; 566,149-row full test for supplement run (see § Evaluation protocol).
- **Constraints:** Applied in **raw physical feature space** (inverse StandardScaler → project → rescale); immutable port/flags; integer counts; 20% perturbation cap; relaxed IAT–duration coherence (50% deficit lift; residual strict timing incoherence may remain—not claimed fully physically feasible).
- **RF attacks:** Transfer only (no gradient through RF); evaluate unconstrained and constrained MLP FGSM modes.
- **Multi-seed:** Headline constrained PGD ε=0.01 across stratified test subsample seeds.
