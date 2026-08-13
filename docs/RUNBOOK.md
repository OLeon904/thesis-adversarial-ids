# Thesis Experiment Runbook

Step-by-step reproduction from an empty machine to all thesis artifacts. Commands match [README.md](../README.md).

**Workspace root** (adjust if yours differs):

```text
c:\Users\Leon\Documents\College Courses\Thesis Course\thesis-adversarial-ids
```

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|--------|
| Python 3.10+ | `py -3 --version` on Windows |
| ~12–15 GB free disk | Raw CSVs + processed NPZ + results (see §10) |
| CPU or GPU | All scripts use CUDA when available; thesis numbers below are **CPU** estimates |
| Network | One-time Hugging Face download (~300 MB zip) |

---

## 2. Environment (empty machine)

```powershell
cd "c:\Users\Leon\Documents\College Courses\Thesis Course\thesis-adversarial-ids"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```

`huggingface_hub` is included in `requirements.txt` (needed by `scripts/download_cicids2017.py`).

Optional quick sanity check (synthetic data, **not** for thesis numbers):

```powershell
py -3 scripts/generate_synthetic_cicids.py
py -3 scripts/run_preprocess.py --pilot
py -3 scripts/run_baselines.py --pilot
```

---

## 3. Download CICIDS2017

```powershell
py -3 scripts/download_cicids2017.py
```

- Source: Hugging Face mirror `bencorn/CICIDS2017`, file `csvs/MachineLearningCSV.zip`
- Output: eight day CSVs in `data/raw/` (e.g. `Monday-WorkingHours.pcap_ISCX.csv`, …)
- Manual alternative: [UNB CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) → extract CSVs into `data/raw/`

Verify:

```powershell
(Get-ChildItem data\raw\*.csv).Count   # expect 8
```

**Do not** use `synthetic_smoke.csv` for final thesis metrics.

---

## 4. Configure thesis subsamples (recommended)

Edit `config/default.yaml` before long runs:

| Key | Thesis default | Full-scale alternative |
|-----|----------------|------------------------|
| `data.max_train_samples` | `500000` | `null` (all ~1.98M train rows) |
| `attacks.max_test_samples` | `100000` | `null` (all 566,149 test rows) |

Current repo default has `attacks.max_test_samples: 100000` (primary thesis protocol). For the full 566,149-row test set, set `attacks.max_test_samples: null` (pinned full-test attack run: `20260520T220618Z`). See §9 and §13.

Training cap note (from README): `max_train_samples: 500000` keeps baseline training on CPU to roughly **2–3 hours**; `null` is much slower without a GPU.

---

## 5. Preprocess

```powershell
py -3 scripts/run_preprocess.py
```

- Reads all CSVs in `data/raw/`, cleans labels, stratified 70/10/20 split
- Writes `data/processed/splits.npz`, `scaler.joblib`, `label_encoder.joblib`, `metadata.json`
- Full dataset: **2,830,743** flows → train **1,981,520** / val **283,074** / test **566,149**

Pilot (50k rows, timing only):

```powershell
py -3 scripts/run_preprocess.py --pilot
```

---

## 6. Baselines (RF + MLP)

```powershell
py -3 scripts/run_baselines.py
```

- Output: `results/baselines/<run_id>/`
  - `rf_model.joblib`, `mlp_model.pt`
  - `summary.json`, `rf_test_metrics.json`, `mlp_test_metrics.json`
- Records `train_rows_used` and full `test_rows` in `summary.json`

Pilot:

```powershell
py -3 scripts/run_baselines.py --pilot
```

Note the baseline `<run_id>` (e.g. `20260520T192855Z`) for later steps.

---

## 7. Adversarial attacks (baseline MLP)

```powershell
py -3 scripts/run_attacks.py --mode both
```

- FGSM + PGD, unconstrained + constrained, ε ∈ {0.005, 0.01, 0.02} → **12** metric JSON files
- Output: `results/attacks/<run_id>/` + `manifest.json`
- Default checkpoint: latest `results/baselines/<run_id>/mlp_model.pt`

Optional:

```powershell
py -3 scripts/run_attacks.py --checkpoint results/baselines/<run_id>/mlp_model.pt
py -3 scripts/run_attacks.py --mode constrained --pilot
```

---

## 8. Adversarial training (MLP)

```powershell
py -3 scripts/run_adversarial_training.py
```

- Output: `results/adv_train/<run_id>/` with `pass_1.pt` … `pass_3.pt`, `summary.json`
- Loads baseline MLP from latest baseline run (or `--baseline-run <run_id>`)

Optional:

```powershell
py -3 scripts/run_adversarial_training.py --baseline-run <baseline_run_id> --passes 3
py -3 scripts/run_adversarial_training.py --constrained
```

`--constrained` applies raw-space projection during inner PGD (aligned train/test threat). Run ids are prefixed `constrained_` (e.g. `constrained_20260605T004621Z`).

Note the adv train `<run_id>` (e.g. `20260520T193255Z`).

---

## 9. Evaluate attacks on adv-trained MLP

```powershell
py -3 scripts/eval_adv_trained_attacks.py --adv-run <adv_run_id> --passes 3 --mode both
```

- Output: `results/attacks/adv_eval_<adv_run_id>/` (36 metric files for 3 passes × 12 configs)
- Uses the same test cap as `attacks.max_test_samples` in config

Example:

```powershell
py -3 scripts/eval_adv_trained_attacks.py --adv-run 20260520T193255Z --passes 3 --mode both
```

---

## 10. Random Forest transfer attacks

```powershell
py -3 scripts/run_rf_transfer.py --mode unconstrained
py -3 scripts/run_rf_transfer.py --mode constrained
```

- MLP white-box FGSM (default ε = first value in `attacks.epsilon_values`, typically **0.01**)
- `--mode unconstrained` (default) or `--mode constrained` sets the MLP attack before RF evaluation
- Output: `results/rf_transfer/<run_id>/transfer_results.json`
- Optional: `--baseline-run <run_id>`, `--attack pgd`, `--epsilon 0.01`, `--pilot`

---

## 11. Plots

```powershell
py -3 scripts/plot_attack_results.py --run-id <attack_run_id>
```

Output: `results/attacks/<run_id>/plots/*.png` (ASR + robust accuracy bar charts).

`plot_attack_results.py` without `--run-id` uses the **latest** directory under `results/attacks/`. For baseline attack plots, pass the non-`adv_eval_` run id.

---

## 12. Constraint violation audit

Before or after constrained runs, verify that raw-space projection removes physical-feasibility violations:

```powershell
py -3 scripts/audit_constraint_violations.py
py -3 scripts/audit_constraint_violations.py --n-samples 10000 --epsilon 0.01 --out results/constraint_audit.json
```

- Simulates scaled L∞ perturbations at ε, decodes to raw CICFlowMeter units, counts violations, applies `project_batch`, recounts
- Default output: `results/constraint_audit.json` (override with `--out`)

---

## 13. Switching 100k test subsample vs full test set

Attack scripts subsample the test split when `attacks.max_test_samples` is a positive integer; when `null`, the entire test set is used.

**100k test subsample (primary attack eval):**

```yaml
# config/default.yaml
attacks:
  max_test_samples: 100000
```

Then re-run (or run once):

```powershell
py -3 scripts/run_attacks.py --mode both
py -3 scripts/eval_adv_trained_attacks.py --adv-run <adv_run_id> --passes 3 --mode both
py -3 scripts/run_rf_transfer.py --mode unconstrained
py -3 scripts/run_rf_transfer.py --mode constrained
```

Console prints: `Test subsample: 100,000 / 566,149 rows`. `manifest.json` records `n_test_samples`.

**Full test set (566,149 rows):**

```yaml
attacks:
  max_test_samples: null
```

Runtime scales ~5–6× vs 100k for attack and adv-eval steps. RF transfer uses the same cap.

**Pilot / smoke (fast):**

```powershell
py -3 scripts/run_attacks.py --pilot
```

With `--pilot`, if `max_test_samples` is `null`, attacks default to **10,000** test rows; if set in config, that value is used. Preprocess/baselines `--pilot` caps at `pilot_max_rows` (50,000) and 50,000 train rows.

---

## 14. Run ID naming

| Pattern | When | Example |
|---------|------|---------|
| `%Y%m%dT%H%M%SZ` | Normal run (UTC) | `20260520T193215Z` |
| `pilot_<timestamp>` | `--pilot` on preprocess/baselines/attacks/adv_train/rf_transfer | `pilot_20260520T120000Z` |
| `constrained_<timestamp>` | `--constrained` on `run_adversarial_training.py` | `constrained_20260605T004621Z` |
| `adv_eval_<adv_train_run_id>` | `eval_adv_trained_attacks.py` | `adv_eval_20260520T193255Z` |
| `adv_eval_constrained_<adv_train_run_id>` | adv eval of constrained adv train | `adv_eval_constrained_20260605T004621Z` |

Directories:

```text
results/baselines/<run_id>/
results/attacks/<run_id>/              # baseline MLP attacks
results/attacks/adv_eval_<adv_run_id>/ # adv-trained eval
results/adv_train/<run_id>/
results/rf_transfer/<run_id>/
```

Pin a baseline in config (optional):

```yaml
paths:
  baseline_run: 20260520T192855Z
```

### Pinned thesis runs (June 2026)

All manuscript run IDs are recorded in [`config/thesis_results_runs.yaml`](../config/thesis_results_runs.yaml):

| Key | Run ID |
|-----|--------|
| `baselines` | `20260520T192855Z` |
| `attacks_100k` | `20260520T220532Z` |
| `attacks_full_test` | `20260520T220618Z` |
| `adv_train` | `20260520T193255Z` |
| `adv_eval` | `adv_eval_20260520T193255Z` |
| `rf_transfer` | `20260520T231047Z` |
| `rf_transfer_constrained` | `20260605T004611Z` |
| `adv_train_constrained` | `constrained_20260605T010438Z` |
| `adv_eval_constrained` | `adv_eval_constrained_20260605T010438Z` |
| `multi_seed_headline` | `headline_20260605T010708Z` |
| `adv_train_constrained_pilot` | `constrained_20260605T004621Z` (1-pass pilot) |
| `adv_eval_constrained_pilot` | `adv_eval_constrained_20260605T004621Z` |

---

## 15. Expected CPU runtime (order of magnitude)

Assumes `max_train_samples: 500000`, single machine, no GPU. Your times vary with CPU and disk.

| Step | Command | ~100k test | ~full test (566k) |
|------|---------|------------|-------------------|
| Download | `download_cicids2017.py` | 10–30 min (network) | same |
| Preprocess | `run_preprocess.py` | 20–45 min | same |
| Baselines | `run_baselines.py` | **2–3 h** | **8–15 h** if `max_train_samples: null` |
| Attacks | `run_attacks.py --mode both` | **1–2 h** | **5–8 h** |
| Adv train (3 passes) | `run_adversarial_training.py` | **2–4 h** | longer with full train |
| Adv eval | `eval_adv_trained_attacks.py` | **3–6 h** | **15–25 h** |
| RF transfer | `run_rf_transfer.py` | **15–30 min** | scales with test cap |
| Plots | `plot_attack_results.py` | **< 1 min** | same |

**End-to-end path (500k train + 100k test):** roughly **12–18 hours** CPU wall time, mostly baselines + adv train + adv eval.

**Pilot path** (`--pilot` throughout): under **30 minutes** after preprocess.

---

## 16. Disk space

| Location | Approximate size |
|----------|------------------|
| `data/raw/` (8 CSVs) | ~1.5–2 GB |
| Hugging Face zip cache | ~300 MB (outside repo, in HF cache) |
| `data/processed/` (`splits.npz` + artifacts) | ~400–800 MB |
| `results/baselines/` | ~50–200 MB per run |
| `results/attacks/` | ~1–5 MB JSON per run; adv_eval runs larger |
| `results/adv_train/` | ~10–30 MB per run (3 checkpoints) |
| `results/rf_transfer/` | ~1–50 MB (optional `x_adv_*.npy`) |
| `.venv` + PyTorch | ~2–3 GB |

**Total plan for ~15 GB free** on the project drive.

---

## 17. Full reproduction checklist

Run in order; record each `<run_id>` from console or `summary.json`. Pinned thesis ids are in [`config/thesis_results_runs.yaml`](../config/thesis_results_runs.yaml).

```powershell
cd "c:\Users\Leon\Documents\College Courses\Thesis Course\thesis-adversarial-ids"
.\.venv\Scripts\Activate.ps1

# 1. Data
py -3 scripts/download_cicids2017.py

# 2. Set config: max_train_samples: 500000, max_test_samples: 100000
#    (full test: max_test_samples: null → pinned attacks_full_test 20260520T220618Z)

# 3. Pipeline
py -3 scripts/run_preprocess.py
py -3 scripts/run_baselines.py
py -3 scripts/run_attacks.py --mode both
py -3 scripts/run_adversarial_training.py
py -3 scripts/eval_adv_trained_attacks.py --adv-run <adv_run_id> --passes 3 --mode both
py -3 scripts/run_rf_transfer.py --mode unconstrained
py -3 scripts/run_rf_transfer.py --mode constrained
py -3 scripts/run_adversarial_training.py --constrained
py -3 scripts/eval_adv_trained_attacks.py --adv-run <constrained_adv_run_id> --passes 3 --mode both
py -3 scripts/run_headline_multi_seed.py --seeds 42 43 44 --baseline-run <baseline_run_id>
py -3 scripts/audit_constraint_violations.py

# 4. Plots + validation
py -3 scripts/plot_attack_results.py --run-id <attack_run_id>
py -3 scripts/validate_results.py
```

**Deliverables:**

| Artifact | Path |
|----------|------|
| Processed splits | `data/processed/splits.npz`, `metadata.json` |
| Baseline models | `results/baselines/<run_id>/` |
| Attack metrics | `results/attacks/<run_id>/`, `adv_eval_<adv_run_id>/` |
| Adv-trained checkpoints | `results/adv_train/<run_id>/pass_*.pt` |
| RF transfer | `results/rf_transfer/<run_id>/transfer_results.json` |
| Figures | `results/attacks/<run_id>/plots/*.png` |
| Validation report | `validate_results.py` stdout |

---

## 18. Troubleshooting

| Issue | Action |
|-------|--------|
| `No CSV files in data/raw` | Run `download_cicids2017.py` or copy CSVs manually |
| `Baseline MLP not found` | Run `run_baselines.py` before attacks / adv train |
| `adv-run` not found | Use folder name under `results/adv_train/` exactly |
| Plot script picks wrong run | Pass `--run-id` explicitly; adv_eval dirs are skipped for “latest” only when sorting by mtime — prefer explicit id |
| OOM on attacks | Lower `attacks.batch_size` in config |
| Metrics differ between runs | Match `max_test_samples`, same `seed: 42`, same baseline/adv run ids |

---

## 19. One-line command reference (README parity)

```powershell
py -3 scripts/download_cicids2017.py
py -3 scripts/run_preprocess.py
py -3 scripts/run_baselines.py
py -3 scripts/run_attacks.py --mode both
py -3 scripts/run_adversarial_training.py
py -3 scripts/run_adversarial_training.py --constrained
py -3 scripts/eval_adv_trained_attacks.py --adv-run <adv_run_id> --passes 3 --mode both
py -3 scripts/run_rf_transfer.py --mode unconstrained
py -3 scripts/run_rf_transfer.py --mode constrained
py -3 scripts/audit_constraint_violations.py
py -3 scripts/plot_attack_results.py --run-id <attack_run_id>
py -3 scripts/validate_results.py
```
