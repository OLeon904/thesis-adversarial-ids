# Adversarial Robustness of ML-Based NIDS

CICIDS2017 flow CSV experiments: **Random Forest** + **MLP**, **FGSM/PGD** with optional physical-feasibility constraints in raw feature space, **adversarial training**, and **RF transfer** evaluation.

**Authoritative reproduction guide:** [docs/RUNBOOK.md](docs/RUNBOOK.md). Compiled metrics: [docs/THESIS_RESULTS.md](docs/THESIS_RESULTS.md). Pinned run IDs: [`config/thesis_results_runs.yaml`](config/thesis_results_runs.yaml).

## Quick start

### 1. Environment

```powershell
cd "c:\Users\Leon\Documents\College Courses\Thesis Course\thesis-adversarial-ids"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```

### 2. CICIDS2017 data

```powershell
py -3 scripts/download_cicids2017.py
```

Manual alternative: [UNB CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) → extract into `data/raw/`.

### 3. Preprocess + baselines

```powershell
py -3 scripts/run_preprocess.py
py -3 scripts/run_baselines.py
```

Default protocol: `max_train_samples: 500000`, `attacks.max_test_samples: 100000`. Full test: set `max_test_samples: null` (pinned run `20260520T220618Z`).

### 4. Final workflow (thesis)

```powershell
py -3 scripts/run_attacks.py --mode both
py -3 scripts/run_adversarial_training.py
py -3 scripts/eval_adv_trained_attacks.py --adv-run <adv_run_id> --passes 3 --mode both
py -3 scripts/run_rf_transfer.py --mode unconstrained
py -3 scripts/run_rf_transfer.py --mode constrained
py -3 scripts/run_adversarial_training.py --constrained
py -3 scripts/eval_adv_trained_attacks.py --adv-run <constrained_adv_run_id> --passes 3 --mode both
py -3 scripts/run_headline_multi_seed.py --seeds 42 43 44 --baseline-run <baseline_run_id>
py -3 scripts/audit_constraint_violations.py
py -3 scripts/plot_attack_results.py --run-id <attack_run_id>
py -3 scripts/validate_results.py
```

**Physical-feasibility constraints:** Constrained FGSM/PGD project in **raw CICFlowMeter units** (`inverse_transform` → `project_batch` → `transform`) before each MLP forward pass.

### Config profiles

```powershell
$env:THESIS_CONFIG = "config/profiles/quick.yaml"
py -3 scripts/run_preprocess.py
Remove-Item Env:THESIS_CONFIG
```

## Repository layout

```
config/default.yaml     # hyperparameters (100k test default)
config/profiles/        # quick / full overrides
src/                    # preprocessing, models, attacks, constraints
scripts/                # CLI entry points
docs/RUNBOOK.md         # full reproduction guide
docs/THESIS_RESULTS.md  # compiled metric tables
data/raw/               # CICIDS2017 CSVs (gitignored)
data/processed/         # splits, scaler (gitignored)
results/                # metrics JSON (gitignored)
```
