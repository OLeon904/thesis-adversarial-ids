# Adversarial Robustness of ML-Based NIDS

CICIDS2017 flow CSV experiments: **Random Forest** + **MLP**, **FGSM/PGD** with optional physical-feasibility constraints in raw feature space, **adversarial training**, and **RF transfer** evaluation.

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

Pilot (50k rows):

```powershell
py -3 scripts/run_preprocess.py --pilot
py -3 scripts/run_baselines.py --pilot
```

### 4. Attacks, adversarial training, RF transfer

```powershell
py -3 scripts/run_attacks.py --mode both
py -3 scripts/run_adversarial_training.py
py -3 scripts/eval_adv_trained_attacks.py --adv-run <adv_run_id> --passes 3 --mode both
py -3 scripts/run_rf_transfer.py --epsilon 0.01
py -3 scripts/plot_attack_results.py --run-id <attack_run_id>
py -3 scripts/validate_results.py
```

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for full commands, config profiles, and 100k vs full-test evaluation.

**Physical-feasibility constraints:** Constrained FGSM/PGD project perturbations in **raw CICFlowMeter units** (`inverse_transform` → `project_batch` → `transform`) before each MLP forward pass—not only in StandardScaler space.

**Training note:** `config/default.yaml` sets `max_train_samples: 500000` for CPU-friendly baseline training. Set to `null` for the full training split.

### Config profiles

Profiles under `config/profiles/` merge onto `config/default.yaml`. Use `--config` or `THESIS_CONFIG`:

```powershell
$env:THESIS_CONFIG = "config/profiles/quick.yaml"
py -3 scripts/run_preprocess.py
py -3 scripts/run_baselines.py
Remove-Item Env:THESIS_CONFIG
```

## Repository layout

```
config/default.yaml     # hyperparameters and paths
config/profiles/        # quick / full overrides
src/                    # preprocessing, models, attacks, constraints
scripts/                # CLI entry points
docs/RUNBOOK.md         # reproduction guide
data/raw/               # CICIDS2017 CSVs (gitignored)
data/processed/         # splits, scaler (gitignored)
results/                # metrics JSON (gitignored)
```

## Implementation roadmap

1. Preprocessing + stratified split + baseline RF/MLP
2. FGSM/PGD (unconstrained + raw-space constrained)
3. Constraint layer (mask, integer projection, IAT–duration)
4. Adversarial training + post-training attack eval
5. Transfer attacks to RF (MLP-generated FGSM)
