#!/usr/bin/env python3
"""Plot attack evaluation metrics from results/attacks/<run_id>/."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config

METRIC_JSON_RE = re.compile(
    r"^(?P<attack>fgsm|pgd)_(?P<mode>unconstrained|constrained)_eps(?P<epsilon>[\dp.]+)(?:_metrics)?\.json$",
    re.IGNORECASE,
)
ADV_EVAL_METRIC_JSON_RE = re.compile(
    r"^pass(?P<pass>\d+)_(?P<attack>fgsm|pgd)_(?P<mode>unconstrained|constrained)_eps(?P<epsilon>[\dp.]+)\.json$",
    re.IGNORECASE,
)
ADV_EVAL_RUN_PREFIX = "adv_eval_"


def _parse_epsilon_tag(tag: str) -> float:
    return float(tag.replace("p", "."))


def find_attacks_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["results_dir"]) / "attacks"


def is_adv_eval_run(run_dir: Path) -> bool:
    if run_dir.name.startswith(ADV_EVAL_RUN_PREFIX):
        return True
    return any(ADV_EVAL_METRIC_JSON_RE.match(p.name) for p in run_dir.glob("*.json"))


def list_attack_runs(
    attacks_root: Path,
    *,
    adv_eval_only: bool = False,
    standard_only: bool = False,
) -> list[Path]:
    if not attacks_root.is_dir():
        return []
    runs = sorted(
        (p for p in attacks_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if adv_eval_only:
        runs = [p for p in runs if p.name.startswith(ADV_EVAL_RUN_PREFIX)]
    elif standard_only:
        runs = [
            p
            for p in runs
            if not p.name.startswith(ADV_EVAL_RUN_PREFIX)
            and not p.name.startswith("pilot_")
        ]
    return runs


def _run_has_metrics(run_dir: Path) -> bool:
    if (run_dir / "summary.json").is_file() or (run_dir / "manifest.json").is_file():
        return True
    return any(
        METRIC_JSON_RE.match(p.name) or ADV_EVAL_METRIC_JSON_RE.match(p.name)
        for p in run_dir.glob("*.json")
    )


def resolve_run_dir(
    attacks_root: Path,
    run_id: str | None,
    *,
    adv_eval: bool | None = None,
) -> Path:
    if run_id is not None:
        run_dir = attacks_root / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Attack run not found: {run_dir}")
        return run_dir

    if adv_eval is None:
        adv_eval = False

    runs = list_attack_runs(
        attacks_root,
        adv_eval_only=adv_eval,
        standard_only=not adv_eval,
    )
    if not adv_eval:
        runs = [p for p in runs if _run_has_metrics(p)]
    if not runs:
        kind = "adv_eval" if adv_eval else "standard attack"
        raise FileNotFoundError(f"No {kind} runs found under {attacks_root}")

    return runs[-1]


def _record(
    attack: str,
    mode: str,
    epsilon: float,
    metrics: dict[str, Any],
    *,
    pass_num: int | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "attack": attack.lower(),
        "mode": mode.lower(),
        "epsilon": float(epsilon),
        "asr": float(metrics["asr"]),
        "robust_accuracy": float(metrics["robust_accuracy"]),
    }
    if pass_num is not None:
        rec["pass"] = int(pass_num)
    return rec


def _parse_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if isinstance(summary.get("results"), list):
        for item in summary["results"]:
            attack = str(item.get("attack", "unknown")).lower()
            mode = str(item.get("mode", item.get("constraint_mode", "unknown"))).lower()
            epsilon = float(item["epsilon"])
            metrics = item.get("metrics", item)
            pass_raw = item.get("adv_pass", item.get("pass"))
            pass_num = int(pass_raw) if pass_raw is not None else None
            records.append(_record(attack, mode, epsilon, metrics, pass_num=pass_num))
        return records

    attacks = summary.get("attacks", {})
    if isinstance(attacks, dict):
        for attack_name, modes in attacks.items():
            if not isinstance(modes, dict):
                continue
            for mode_name, eps_map in modes.items():
                if not isinstance(eps_map, dict):
                    continue
                for eps_key, metrics in eps_map.items():
                    if not isinstance(metrics, dict):
                        continue
                    records.append(
                        _record(attack_name, mode_name, float(eps_key), metrics)
                    )
        if records:
            return records

    for key, value in summary.items():
        if not isinstance(value, dict):
            continue
        if "asr" not in value or "robust_accuracy" not in value:
            continue
        match = METRIC_JSON_RE.match(f"{key}.json")
        if match:
            records.append(
                _record(
                    match.group("attack"),
                    match.group("mode"),
                    _parse_epsilon_tag(match.group("epsilon")),
                    value,
                )
            )
    return records


def _parse_metric_file(path: Path) -> dict[str, Any] | None:
    match = METRIC_JSON_RE.match(path.name)
    pass_num: int | None = None
    if not match:
        adv_match = ADV_EVAL_METRIC_JSON_RE.match(path.name)
        if not adv_match:
            return None
        match = adv_match
        pass_num = int(match.group("pass"))

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("metrics", payload) if isinstance(payload, dict) else payload
    if not isinstance(metrics, dict) or "asr" not in metrics or "robust_accuracy" not in metrics:
        return None
    if pass_num is None and isinstance(payload, dict):
        pass_raw = payload.get("adv_pass", payload.get("pass"))
        if pass_raw is not None:
            pass_num = int(pass_raw)
    return _record(
        match.group("attack"),
        match.group("mode"),
        _parse_epsilon_tag(match.group("epsilon")),
        metrics,
        pass_num=pass_num,
    )


def load_attack_records(run_dir: Path) -> list[dict[str, Any]]:
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        records = _parse_summary(summary)
        if records:
            return records

    records = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name in ("summary.json", "manifest.json"):
            continue
        record = _parse_metric_file(path)
        if record is not None:
            records.append(record)

    if not records:
        raise FileNotFoundError(
            f"No attack metrics found in {run_dir}. "
            "Expected summary.json or metric JSON files "
            "(e.g. fgsm_unconstrained_eps0.01.json or pass1_pgd_constrained_eps0p01.json)"
        )
    return records


def _group_records(
    records: list[dict[str, Any]], attack: str
) -> dict[float, dict[str, float]]:
    grouped: dict[float, dict[str, float]] = {}
    for rec in records:
        if rec["attack"] != attack:
            continue
        eps = rec["epsilon"]
        grouped.setdefault(eps, {})
        grouped[eps][rec["mode"]] = rec
    return grouped


def _plot_grouped_bars(
    grouped: dict[float, dict[str, dict[str, float]]],
    metric_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    epsilons = sorted(grouped.keys())
    modes = ["unconstrained", "constrained"]
    x = range(len(epsilons))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, mode in enumerate(modes):
        values = [
            grouped[eps].get(mode, {}).get(metric_key, float("nan"))
            for eps in epsilons
        ]
        offset = (idx - (len(modes) - 1) / 2) * width
        ax.bar(
            [i + offset for i in x],
            values,
            width=width,
            label=mode.replace("_", " ").title(),
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(eps) for eps in epsilons])
    ax.set_xlabel("Epsilon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _group_adv_eval_records(
    records: list[dict[str, Any]], attack: str, pass_num: int
) -> dict[float, dict[str, float]]:
    grouped: dict[float, dict[str, float]] = {}
    for rec in records:
        if rec["attack"] != attack or rec.get("pass") != pass_num:
            continue
        eps = rec["epsilon"]
        grouped.setdefault(eps, {})
        grouped[eps][rec["mode"]] = rec
    return grouped


def _plot_adv_eval_lines(
    grouped_by_pass: dict[int, dict[float, dict[str, dict[str, float]]]],
    metric_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    passes = sorted(grouped_by_pass.keys())
    modes = ["unconstrained", "constrained"]
    n_passes = len(passes)
    fig, axes = plt.subplots(1, n_passes, figsize=(5 * n_passes, 4), sharey=True)
    if n_passes == 1:
        axes = [axes]

    for ax, pass_num in zip(axes, passes):
        grouped = grouped_by_pass[pass_num]
        epsilons = sorted(grouped.keys())
        for mode in modes:
            values = [
                grouped[eps].get(mode, {}).get(metric_key, float("nan"))
                for eps in epsilons
            ]
            ax.plot(
                epsilons,
                values,
                marker="o",
                label=mode.replace("_", " ").title(),
            )
        ax.set_xlabel("Epsilon")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Pass {pass_num}")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_adv_eval_results(run_dir: Path) -> list[Path]:
    records = load_attack_records(run_dir)
    if not any("pass" in rec for rec in records):
        raise ValueError(
            f"Adv-eval run {run_dir.name} has no per-pass records "
            "(expected adv_pass in summary or passN_* metric files)"
        )

    plots_dir = run_dir / "plots"
    saved: list[Path] = []
    attacks = sorted({rec["attack"] for rec in records})

    for attack in attacks:
        passes = sorted(
            {rec["pass"] for rec in records if rec["attack"] == attack and "pass" in rec}
        )
        grouped_by_pass: dict[int, dict[float, dict[str, dict[str, float]]]] = {}
        for pass_num in passes:
            grouped = _group_adv_eval_records(records, attack, pass_num)
            if grouped:
                grouped_by_pass[pass_num] = grouped
        if not grouped_by_pass:
            continue

        suffix = f"_{attack}" if len(attacks) > 1 else ""
        asr_path = plots_dir / f"asr_by_epsilon_per_pass{suffix}.png"
        robust_path = plots_dir / f"robust_accuracy_per_pass{suffix}.png"

        _plot_adv_eval_lines(
            grouped_by_pass,
            metric_key="asr",
            ylabel="Attack Success Rate (ASR)",
            title=f"{attack.upper()}: ASR vs Epsilon (Unconstrained vs Constrained)",
            out_path=asr_path,
        )
        _plot_adv_eval_lines(
            grouped_by_pass,
            metric_key="robust_accuracy",
            ylabel="Robust Accuracy",
            title=f"{attack.upper()}: Robust Accuracy vs Epsilon (Unconstrained vs Constrained)",
            out_path=robust_path,
        )
        saved.extend([asr_path, robust_path])

    return saved


def plot_attack_results(run_dir: Path) -> list[Path]:
    if is_adv_eval_run(run_dir):
        return plot_adv_eval_results(run_dir)

    records = load_attack_records(run_dir)
    plots_dir = run_dir / "plots"
    saved: list[Path] = []

    attacks = sorted({rec["attack"] for rec in records})
    for attack in attacks:
        grouped = _group_records(records, attack)
        if not grouped:
            continue

        suffix = f"_{attack}" if len(attacks) > 1 else ""
        asr_path = plots_dir / f"asr_by_epsilon{suffix}.png"
        robust_path = plots_dir / f"robust_accuracy{suffix}.png"

        _plot_grouped_bars(
            grouped,
            metric_key="asr",
            ylabel="Attack Success Rate (ASR)",
            title=f"{attack.upper()}: Unconstrained vs Constrained ASR by Epsilon",
            out_path=asr_path,
        )
        _plot_grouped_bars(
            grouped,
            metric_key="robust_accuracy",
            ylabel="Robust Accuracy",
            title=f"{attack.upper()}: Robust Accuracy by Epsilon",
            out_path=robust_path,
        )
        saved.extend([asr_path, robust_path])

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ASR and robust accuracy from attack evaluation runs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Attack run id under results/attacks/ (default: latest baseline run)",
    )
    parser.add_argument(
        "--adv-eval",
        action="store_true",
        help="Select latest adv_eval_* run (auto when --run-id starts with adv_eval_)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config/default.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    attacks_root = find_attacks_root(cfg)

    adv_eval: bool | None = args.adv_eval or None
    if args.run_id and args.run_id.startswith(ADV_EVAL_RUN_PREFIX):
        adv_eval = True

    run_dir = resolve_run_dir(attacks_root, args.run_id, adv_eval=adv_eval)

    saved = plot_attack_results(run_dir)
    print(f"Run: {run_dir.name}")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
