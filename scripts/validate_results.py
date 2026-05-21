#!/usr/bin/env python3
"""Validate results/ artifacts: schema, metric consistency, and attack coverage."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config

ATTACKS = ("fgsm", "pgd")
MODES = ("unconstrained", "constrained")
ADV_PASSES = (1, 2, 3)
METRIC_KEYS = ("asr", "robust_accuracy", "macro_f1", "accuracy")
FLOAT_TOL = 1e-5

METRIC_JSON_RE = re.compile(
    r"^(?P<attack>fgsm|pgd)_(?P<mode>unconstrained|constrained)_eps(?P<epsilon>[\dp.]+)(?:_metrics)?\.json$",
    re.IGNORECASE,
)
ADV_EVAL_JSON_RE = re.compile(
    r"^pass(?P<pass>\d+)_(?P<attack>fgsm|pgd)_(?P<mode>unconstrained|constrained)_eps(?P<epsilon>[\dp.]+)\.json$",
    re.IGNORECASE,
)


@dataclass
class Check:
    category: str
    name: str
    passed: bool
    detail: str
    path: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(
        self,
        category: str,
        name: str,
        passed: bool,
        detail: str,
        path: Path | str = "",
    ) -> None:
        self.checks.append(
            Check(category, name, passed, detail, str(path) if path else "")
        )

    def counts(self) -> tuple[int, int]:
        passed = sum(1 for c in self.checks if c.passed)
        return passed, len(self.checks) - passed


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_epsilon_tag(tag: str) -> float:
    return float(tag.replace("p", "."))


def _epsilon_tag(epsilon: float) -> str:
    s = f"{epsilon:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def expected_attack_combos() -> set[tuple[str, str, float]]:
    eps = tuple(load_config()["attacks"]["epsilon_values"])
    return {(a, m, float(e)) for a in ATTACKS for m in MODES for e in eps}


def _row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    m = row.get("metrics", row)
    if "asr" not in m and "robust_accuracy" in row:
        return row
    return m


def _metrics_close(a: float, b: float, tol: float = FLOAT_TOL) -> bool:
    return abs(a - b) <= tol


def _validate_metrics_block(
    report: Report,
    metrics: dict[str, Any],
    *,
    category: str,
    label: str,
    path: Path,
) -> None:
    for key in METRIC_KEYS:
        if key not in metrics:
            report.add(
                category,
                f"{label}: missing metrics.{key}",
                False,
                f"keys present: {sorted(metrics.keys())}",
                path,
            )
        else:
            report.add(
                category,
                f"{label}: has metrics.{key}",
                True,
                f"{key}={metrics[key]}",
                path,
            )


def _compare_flat_nested(
    report: Report,
    *,
    category: str,
    label: str,
    flat: dict[str, Any],
    nested: dict[str, Any],
    path: Path,
) -> None:
    for key in ("asr", "robust_accuracy"):
        fv = flat.get(key)
        nv = nested.get(key)
        if fv is None or nv is None:
            report.add(
                category,
                f"{label}: flat/nested {key}",
                False,
                f"flat={fv} nested={nv}",
                path,
            )
            continue
        ok = _metrics_close(float(fv), float(nv))
        report.add(
            category,
            f"{label}: flat vs nested {key}",
            ok,
            f"flat={float(fv):.6f} nested={float(nv):.6f}",
            path,
        )


def validate_manifest(run_dir: Path, report: Report) -> set[tuple[str, str, float]]:
    """Validate manifest.json; return combos present in manifest rows."""
    manifest_path = run_dir / "manifest.json"
    present: set[tuple[str, str, float]] = set()
    if not manifest_path.is_file():
        report.add(
            "manifest",
            "manifest.json exists",
            False,
            "no manifest (incomplete or legacy run)",
            run_dir,
        )
        return present

    report.add("manifest", "manifest.json exists", True, manifest_path.name, manifest_path)
    manifest = _load_json(manifest_path)

    for key in ("run_id", "results", "epsilon_values", "n_test_samples"):
        ok = key in manifest
        report.add(
            "manifest",
            f"manifest has {key}",
            ok,
            "present" if ok else f"missing; keys={sorted(manifest.keys())}",
            manifest_path,
        )

    results = manifest.get("results")
    if not isinstance(results, list):
        report.add(
            "manifest",
            "manifest.results is list",
            False,
            f"type={type(results).__name__}",
            manifest_path,
        )
        return present

    for i, row in enumerate(results):
        label = f"results[{i}]"
        for key in ("attack", "mode", "epsilon"):
            ok = key in row
            report.add(
                "manifest",
                f"{label} has {key}",
                ok,
                str(row.get(key, "MISSING")),
                manifest_path,
            )
        if not all(k in row for k in ("attack", "mode", "epsilon")):
            continue

        combo = (str(row["attack"]).lower(), str(row["mode"]).lower(), float(row["epsilon"]))
        present.add(combo)
        flat_asr = row.get("asr")
        flat_rob = row.get("robust_accuracy")
        nested = _row_metrics(row)

        if flat_asr is not None and flat_rob is not None:
            report.add(
                "manifest",
                f"{label} flat asr/robust_accuracy",
                True,
                f"asr={flat_asr} robust={flat_rob}",
                manifest_path,
            )
        else:
            report.add(
                "manifest",
                f"{label} flat asr/robust_accuracy",
                False,
                "expected top-level asr and robust_accuracy",
                manifest_path,
            )

        detail_path = row.get("path")
        if detail_path:
            dp = Path(detail_path)
            if not dp.is_file():
                dp = run_dir / dp.name
            if dp.is_file():
                detail = _load_json(dp)
                dm = detail.get("metrics", detail)
                _compare_flat_nested(
                    report,
                    category="nested_metrics",
                    label=f"{combo[0]} {combo[1]} ε={combo[2]}",
                    flat={"asr": flat_asr, "robust_accuracy": flat_rob},
                    nested=dm,
                    path=dp,
                )
            else:
                report.add(
                    "nested_metrics",
                    f"{label} detail file",
                    False,
                    f"not found: {detail_path}",
                    run_dir,
                )
        elif "metrics" in row:
            _compare_flat_nested(
                report,
                category="nested_metrics",
                label=f"{label} inline",
                flat={"asr": flat_asr, "robust_accuracy": flat_rob},
                nested=nested,
                path=manifest_path,
            )

    expected = expected_attack_combos()
    missing = expected - present
    extra = present - expected
    report.add(
        "coverage",
        f"{run_dir.name}: attack grid complete",
        not missing,
        f"missing={sorted(missing)} extra={sorted(extra)}",
        run_dir,
    )
    return present


def validate_attack_metric_files(run_dir: Path, report: Report) -> set[tuple[str, str, float]]:
    """Validate per-attack JSON files (baseline run layout)."""
    present: set[tuple[str, str, float]] = set()
    for path in sorted(run_dir.glob("*.json")):
        if path.name in ("manifest.json", "summary.json"):
            continue
        match = METRIC_JSON_RE.match(path.name)
        if not match:
            continue
        payload = _load_json(path)
        combo = (
            match.group("attack").lower(),
            match.group("mode").lower(),
            _parse_epsilon_tag(match.group("epsilon")),
        )
        present.add(combo)
        metrics = payload.get("metrics", payload)
        _validate_metrics_block(
            report, metrics, category="attack_json", label=path.name, path=path
        )
        for key in ("attack", "mode", "epsilon"):
            if key in payload and key != "metrics":
                exp = combo[0] if key == "attack" else combo[1] if key == "mode" else combo[2]
                got = payload[key]
                if key == "epsilon":
                    got = float(got)
                ok = str(got).lower() == str(exp).lower() if key != "epsilon" else _metrics_close(
                    float(got), float(exp)
                )
                report.add(
                    "attack_json",
                    f"{path.name}: top-level {key} matches filename",
                    ok,
                    f"file={exp} json={got}",
                    path,
                )
    return present


def validate_adv_eval(run_dir: Path, report: Report) -> None:
    """Validate adv_eval_* run: summary schema, passes 1–3, 36 combos."""
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        report.add(
            "adv_eval",
            "summary.json exists",
            False,
            "required for adv_eval runs",
            run_dir,
        )
        return

    report.add("adv_eval", "summary.json exists", True, summary_path.name, summary_path)
    summary = _load_json(summary_path)
    for key in ("adv_run", "results", "n_test"):
        report.add(
            "adv_eval",
            f"summary has {key}",
            key in summary,
            "present" if key in summary else f"keys={sorted(summary.keys())}",
            summary_path,
        )

    rows = summary.get("results", [])
    if not isinstance(rows, list):
        report.add(
            "adv_eval",
            "summary.results is list",
            False,
            f"type={type(rows).__name__}",
            summary_path,
        )
        return

    present: set[tuple[int, str, str, float]] = set()
    for i, row in enumerate(rows):
        label = f"results[{i}]"
        for key in ("adv_pass", "attack", "mode", "epsilon", "metrics"):
            ok = key in row
            report.add(
                "adv_eval",
                f"{label} has {key}",
                ok,
                str(row.get(key, "MISSING")),
                summary_path,
            )
        if not all(k in row for k in ("adv_pass", "attack", "mode", "epsilon")):
            continue
        p = int(row["adv_pass"])
        combo = (p, str(row["attack"]).lower(), str(row["mode"]).lower(), float(row["epsilon"]))
        present.add(combo)
        metrics = _row_metrics(row)
        _validate_metrics_block(
            report, metrics, category="adv_eval", label=label, path=summary_path
        )

    expected_eps = load_config()["attacks"]["epsilon_values"]
    expected: set[tuple[int, str, str, float]] = {
        (p, a, m, float(e))
        for p in ADV_PASSES
        for a in ATTACKS
        for m in MODES
        for e in expected_eps
    }
    missing = expected - present
    report.add(
        "adv_eval",
        f"{run_dir.name}: passes 1-3 grid complete",
        not missing,
        f"missing {len(missing)} combos" + (f": {sorted(missing)[:5]}..." if missing else ""),
        run_dir,
    )

    passes_seen = {c[0] for c in present}
    for p in ADV_PASSES:
        report.add(
            "adv_eval",
            f"pass {p} present",
            p in passes_seen,
            f"combos={sum(1 for c in present if c[0] == p)}/12",
            run_dir,
        )

    # Per-file consistency with summary
    file_combos: set[tuple[int, str, str, float]] = set()
    for path in sorted(run_dir.glob("pass*.json")):
        match = ADV_EVAL_JSON_RE.match(path.name)
        if not match:
            continue
        p = int(match.group("pass"))
        combo = (
            p,
            match.group("attack").lower(),
            match.group("mode").lower(),
            _parse_epsilon_tag(match.group("epsilon")),
        )
        file_combos.add(combo)
        payload = _load_json(path)
        file_metrics = _row_metrics(payload)
        summary_row = next(
            (
                r
                for r in rows
                if int(r.get("adv_pass", -1)) == p
                and str(r.get("attack", "")).lower() == combo[1]
                and str(r.get("mode", "")).lower() == combo[2]
                and float(r.get("epsilon", -1)) == combo[3]
            ),
            None,
        )
        if summary_row is None:
            report.add(
                "adv_eval",
                f"{path.name} listed in summary",
                False,
                "no matching summary row",
                path,
            )
            continue
        summary_metrics = _row_metrics(summary_row)
        for key in ("asr", "robust_accuracy"):
            ok = _metrics_close(
                float(file_metrics[key]), float(summary_metrics[key])
            )
            report.add(
                "adv_eval",
                f"{path.name} vs summary {key}",
                ok,
                f"file={file_metrics[key]:.6f} summary={summary_metrics[key]:.6f}",
                path,
            )

    missing_files = expected - file_combos
    report.add(
        "adv_eval",
        f"{run_dir.name}: per-pass JSON files complete",
        not missing_files,
        f"missing {len(missing_files)} files",
        run_dir,
    )


def validate_baseline_summary(path: Path, report: Report) -> None:
    summary = _load_json(path)
    for key in ("run_id", "models", "test_rows"):
        report.add(
            "baseline",
            f"summary has {key}",
            key in summary,
            "ok" if key in summary else f"keys={sorted(summary.keys())}",
            path,
        )
    models = summary.get("models", {})
    for model_key in ("random_forest", "mlp"):
        block = models.get(model_key, {})
        metrics = block.get("metrics", {})
        ok = all(k in metrics for k in ("accuracy", "macro_f1", "weighted_f1"))
        report.add(
            "baseline",
            f"models.{model_key}.metrics core fields",
            ok,
            f"keys={sorted(metrics.keys())}",
            path,
        )


def validate_adv_train_summary(path: Path, report: Report) -> None:
    summary = _load_json(path)
    for key in ("run_id", "pass_metrics", "passes"):
        report.add(
            "adv_train",
            f"summary has {key}",
            key in summary,
            "ok" if key in summary else f"keys={sorted(summary.keys())}",
            path,
        )
    pm = summary.get("pass_metrics", [])
    n = len(pm) if isinstance(pm, list) else 0
    report.add(
        "adv_train",
        "three pass metrics rows",
        n >= 3,
        f"pass_metrics count={n}",
        path,
    )


def validate_transfer_results(path: Path, report: Report) -> None:
    data = _load_json(path)
    for key in ("attack", "epsilon", "rf_clean_accuracy", "rf_transfer_accuracy", "asr"):
        report.add(
            "rf_transfer",
            f"transfer_results has {key}",
            key in data,
            "ok" if key in data else f"keys={sorted(data.keys())}",
            path,
        )


def test_script_loaders(results_dir: Path, report: Report) -> None:
    """Exercise plot/generate loader paths on disk (no matplotlib required)."""
    attacks_root = results_dir / "attacks"

    # plot_attack_results.load_attack_records
    try:
        from scripts import plot_attack_results as plot_mod

        for run_dir in sorted(attacks_root.iterdir()) if attacks_root.is_dir() else []:
            if not run_dir.is_dir():
                continue
            if run_dir.name.startswith("pilot_") and not any(run_dir.glob("*.json")):
                report.add(
                    "loaders",
                    f"plot.load_attack_records({run_dir.name})",
                    True,
                    "empty pilot dir (skipped)",
                    run_dir,
                )
                continue
            try:
                records = plot_mod.load_attack_records(run_dir)
                n = len(records)
                ok = n > 0
                if run_dir.name.startswith("adv_eval_"):
                    passes = {r.get("pass") for r in records if r.get("pass") is not None}
                    ok = ok and passes.issuperset({1, 2, 3})
                    detail = f"{n} records, passes={sorted(passes)}"
                elif run_dir.name.startswith("pilot_"):
                    ok = True
                    detail = f"{n} records (pilot)"
                else:
                    detail = f"{n} records"
                    if (run_dir / "manifest.json").is_file():
                        ok = ok and n >= 12
                report.add(
                    "loaders",
                    f"plot.load_attack_records({run_dir.name})",
                    ok,
                    detail,
                    run_dir,
                )
            except Exception as exc:
                report.add(
                    "loaders",
                    f"plot.load_attack_records({run_dir.name})",
                    False,
                    str(exc),
                    run_dir,
                )
    except Exception as exc:
        report.add("loaders", "import plot_attack_results", False, str(exc))

    def _latest_run(parent: Path, *, prefix: str = "", marker: str | None = None) -> Path | None:
        if not parent.is_dir():
            return None
        candidates = [
            d
            for d in parent.iterdir()
            if d.is_dir() and (not prefix or d.name.startswith(prefix))
        ]
        if marker:
            candidates = [d for d in candidates if (d / marker).is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    atk_parent = attacks_root
    atk = _latest_run(atk_parent, marker="manifest.json")
    report.add(
        "loaders",
        "latest attack manifest",
        atk is not None and (atk / "manifest.json").is_file(),
        f"selected={atk.name if atk else None}",
        atk_parent,
    )
    adv_eval = _latest_run(atk_parent, prefix="adv_eval_", marker="summary.json")
    ok = adv_eval is not None and (adv_eval / "summary.json").is_file()
    report.add(
        "loaders",
        "latest adv_eval summary",
        ok,
        f"run={adv_eval.name if adv_eval else None}",
        adv_eval or atk_parent,
    )


def scan_results(results_dir: Path, *, test_loaders: bool) -> Report:
    report = Report()

    baselines = results_dir / "baselines"
    if baselines.is_dir():
        for run_dir in sorted(baselines.iterdir()):
            if run_dir.name.startswith("pilot_"):
                continue
            sp = run_dir / "summary.json"
            if sp.is_file():
                validate_baseline_summary(sp, report)

    adv_train = results_dir / "adv_train"
    if adv_train.is_dir():
        for run_dir in sorted(adv_train.iterdir()):
            sp = run_dir / "summary.json"
            if sp.is_file():
                validate_adv_train_summary(sp, report)

    rf_transfer = results_dir / "rf_transfer"
    if rf_transfer.is_dir():
        for run_dir in sorted(rf_transfer.iterdir()):
            tp = run_dir / "transfer_results.json"
            if tp.is_file():
                validate_transfer_results(tp, report)

    attacks = results_dir / "attacks"
    if attacks.is_dir():
        for run_dir in sorted(attacks.iterdir()):
            if not run_dir.is_dir():
                continue
            if run_dir.name.startswith("adv_eval_"):
                validate_adv_eval(run_dir, report)
            elif run_dir.name.startswith("pilot_"):
                report.add(
                    "coverage",
                    f"{run_dir.name}: pilot skipped",
                    True,
                    "pilot runs excluded from full grid",
                    run_dir,
                )
            else:
                manifest_combos = validate_manifest(run_dir, report)
                file_combos = validate_attack_metric_files(run_dir, report)
                expected = expected_attack_combos()
                if not manifest_combos:
                    missing = expected - file_combos
                    report.add(
                        "coverage",
                        f"{run_dir.name}: attack grid (files only)",
                        not missing,
                        f"missing={sorted(missing)}",
                        run_dir,
                    )
                else:
                    missing_files = manifest_combos - file_combos
                    report.add(
                        "coverage",
                        f"{run_dir.name}: manifest files on disk",
                        not missing_files,
                        f"missing file combos={sorted(missing_files)}",
                        run_dir,
                    )

    if test_loaders:
        test_script_loaders(results_dir, report)

    return report


def print_report(report: Report, *, verbose: bool = False) -> None:
    passed, failed = report.counts()
    print("=" * 72)
    print("RESULTS VALIDATION REPORT")
    print("=" * 72)
    print(f"Total checks: {len(report.checks)}  |  PASS: {passed}  |  FAIL: {failed}")
    print()

    by_cat: dict[str, list[Check]] = {}
    for c in report.checks:
        by_cat.setdefault(c.category, []).append(c)

    for category in sorted(by_cat):
        items = by_cat[category]
        cat_pass = sum(1 for c in items if c.passed)
        print(f"## {category} ({cat_pass}/{len(items)} pass)")
        for c in items:
            if verbose or not c.passed:
                loc = f" [{c.path}]" if c.path else ""
                print(f"  [{c.status}] {c.name}{loc}")
                if c.detail:
                    print(f"         {c.detail}")
        print()

    if failed:
        print("FAILED CHECKS SUMMARY")
        for c in report.checks:
            if not c.passed:
                print(f"  - [{c.category}] {c.name}: {c.detail}")
    else:
        print("All checks passed.")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate thesis experiment results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Path to results/ (default: project results/)",
    )
    parser.add_argument(
        "--no-loader-test",
        action="store_true",
        help="Skip plot/generate loader smoke tests",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print all checks (default: failures and category summaries only)",
    )
    args = parser.parse_args()

    report = scan_results(args.results_dir.resolve(), test_loaders=not args.no_loader_test)
    print_report(report, verbose=args.verbose)
    _, failed = report.counts()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
