#!/usr/bin/env python3
"""Check a privacy-minimized, AGGREGATE-ONLY controlled-experiment package.

Recalculates summary metrics FROM STORED AGGREGATE COUNTS/WINDOWS.
Does not re-evaluate the original individual participant runs, recover any
private recordings, or establish that participants gave consent.
Python 3.10+ standard library only.
"""
from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(None in row for row in rows):
        raise ValueError(f"Missing/invalid rows: {name}")
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def approx(actual: float, expected: float, desc: str, tol: float = 0.000002) -> None:
    require(math.isfinite(actual) and abs(actual - expected) <= tol,
            f"{desc}: calculated={actual}, stored={expected}")


def divide(x: float, y: float) -> float:
    return x / y if y else 0.0


def metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    p = divide(tp, tp + fp)
    r = divide(tp, tp + fn)
    sp = divide(tn, tn + fp)
    pn = divide(tn, tn + fn)
    fp1 = divide(2 * p * r, p + r)
    fn1 = divide(2 * pn * sp, pn + sp)
    return {
        "accuracy": divide(tp + tn, tp + tn + fp + fn),
        "precision": p, "recall": r, "specificity": sp,
        "f1_positive": fp1, "f1_negative": fn1,
        "macro_f1": (fp1 + fn1) / 2,
    }


def check_confusion(name: str, group_key: str,
                    expected: dict[str, tuple[int, int, int, int]]) -> None:
    rows = load(name)
    require(len(rows) == len(expected), f"Unexpected row count in {name}")
    found: set[str] = set()
    for row in rows:
        key = row[group_key]
        require(key in expected and key not in found, f"Unexpected/duplicate group {key}")
        found.add(key)
        counts = tuple(int(row[k]) for k in ("tp", "tn", "fp", "fn"))
        require(counts == expected[key], f"Unexpected confusion counts {name}/{key}")
        if "runs" in row:
            require(int(row["runs"]) == sum(counts), f"Run count {name}/{key}")
        for metric, actual in metrics(*counts).items():
            approx(actual, float(row[metric]), f"{name}/{key}/{metric}")
    require(set(expected) == found, f"Missing groups: {name}")


def check_cues() -> None:
    expected = {
        ("global", "gaze"): (7, 5, 2, 3, 0),
        ("global", "nod"): (6, 2, 4, 3, 0),
        ("personalized", "gaze"): (7, 5, 2, 3, 0),
        ("personalized", "nod"): (6, 5, 1, 3, 0),
    }
    seen = set()
    for row in load("h2_pooled_cues.csv"):
        key = (row["mode"], row["cue"])
        require(key in expected and key not in seen, f"Invalid cue group {key}")
        seen.add(key)
        v = tuple(int(row[k]) for k in (
            "positive_runs", "detected_positive_runs",
            "missed_positive_runs", "normal_runs", "normal_false_activations"))
        require(v == expected[key], f"Cue group counts {key}")
        require(v[0] == v[1] + v[2], f"Missing positive cue runs {key}")
        approx(divide(v[1], v[0]), float(row["recall"]), f"Cue recall {key}")
        approx(divide(v[3] - v[4], v[3]), float(row["specificity"]),
               f"Cue specificity {key}")
    require(seen == set(expected), "Missing H2 cue groups")


def check_h3() -> None:
    expected_counts = {
        "cpu_only": 8,
        "gpu_only": 7,
        "gpu_with_local_llm": 6,
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in load("h3_window_summaries.csv"):
        require(r["condition"] in expected_counts, "Unexpected benchmark condition")
        grouped[r["condition"]].append(r)
    require(set(grouped) == set(expected_counts), "Missing benchmark group")
    for key, n in expected_counts.items():
        group = grouped[key]
        require(len(group) == n, f"Unexpected {key} window count")
        require({int(r["window_index"]) for r in group} == set(range(1, n + 1)),
                f"Repeated window index in {key}")
        for r in group:
            require(int(r["callbacks"]) == 30, f"Unexpected callbacks {key}")
            for col in ("e2e_mean_ms", "e2e_p95_ms", "mp_mean_ms", "callback_rate"):
                require(math.isfinite(float(r[col])) and float(r[col]) >= 0,
                        f"Invalid timing {key}/{col}")
            require(float(r["e2e_mean_ms"]) <= float(r["e2e_p95_ms"]) + 5,
                    f"Invalid approximate mean/window p95 {key}")
    def average(key: str, field: str) -> float:
        return sum(float(x[field]) for x in grouped[key]) / len(grouped[key])
    approx(average("cpu_only", "e2e_mean_ms"), 77.05, "CPU E2E", 0.02)
    approx(average("gpu_only", "e2e_mean_ms"), 91.08, "GPU E2E", 0.02)
    approx(average("gpu_with_local_llm", "e2e_mean_ms"), 286.72,
           "GPU+LLM E2E", 0.02)
    approx(average("gpu_with_local_llm", "e2e_p95_ms"), 394.33,
           "Arithmetic mean of window p95", 0.02)
    approx(average("gpu_with_local_llm", "mp_mean_ms"), 226.67,
           "GPU+LLM MediaPipe stage", 0.02)
    approx(average("gpu_with_local_llm", "callback_rate"), 6.68,
           "GPU+LLM callback rate", 0.02)
    require(sum(int(x["callbacks"]) for x in grouped["gpu_with_local_llm"]) == 180,
            "Expected 180 GPU+LLM callbacks")


def check_table10() -> None:
    rows = load("table10_na_aggregate.csv")
    require(len(rows) == 3, "Expected three nonconcurrent run groups")
    require({r["group"] for r in rows} == {"group_1", "group_2", "group_3"},
            "Invalid memory evidence group")
    for r in rows:
        require(int(r["records"]) >= 1, "Empty benchmark group")
        require(r["concurrent_app"] == "0"
                and r["peak_app_rss_kb"] == "NA",
                "Nonconcurrent application RSS should be N/A, not zero")


def check_checksums() -> None:
    manifest = ROOT / "SHA256SUMS.txt"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    expect = set()
    for line in lines:
        digest, rel = line.split("  ", 1)
        require(len(digest) == 64, "Invalid SHA256")
        require(not rel.startswith("/") and ".." not in Path(rel).parts,
                "Invalid manifest path")
        path = ROOT / rel
        require(path.is_file(), f"Missing manifest file {rel}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                f"Checksum mismatch {rel}")
        require(rel not in expect, f"Duplicate manifest path {rel}")
        expect.add(rel)
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
              if p.is_file() and ".git" not in p.relative_to(ROOT).parts
              and p.name != "SHA256SUMS.txt"}
    require(expect == actual, "Unlisted files or missing manifest entries")


def main() -> None:
    check_confusion("h1_mode_confusion.csv", "mode", {
        "A": (1, 2, 0, 9), "B": (5, 2, 0, 5),
        "C": (5, 2, 0, 5), "D": (3, 2, 0, 7),
        "E": (4, 2, 0, 6),
    })
    print("[PASS] H1: five aggregate confusion matrices / 12 controlled runs")
    check_confusion("h2_pooled_confusion.csv", "mode", {
        "global": (1, 3, 0, 9),
        "personalized": (5, 3, 0, 5),
    })
    check_cues()
    print("[PASS] H2: pooled counts, no per-person/per-run records")
    check_h3()
    print("[PASS] H3: 8 CPU, 7 GPU, 6 concurrent timing windows")
    check_table10()
    print("[PASS] Table 10: nonconcurrent APK RSS marked N/A")
    check_checksums()
    print("[PASS] SHA-256 manifest")
    print("[AUDIT] COMPLETE: numerical consistency only; not an ethics waiver")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, TypeError) as err:
        raise SystemExit("[FAIL] " + str(err))
