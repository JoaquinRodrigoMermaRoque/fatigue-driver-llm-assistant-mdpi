#!/usr/bin/env python3
import argparse
import re
import sys

FIELDS = [
    "delegate",
    "window",
    "samples",
    "window_start_ms",
    "window_end_ms",
    "e2e_mean_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_max_ms",
    "pre_mean_ms",
    "pre_p50_ms",
    "pre_p95_ms",
    "pre_max_ms",
    "mp_mean_ms",
    "mp_p50_ms",
    "mp_p95_ms",
    "mp_max_ms",
    "callback_fps",
]

PAIR_RE = re.compile(r"([A-Za-z0-9_]+)=([^,\s]+)")


INTEGER_FIELDS = {
    "window",
    "samples",
    "window_start_ms",
    "window_end_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_max_ms",
    "pre_p50_ms",
    "pre_p95_ms",
    "pre_max_ms",
    "mp_p50_ms",
    "mp_p95_ms",
    "mp_max_ms",
}

FLOAT_FIELDS = {
    "e2e_mean_ms",
    "pre_mean_ms",
    "mp_mean_ms",
    "callback_fps",
}


def validate_row(values):
    try:
        ints = {
            key: int(values[key])
            for key in INTEGER_FIELDS
        }
        floats = {
            key: float(values[key])
            for key in FLOAT_FIELDS
        }
    except (KeyError, ValueError):
        return False, "non_numeric_field"

    if ints["samples"] <= 0:
        return False, "invalid_sample_count"

    if ints["window_end_ms"] < ints["window_start_ms"]:
        return False, "invalid_window_clock"

    for prefix in ("e2e", "pre", "mp"):
        p50 = ints[f"{prefix}_p50_ms"]
        p95 = ints[f"{prefix}_p95_ms"]
        maximum = ints[f"{prefix}_max_ms"]
        mean = floats[f"{prefix}_mean_ms"]

        if min(p50, p95, maximum, mean) < 0:
            return False, f"negative_{prefix}"

        if not (p50 <= p95 <= maximum):
            return False, f"invalid_{prefix}_percentiles"

        if mean > maximum:
            return False, f"mean_exceeds_{prefix}_max"

    if floats["callback_fps"] < 0:
        return False, "negative_callback_fps"

    mean_sum_error = abs(
        floats["e2e_mean_ms"]
        - floats["pre_mean_ms"]
        - floats["mp_mean_ms"]
    )
    if mean_sum_error > 0.10:
        return False, "timing_components_do_not_sum"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logcat_path")
    parser.add_argument(
        "--min-window-start-ms",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--drop-first-window",
        action="store_true",
        help=(
            "Discard the first complete telemetry window after the selected "
            "start point because it may contain callbacks accumulated before "
            "the measurement began."
        ),
    )
    parser.add_argument(
        "--after-marker",
        default=None,
        help=(
            "Ignore logcat lines until this exact marker substring has been "
            "seen. Recommended for benchmark RUN_ID isolation."
        ),
    )
    args = parser.parse_args()

    print("\t".join(FIELDS))
    emitted_candidates = 0
    marker_seen = args.after_marker is None

    with open(args.logcat_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not marker_seen:
                if args.after_marker in line:
                    marker_seen = True
                continue
            if "H7_9_PERF" not in line or "delegate=" not in line:
                continue

            values = dict(PAIR_RE.findall(line))
            if not all(field in values for field in FIELDS):
                continue

            if (
                args.min_window_start_ms is not None
                and int(values["window_start_ms"])
                < args.min_window_start_ms
            ):
                continue

            valid, reason = validate_row(values)
            if not valid:
                print(
                    f"Skipping malformed H7_9_PERF row: {reason}",
                    file=sys.stderr,
                )
                continue

            emitted_candidates += 1
            if args.drop_first_window and emitted_candidates == 1:
                continue

            print("\t".join(values[field] for field in FIELDS))


if __name__ == "__main__":
    main()
