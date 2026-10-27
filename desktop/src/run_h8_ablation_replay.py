"""H8 offline A-E replay aligned with the validated Android implementation.

Replays one H8 capture through ablation modes A-E.

For the H1 ablation, the primary replay keeps the temporal cue outputs already
computed by Android fixed (closure, PERCLOS, yawn, gaze-away and nod events)
and changes only the active cue set/fusion. This guarantees that A-E use the
same logged frames and preserves real invalid/no-face effects from the phone.

A secondary raw-cue reconstruction is retained only as an audit diagnostic.
This script is intentionally separate from the older desktop ablation runner.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, deque
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import fmean

try:
    from .temporal import (
        EyeTemporalAnalyzer,
        GazeTemporalAnalyzer,
        HeadPoseTemporalAnalyzer,
        MouthTemporalAnalyzer,
    )
except ImportError:
    from temporal import (
        EyeTemporalAnalyzer,
        GazeTemporalAnalyzer,
        HeadPoseTemporalAnalyzer,
        MouthTemporalAnalyzer,
    )


EAR_THRESHOLD = 0.14
MAR_THRESHOLD = 0.27
EYE_WINDOW_S = 15.0
GAZE_LOW = 0.37
GAZE_HIGH = 0.62
GAZE_VERTICAL_LOW = 0.38
GAZE_VERTICAL_HIGH = 0.42
MAX_GAZE_ASYMMETRY = 0.20

POSE_CANONICAL_PITCH = 12.2
POSE_CANONICAL_YAW = -2.9
POSE_CANONICAL_ROLL = 0.1

MIN_AVAILABLE_WEIGHT_RATIO = 0.55
ACKNOWLEDGEMENT_POLICY = "automatic_each_frame"

WEIGHTS = {
    "closure": 0.30,
    "perclos": 0.25,
    "yawn": 0.15,
    "gaze": 0.10,
    "nod": 0.20,
}

MODES = {
    "A": ("closure",),
    "B": ("closure", "yawn"),
    "C": ("closure", "yawn", "perclos"),
    "D": ("closure", "yawn", "perclos", "gaze"),
    "E": ("closure", "yawn", "perclos", "gaze", "nod"),
}

REQUIRED_FIELDS = {
    "feature_run_id",
    "subject_id",
    "clip_id",
    "source",
    "frame_index",
    "t_ms",
    "face_detected",
    "valid_pose",
    "ear",
    "mar",
    "gaze_h_raw",
    "gaze_v_raw",
    "gaze_h_asymmetry",
    "pitch_deg",
    "yaw_deg",
    "roll_deg",
    "calibration_yaw_base_deg",
    "calibration_pitch_base_deg",
    "calibration_roll_base_deg",
    "calibration_gaze_h",
    "calibration_gaze_v",
    "score",
    "state",
    "state_event",
    "closure_s",
    "perclos",
    "yawn_event",
    "gaze_away_s",
    "nod_event",
}

OUTPUT_FIELDS = (
    "feature_run_id",
    "subject_id",
    "clip_id",
    "source",
    "frame_index",
    "t_ms",
    "config_id",
    "active_cues",
    "acknowledgement_policy",
    "data_sufficient",
    "available_weight_ratio",
    "score",
    "state",
    "state_event",
    "evidence_closure",
    "evidence_perclos",
    "evidence_yawn",
    "evidence_gaze",
    "evidence_nod",
    "closure_s",
    "perclos",
    "yawn_event",
    "yawns_last_5_min",
    "gaze_away_s",
    "nod_event",
    "nods_last_2_min",
    "capture_score",
    "capture_state",
    "capture_state_event",
)


def fval(row: dict[str, str], key: str) -> float:
    text = row.get(key, "").strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def ival(row: dict[str, str], key: str) -> int:
    value = fval(row, key)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{key} no es entero: {value!r}")
    return int(value)


def bval(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() in {"1", "true", "yes", "si"}


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def ramp(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


@dataclass(frozen=True)
class Evidence:
    closure: float
    perclos: float
    yawn: float
    gaze: float
    nod: float

    def as_dict(self) -> dict[str, float]:
        return {
            "closure": self.closure,
            "perclos": self.perclos,
            "yawn": self.yawn,
            "gaze": self.gaze,
            "nod": self.nod,
        }


class RollingEventCounter:
    def __init__(self, window_s: float):
        self.window_ms = int(window_s * 1000.0)
        self.timestamps: deque[int] = deque()
        self.last_timestamp_ms: int | None = None
        self.previous_event = False

    def reset(self) -> None:
        self.timestamps.clear()
        self.last_timestamp_ms = None
        self.previous_event = False

    def update(self, timestamp_ms: int, event: bool, valid: bool) -> int:
        if self.last_timestamp_ms is not None and timestamp_ms < self.last_timestamp_ms:
            raise ValueError("Los timestamps deben ser monotónicos")
        self.last_timestamp_ms = timestamp_ms

        cutoff = timestamp_ms - self.window_ms
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

        if not valid:
            self.previous_event = False
            return len(self.timestamps)

        if event and not self.previous_event:
            self.timestamps.append(timestamp_ms)

        self.previous_event = event
        return len(self.timestamps)


class RecentEventWindows:
    def __init__(self):
        self.yawns = RollingEventCounter(300.0)
        self.nods = RollingEventCounter(120.0)

    def reset(self) -> None:
        self.yawns.reset()
        self.nods.reset()

    def update(
        self,
        timestamp_ms: int,
        yawn_event: bool,
        nod_event: bool,
        yawn_valid: bool,
        nod_valid: bool,
    ) -> tuple[int, int]:
        return (
            self.yawns.update(timestamp_ms, yawn_event, yawn_valid),
            self.nods.update(timestamp_ms, nod_event, nod_valid),
        )


class AndroidStateMachine:
    def __init__(self):
        self.state = "NORMAL"
        self.candidate_key: str | None = None
        self.candidate_since_ms: int | None = None
        self.last_timestamp_ms: int | None = None
        self.critical_acknowledged = False

    def _validate(self, timestamp_ms: int) -> None:
        if self.last_timestamp_ms is not None and timestamp_ms < self.last_timestamp_ms:
            raise ValueError("Los timestamps deben ser monotónicos")
        self.last_timestamp_ms = timestamp_ms

    def _clear(self) -> None:
        self.candidate_key = None
        self.candidate_since_ms = None

    def _enter(self, target: str, event: str) -> tuple[str, str | None]:
        if self.state == target:
            self._clear()
            return self.state, None
        self.state = target
        self._clear()
        if target in {"CRITICAL", "NORMAL"}:
            self.critical_acknowledged = False
        return self.state, event

    def _persist(
        self,
        key: str,
        target: str,
        dwell_ms: int,
        timestamp_ms: int,
        event: str,
    ) -> tuple[str, str | None]:
        if self.candidate_key != key:
            self.candidate_key = key
            self.candidate_since_ms = timestamp_ms
            elapsed = dwell_ms == 0
        else:
            elapsed = (
                timestamp_ms - int(self.candidate_since_ms)
                >= dwell_ms
            )
        if elapsed:
            return self._enter(target, event)
        return self.state, None

    def invalidate(self, timestamp_ms: int) -> tuple[str, str | None]:
        self._validate(timestamp_ms)
        self._clear()
        return self.state, None

    def update(
        self,
        timestamp_ms: int,
        score: float,
        closure_s: float,
        acknowledged: bool,
    ) -> tuple[str, str | None]:
        self._validate(timestamp_ms)

        if closure_s >= 3.0:
            if self.state != "CRITICAL":
                return self._enter("CRITICAL", "prolonged_eye_closure")
            self.critical_acknowledged = False
            self._clear()
            return self.state, None

        if self.state != "NORMAL" and score < 0.20:
            return self._persist(
                "any_to_normal", "NORMAL", 5000, timestamp_ms, "state_normal"
            )

        if self.state == "CRITICAL":
            acknowledgement_event = None
            if acknowledged and not self.critical_acknowledged:
                acknowledgement_event = "critical_acknowledged"
            if acknowledged:
                self.critical_acknowledged = True
            if not self.critical_acknowledged:
                self._clear()
                return self.state, None
            if score < 0.40:
                state, event = self._persist(
                    "critical_to_recovery",
                    "RECOVERY",
                    30000,
                    timestamp_ms,
                    "state_recovery",
                )
                return state, event or acknowledgement_event
            self._clear()
            return self.state, acknowledgement_event

        if score >= 0.75 and self.state == "FATIGUE":
            return self._persist(
                "to_critical", "CRITICAL", 10000, timestamp_ms, "state_critical"
            )

        if score >= 0.35 and self.state == "SUSPECTED":
            return self._persist(
                "to_fatigue", "FATIGUE", 3000, timestamp_ms, "state_fatigue"
            )

        if score >= 0.20 and self.state in {"NORMAL", "RECOVERY"}:
            return self._persist(
                "to_suspected", "SUSPECTED", 1500, timestamp_ms, "state_suspected"
            )

        if self.state == "SUSPECTED" and score < 0.20:
            return self._persist(
                "suspected_to_normal", "NORMAL", 10000, timestamp_ms, "state_normal"
            )

        if self.state == "FATIGUE" and score < 0.35:
            return self._persist(
                "fatigue_to_recovery", "RECOVERY", 15000, timestamp_ms, "state_recovery"
            )

        if self.state == "RECOVERY" and score < 0.20:
            return self._persist(
                "recovery_to_normal", "NORMAL", 10000, timestamp_ms, "state_normal"
            )

        self._clear()
        return self.state, None


class AndroidH8Engine:
    def __init__(self, mode: str):
        self.mode = mode
        self.active_cues = MODES[mode]
        self.events = RecentEventWindows()
        self.state_machine = AndroidStateMachine()

    def update(
        self,
        *,
        timestamp_ms: int,
        closure_s: float,
        perclos: float,
        gaze_away_s: float,
        yawn_event: bool,
        nod_event: bool,
        face_valid: bool,
        eye_valid: bool,
        mouth_valid: bool,
        gaze_valid: bool,
        pose_valid: bool,
    ) -> dict[str, object]:
        yawns, nods = self.events.update(
            timestamp_ms,
            yawn_event,
            nod_event,
            face_valid and mouth_valid,
            face_valid and pose_valid,
        )

        safe_closure = closure_s if face_valid and eye_valid else 0.0
        safe_perclos = perclos if face_valid and eye_valid else 0.0
        safe_gaze = gaze_away_s if face_valid and gaze_valid else 0.0

        for value in (safe_closure, safe_perclos, safe_gaze):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("Métrica válida no finita/no negativa")

        evidence = Evidence(
            closure=ramp(safe_closure, 0.40, 2.00),
            perclos=ramp(safe_perclos, 0.15, 0.40),
            yawn=ramp(float(yawns), 0.0, 3.0),
            gaze=ramp(safe_gaze, 1.50, 5.00),
            nod=ramp(float(nods), 0.0, 3.0),
        )
        ev = evidence.as_dict()
        denominator = sum(WEIGHTS[c] for c in self.active_cues)
        score = sum(WEIGHTS[c] * ev[c] for c in self.active_cues) / denominator
        score = clamp01(score)

        availability_map = {
            "closure": face_valid and eye_valid,
            "perclos": face_valid and eye_valid,
            "yawn": face_valid and mouth_valid,
            "gaze": face_valid and gaze_valid,
            "nod": face_valid and pose_valid,
        }
        available_weight_ratio = (
            sum(
                WEIGHTS[c]
                for c in self.active_cues
                if availability_map[c]
            )
            / denominator
        )
        available_weight_ratio = round(available_weight_ratio, 12)
        sufficient = (
            face_valid
            and eye_valid
            and available_weight_ratio >= MIN_AVAILABLE_WEIGHT_RATIO
        )

        if sufficient:
            state, state_event = self.state_machine.update(
                timestamp_ms,
                score,
                safe_closure,
                acknowledged=True,
            )
        else:
            state, state_event = self.state_machine.invalidate(timestamp_ms)

        if state_event == "state_normal":
            self.events.reset()

        return {
            "score": score,
            "evidence": evidence,
            "state": state,
            "state_event": state_event or "",
            "yawns": yawns,
            "nods": nods,
            "data_sufficient": sufficient,
            "available_weight_ratio": available_weight_ratio,
        }


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_FIELDS - fields
        if missing:
            raise RuntimeError(
                "Faltan columnas H8: " + ", ".join(sorted(missing))
            )
        rows = list(reader)

    if not rows:
        raise RuntimeError("El CSV H8 está vacío")

    rows.sort(key=lambda r: ival(r, "frame_index"))

    run_ids = {r["feature_run_id"] for r in rows}
    sources = {r["source"] for r in rows}
    if len(run_ids) != 1 or len(sources) != 1:
        raise RuntimeError("El archivo debe contener un único run/source")

    previous_frame = -1
    previous_t = -1
    for row in rows:
        frame = ival(row, "frame_index")
        t_ms = ival(row, "t_ms")
        if frame <= previous_frame:
            raise RuntimeError("frame_index no es estrictamente creciente")
        if t_ms <= previous_t:
            raise RuntimeError("t_ms no es estrictamente creciente")
        previous_frame = frame
        previous_t = t_ms

    calibration_keys = (
        "calibration_yaw_base_deg",
        "calibration_pitch_base_deg",
        "calibration_roll_base_deg",
        "calibration_gaze_h",
        "calibration_gaze_v",
    )
    reference = tuple(fval(rows[0], k) for k in calibration_keys)
    if not all(math.isfinite(v) for v in reference):
        raise RuntimeError("El run no contiene calibración final válida")

    for row in rows[1:]:
        current = tuple(fval(row, k) for k in calibration_keys)
        if any(abs(a - b) > 1e-7 for a, b in zip(reference, current)):
            raise RuntimeError("La calibración cambió dentro del mismo run")

    return rows


def calibrated_gaze(row: dict[str, str]) -> tuple[float, float]:
    raw_h = fval(row, "gaze_h_raw")
    raw_v = fval(row, "gaze_v_raw")
    center_h = fval(row, "calibration_gaze_h")
    center_v = fval(row, "calibration_gaze_v")

    canonical_h = (GAZE_LOW + GAZE_HIGH) / 2.0
    canonical_v = (GAZE_VERTICAL_LOW + GAZE_VERTICAL_HIGH) / 2.0

    return (
        clamp01(raw_h - center_h + canonical_h),
        clamp01(raw_v - center_v + canonical_v),
    )


def calibrated_pose(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        fval(row, "pitch_deg")
        - fval(row, "calibration_pitch_base_deg")
        + POSE_CANONICAL_PITCH,
        fval(row, "yaw_deg")
        - fval(row, "calibration_yaw_base_deg")
        + POSE_CANONICAL_YAW,
        fval(row, "roll_deg")
        - fval(row, "calibration_roll_base_deg")
        + POSE_CANONICAL_ROLL,
    )



def replay_mode_from_captured_temporals(
    rows: list[dict[str, str]],
    mode: str,
) -> list[dict[str, object]]:
    """Replay A-E using temporal cue outputs already computed by Android.

    H1 changes only the active cue set/fusion. Therefore the fairest replay
    keeps Android's per-frame temporal outputs fixed and changes only mode A-E.
    This also preserves invalid/no-face effects that occurred between logged
    result frames and cannot be reconstructed exactly from the raw-feature CSV.
    """
    engine = AndroidH8Engine(mode)
    out: list[dict[str, object]] = []

    for row in rows:
        t_ms = ival(row, "t_ms")
        face_valid = bval(row, "face_detected")
        eye_valid = bval(row, "valid_eye")
        mouth_valid = bval(row, "valid_mouth")
        gaze_valid = bval(row, "valid_gaze")
        pose_valid = bval(row, "valid_pose")

        closure = fval(row, "closure_s")
        perclos = fval(row, "perclos")
        gaze_away = fval(row, "gaze_away_s")

        # On the first valid eye frame Android may export NaN PERCLOS while
        # valid_eye is false. Safe zeros reproduce FatigueEngine semantics.
        closure_for_engine = closure if math.isfinite(closure) else 0.0
        perclos_for_engine = perclos if math.isfinite(perclos) else 0.0
        gaze_for_engine = gaze_away if math.isfinite(gaze_away) else 0.0

        metrics = engine.update(
            timestamp_ms=t_ms,
            closure_s=closure_for_engine,
            perclos=perclos_for_engine,
            gaze_away_s=gaze_for_engine,
            yawn_event=bval(row, "yawn_event"),
            nod_event=bval(row, "nod_event"),
            face_valid=face_valid,
            eye_valid=eye_valid,
            mouth_valid=mouth_valid,
            gaze_valid=gaze_valid,
            pose_valid=pose_valid,
        )
        ev: Evidence = metrics["evidence"]

        out.append(
            {
                "feature_run_id": row["feature_run_id"],
                "subject_id": row["subject_id"],
                "clip_id": row["clip_id"],
                "source": row["source"],
                "frame_index": ival(row, "frame_index"),
                "t_ms": t_ms,
                "config_id": mode,
                "active_cues": ";".join(MODES[mode]),
                "acknowledgement_policy": ACKNOWLEDGEMENT_POLICY,
                "data_sufficient": int(bool(metrics["data_sufficient"])),
                "available_weight_ratio": metrics["available_weight_ratio"],
                "score": metrics["score"],
                "state": metrics["state"],
                "state_event": metrics["state_event"],
                "evidence_closure": ev.closure,
                "evidence_perclos": ev.perclos,
                "evidence_yawn": ev.yawn,
                "evidence_gaze": ev.gaze,
                "evidence_nod": ev.nod,
                "closure_s": closure_for_engine,
                "perclos": perclos_for_engine,
                "yawn_event": int(bval(row, "yawn_event")),
                "yawns_last_5_min": metrics["yawns"],
                "gaze_away_s": gaze_for_engine,
                "nod_event": int(bval(row, "nod_event")),
                "nods_last_2_min": metrics["nods"],
                "capture_score": fval(row, "score"),
                "capture_state": row["state"],
                "capture_state_event": row["state_event"],
            }
        )

    return out


def replay_mode(
    rows: list[dict[str, str]],
    mode: str,
    calibration_mode: str = "personalized",
) -> list[dict[str, object]]:
    if calibration_mode not in {"personalized", "global"}:
        raise ValueError(
            "calibration_mode must be 'personalized' or 'global'"
        )
    eye = EyeTemporalAnalyzer(
        window_s=EYE_WINDOW_S,
        ear_threshold=EAR_THRESHOLD,
        min_blink_s=0.08,
        max_blink_s=0.80,
    )
    mouth = MouthTemporalAnalyzer(
        mar_threshold=MAR_THRESHOLD,
        min_yawn_s=1.50,
    )
    # Android H8 has verticalEnabled=false. Extreme vertical bounds disable
    # vertical classification while preserving the desktop temporal filter.
    gaze = GazeTemporalAnalyzer(
        horizontal_low=GAZE_LOW,
        horizontal_high=GAZE_HIGH,
        vertical_low=-1e9,
        vertical_high=1e9,
        min_away_s=1.50,
        smoothing_tau_s=0.15,
    )
    pose = HeadPoseTemporalAnalyzer(
        baseline_pitch_deg=POSE_CANONICAL_PITCH,
        baseline_yaw_deg=POSE_CANONICAL_YAW,
        baseline_roll_deg=POSE_CANONICAL_ROLL,
        yaw_threshold_deg=20.0,
        pitch_threshold_deg=15.0,
        roll_threshold_deg=20.0,
        min_away_s=1.50,
        nod_down_threshold_deg=15.0,
        nod_return_threshold_deg=8.0,
        min_nod_s=0.30,
        max_nod_s=2.50,
        smoothing_tau_s=0.15,
    )
    engine = AndroidH8Engine(mode)
    out: list[dict[str, object]] = []

    for row in rows:
        t_ms = ival(row, "t_ms")
        t_s = t_ms / 1000.0
        face_valid = bval(row, "face_detected")

        ear_raw = fval(row, "ear")
        mar_raw = fval(row, "mar")
        raw_h = fval(row, "gaze_h_raw")
        raw_v = fval(row, "gaze_v_raw")
        asym_h = fval(row, "gaze_h_asymmetry")

        geometry_valid = face_valid
        eye_m = eye.update(t_s, ear_raw, geometry_valid)
        mouth_m = mouth.update(t_s, mar_raw, geometry_valid)

        gaze_inputs_finite = all(
            math.isfinite(v) for v in (raw_h, raw_v, asym_h)
        )
        gaze_input_valid = (
            geometry_valid
            and math.isfinite(ear_raw)
            and ear_raw >= EAR_THRESHOLD
            and gaze_inputs_finite
            and asym_h <= MAX_GAZE_ASYMMETRY
        )
        if gaze_inputs_finite:
            if calibration_mode == "personalized":
                gaze_h, gaze_v = calibrated_gaze(row)
            else:
                gaze_h = clamp01(raw_h)
                gaze_v = clamp01(raw_v)
        else:
            gaze_h = gaze_v = float("nan")
        gaze_m = gaze.update(t_s, gaze_h, gaze_v, gaze_input_valid)

        raw_pose = (
            fval(row, "pitch_deg"),
            fval(row, "yaw_deg"),
            fval(row, "roll_deg"),
        )
        pose_input_valid = (
            face_valid
            and bval(row, "valid_pose")
            and all(math.isfinite(v) for v in raw_pose)
        )
        if pose_input_valid:
            if calibration_mode == "personalized":
                pitch, yaw, roll = calibrated_pose(row)
            else:
                pitch, yaw, roll = raw_pose
        else:
            pitch = yaw = roll = float("nan")
        pose_m = pose.update(t_s, pitch, yaw, roll, pose_input_valid)

        eye_valid = face_valid and math.isfinite(eye_m.perclos)
        mouth_valid = face_valid
        gaze_valid = gaze_m.direction != "NO_VALIDA"
        pose_valid = pose_m.direction != "NO_VALIDA"
        perclos = eye_m.perclos if math.isfinite(eye_m.perclos) else 0.0

        metrics = engine.update(
            timestamp_ms=t_ms,
            closure_s=eye_m.closure_duration_s,
            perclos=perclos,
            gaze_away_s=gaze_m.away_duration_s,
            yawn_event=mouth_m.yawn_event,
            nod_event=pose_m.nod_event,
            face_valid=face_valid,
            eye_valid=eye_valid,
            mouth_valid=mouth_valid,
            gaze_valid=gaze_valid,
            pose_valid=pose_valid,
        )
        ev: Evidence = metrics["evidence"]

        out.append(
            {
                "feature_run_id": row["feature_run_id"],
                "subject_id": row["subject_id"],
                "clip_id": row["clip_id"],
                "source": row["source"],
                "frame_index": ival(row, "frame_index"),
                "t_ms": t_ms,
                "config_id": mode,
                "active_cues": ";".join(MODES[mode]),
                "acknowledgement_policy": ACKNOWLEDGEMENT_POLICY,
                "data_sufficient": int(bool(metrics["data_sufficient"])),
                "available_weight_ratio": metrics["available_weight_ratio"],
                "score": metrics["score"],
                "state": metrics["state"],
                "state_event": metrics["state_event"],
                "evidence_closure": ev.closure,
                "evidence_perclos": ev.perclos,
                "evidence_yawn": ev.yawn,
                "evidence_gaze": ev.gaze,
                "evidence_nod": ev.nod,
                "closure_s": eye_m.closure_duration_s,
                "perclos": perclos,
                "yawn_event": int(mouth_m.yawn_event),
                "yawns_last_5_min": metrics["yawns"],
                "gaze_away_s": gaze_m.away_duration_s,
                "nod_event": int(pose_m.nod_event),
                "nods_last_2_min": metrics["nods"],
                "capture_score": fval(row, "score"),
                "capture_state": row["state"],
                "capture_state_event": row["state_event"],
            }
        )

        # CameraFragment resets the temporal cue analyzers after state_normal.
        if metrics["state_event"] == "state_normal":
            eye.reset()
            mouth.reset()
            gaze.reset()
            pose.reset()

    return out


def verify_same_frames(by_mode: dict[str, list[dict[str, object]]]) -> None:
    reference = None
    for mode, rows in by_mode.items():
        ids = tuple((r["frame_index"], r["t_ms"]) for r in rows)
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Frames duplicados en modo {mode}")
        if reference is None:
            reference = ids
        elif ids != reference:
            raise RuntimeError(f"El modo {mode} no usa exactamente los mismos frames")


def write_output(path: Path, by_mode: dict[str, list[dict[str, object]]]) -> None:
    verify_same_frames(by_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for mode in MODES:
            writer.writerows(by_mode[mode])


def print_summary(by_mode: dict[str, list[dict[str, object]]]) -> None:
    print()
    print(f"{'Modo':<5} {'Frames':>7} {'Score mean':>11} {'Score max':>10} Estados")
    for mode in MODES:
        rows = by_mode[mode]
        scores = [float(r["score"]) for r in rows]
        states = Counter(str(r["state"]) for r in rows)
        states_text = ",".join(f"{k}:{v}" for k, v in sorted(states.items()))
        print(
            f"{mode:<5} {len(rows):>7} {fmean(scores):>11.4f} "
            f"{max(scores):>10.4f} {states_text}"
        )

    e_rows = by_mode["E"]
    score_diffs = [
        abs(float(r["score"]) - float(r["capture_score"]))
        for r in e_rows
        if math.isfinite(float(r["capture_score"]))
    ]
    state_mismatch = sum(
        str(r["state"]) != str(r["capture_state"])
        for r in e_rows
    )
    event_mismatch = sum(
        str(r["state_event"]) != str(r["capture_state_event"])
        for r in e_rows
    )
    print()
    print("Validación E contra ejecución Android capturada:")
    print(
        "max_abs_score_diff =",
        f"{max(score_diffs) if score_diffs else float('nan'):.8f}",
    )
    print("state_mismatches =", state_mismatch)
    print("event_mismatches =", event_mismatch)



def _max_abs_pair_diff(
    rows: list[dict[str, object]],
    replay_key: str,
    capture_rows: list[dict[str, str]],
    capture_key: str,
) -> float:
    diffs = []
    for replay_row, capture_row in zip(rows, capture_rows):
        a = float(replay_row[replay_key])
        b = fval(capture_row, capture_key)
        if math.isfinite(a) and math.isfinite(b):
            diffs.append(abs(a - b))
    return max(diffs) if diffs else float("nan")


def _event_mismatch_count(
    rows: list[dict[str, object]],
    replay_key: str,
    capture_rows: list[dict[str, str]],
    capture_key: str,
) -> int:
    return sum(
        int(replay_row[replay_key]) != int(bval(capture_row, capture_key))
        for replay_row, capture_row in zip(rows, capture_rows)
    )


def validate_engine_from_captured_temporals(
    capture_rows: list[dict[str, str]],
) -> tuple[float, int, int]:
    """Isolate fusion/state logic from raw-cue temporal reconstruction."""
    engine = AndroidH8Engine("E")
    score_diffs = []
    state_mismatches = 0
    event_mismatches = 0

    for row in capture_rows:
        face_valid = bval(row, "face_detected")
        eye_valid = bval(row, "valid_eye")
        mouth_valid = bval(row, "valid_mouth")
        gaze_valid = bval(row, "valid_gaze")
        pose_valid = bval(row, "valid_pose")

        closure = fval(row, "closure_s")
        perclos = fval(row, "perclos")
        gaze_away = fval(row, "gaze_away_s")

        # Android feeds NaN PERCLOS only before eye_valid becomes true.
        safe_perclos = perclos if math.isfinite(perclos) else 0.0
        safe_closure = closure if math.isfinite(closure) else 0.0
        safe_gaze = gaze_away if math.isfinite(gaze_away) else 0.0

        metrics = engine.update(
            timestamp_ms=ival(row, "t_ms"),
            closure_s=safe_closure,
            perclos=safe_perclos,
            gaze_away_s=safe_gaze,
            yawn_event=bval(row, "yawn_event"),
            nod_event=bval(row, "nod_event"),
            face_valid=face_valid,
            eye_valid=eye_valid,
            mouth_valid=mouth_valid,
            gaze_valid=gaze_valid,
            pose_valid=pose_valid,
        )

        capture_score = fval(row, "score")
        if math.isfinite(capture_score):
            score_diffs.append(
                abs(float(metrics["score"]) - capture_score)
            )

        state_mismatches += (
            str(metrics["state"]) != row["state"]
        )
        event_mismatches += (
            str(metrics["state_event"]) != row["state_event"]
        )

    return (
        max(score_diffs) if score_diffs else float("nan"),
        state_mismatches,
        event_mismatches,
    )


def print_reconstruction_diagnostics(
    capture_rows: list[dict[str, str]],
    e_rows: list[dict[str, object]],
) -> None:
    gaps = [
        ival(b, "t_ms") - ival(a, "t_ms")
        for a, b in zip(capture_rows, capture_rows[1:])
    ]

    print()
    print("Diagnóstico temporal E (reconstruido vs Android capturado):")
    print(
        "max_abs_closure_diff =",
        f"{_max_abs_pair_diff(e_rows, 'closure_s', capture_rows, 'closure_s'):.8f}",
    )
    print(
        "max_abs_perclos_diff =",
        f"{_max_abs_pair_diff(e_rows, 'perclos', capture_rows, 'perclos'):.8f}",
    )
    print(
        "yawn_event_mismatches =",
        _event_mismatch_count(
            e_rows, "yawn_event", capture_rows, "yawn_event"
        ),
    )
    print(
        "max_abs_gaze_away_diff =",
        f"{_max_abs_pair_diff(e_rows, 'gaze_away_s', capture_rows, 'gaze_away_s'):.8f}",
    )
    print(
        "nod_event_mismatches =",
        _event_mismatch_count(
            e_rows, "nod_event", capture_rows, "nod_event"
        ),
    )
    print("max_logged_gap_ms =", max(gaps) if gaps else 0)
    print("gaps_gt_250ms =", sum(g > 250 for g in gaps))

    score_diff, state_mm, event_mm = (
        validate_engine_from_captured_temporals(capture_rows)
    )
    print()
    print("Control del motor E usando temporales ya calculados por Android:")
    print(
        "max_abs_score_diff_captured_temporals =",
        f"{score_diff:.8f}",
    )
    print(
        "state_mismatches_captured_temporals =",
        state_mm,
    )
    print(
        "event_mismatches_captured_temporals =",
        event_mm,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay H8 A-E sobre exactamente los mismos frames capturados."
    )
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.features)
    by_mode = {
        mode: replay_mode_from_captured_temporals(rows, mode)
        for mode in MODES
    }
    write_output(args.output, by_mode)

    print("run =", rows[0]["feature_run_id"])
    print("source =", rows[0]["source"])
    print("frames por modo =", len(rows))
    print("output =", args.output)
    print_summary(by_mode)

    print()
    print("Validación principal E contra ejecución Android capturada:")
    e_rows = by_mode["E"]
    score_diffs = [
        abs(float(r["score"]) - float(r["capture_score"]))
        for r in e_rows
        if math.isfinite(float(r["capture_score"]))
    ]
    print(
        "max_abs_score_diff =",
        f"{max(score_diffs) if score_diffs else float('nan'):.8f}",
    )
    print(
        "state_mismatches =",
        sum(str(r["state"]) != str(r["capture_state"]) for r in e_rows),
    )
    print(
        "event_mismatches =",
        sum(
            str(r["state_event"]) != str(r["capture_state_event"])
            for r in e_rows
        ),
    )

    # Secondary audit only: shows why raw reconstruction is not exact when
    # no-face/onEmpty frames occurred between logged result rows.
    raw_e = replay_mode(rows, "E")
    print_reconstruction_diagnostics(rows, raw_e)


if __name__ == "__main__":
    main()
