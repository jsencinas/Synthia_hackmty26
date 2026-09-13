"""Detector endpoint with calls from the dataset.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TIMEOUT_S = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Full POST /detect URL")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--n", type=int, default=20, help="Number of calls; 0 means all")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.csv")
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "audio")
    args = parser.parse_args()
    if args.n < 0:
        parser.error("--n must be zero or positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def load_rows(path: Path, split: str, limit: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
    if not rows:
        raise ValueError(f"No rows found for split={split!r} in {path}")
    return rows if limit == 0 else rows[:limit]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_response(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "response is not a JSON object"
    if not isinstance(payload.get("is_synthetic"), bool):
        return False, "is_synthetic is missing or is not boolean"
    if "confidence" in payload:
        confidence = payload["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            return False, "confidence is not a number between 0 and 1"
    return True, None


def send_call(
    url: str,
    row: dict[str, str],
    audio_dir: Path,
    timeout_s: float,
) -> tuple[bool | None, float, str | None]:
    call_id = row["anon_id"]
    wav_path = audio_dir / f"{call_id}.wav"
    request_body = {
        "call_id": call_id,
        "audio_base64": base64.b64encode(wav_path.read_bytes()).decode("ascii"),
        "sample_rate": 8000,
        "channels": 2,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = response.status
            raw_body = response.read()
    except (OSError, urllib.error.URLError) as exc:
        return None, time.perf_counter() - started, f"request failed: {exc}"

    elapsed = time.perf_counter() - started
    if status != 200:
        return None, elapsed, f"HTTP {status}"
    if elapsed > DEFAULT_TIMEOUT_S:
        return None, elapsed, f"exceeded judge limit ({elapsed:.2f}s)"
    try:
        payload = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, elapsed, f"invalid JSON response: {exc}"
    valid, error = validate_response(payload)
    if not valid:
        return None, elapsed, error
    return payload["is_synthetic"], elapsed, None


def main() -> int:
    args = parse_args()
    rows = load_rows(args.manifest, args.split, args.n)
    correct_by_label = {"human": 0, "synthetic": 0}
    total_by_label = {"human": 0, "synthetic": 0}
    latencies: list[float] = []
    errors: list[str] = []

    for index, row in enumerate(rows, start=1):
        label = row["label"]
        total_by_label[label] += 1
        try:
            prediction, elapsed, error = send_call(
                args.url, row, args.audio_dir, args.timeout
            )
        except (OSError, KeyError, ValueError) as exc:
            prediction, elapsed, error = None, 0.0, f"local data error: {exc}"
        if elapsed:
            latencies.append(elapsed)
        expected = label == "synthetic"
        is_correct = error is None and prediction == expected
        if is_correct:
            correct_by_label[label] += 1
        status = "OK" if is_correct else "WRONG"
        detail = f" ({error})" if error else ""
        print(f"[{index:>3}/{len(rows)}] {row['anon_id']} {status} {elapsed:.3f}s{detail}")
        if error:
            errors.append(f"{row['anon_id']}: {error}")

    recalls = [
        correct_by_label[label] / total
        for label, total in total_by_label.items()
        if total > 0
    ]
    balanced_accuracy = sum(recalls) / len(recalls)
    print(f"\nBalanced accuracy: {balanced_accuracy:.4f}")
    if latencies:
        print(
            "Latency: "
            f"mean={sum(latencies) / len(latencies):.3f}s "
            f"p95={percentile(latencies, 0.95):.3f}s "
            f"max={max(latencies):.3f}s"
        )
    else:
        print("Latency: no requests completed")
    print(f"Protocol/request errors: {len(errors)}")
    for error in errors:
        print(f"  - {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
