# Let's write a robust, self-contained script "tendance.py" that:
# - reads a CSV with a time-series column
# - injects a chosen anomaly into that series
# - writes back the CSV along with metadata and optional plot
#
# We keep only standard stack: pandas, numpy, matplotlib
# and pack every behavior behind CLI flags so it's portable.
#
# Python 3.9+

"""
tendance.py — inject trend ("tendance") anomalies into a time series CSV.

Usage (examples):
  python executable/test/tendance.py \
      --input data/Golden/chunking/output_of_the_1th_chunk.csv \
      --ts-col "Temperatur Exzenterlager links" \
      --anomaly spike \
      --start 200 \
      --magnitude 50 \
      --output-dir data/Anomaly \
      --save-plot \
      --seed 42

  python executable/test/tendance.py \
      --input data/Golden/chunking/output_of_the_1th_chunk.csv \
      --ts-col "Temperatur Exzenterlager links" \
      --anomaly level_shift \
      --start 150 \
      --length 100 \
      --magnitude 30 \
      --output-dir data/Anomaly \
      --save-plot \
      --seed 42

  # Example for deviation (affine transform y <- a*y + b from start to the end):
  python executable/test/tendance.py \
      --input data/Golden/chunking/output_of_the_1th_chunk.csv \
      --ts-col "Temperatur Exzenterlager links" \
      --anomaly deviation \
      --start 250 \
      --a 1.02 \
      --b 15 \
      --output-dir data/Anomaly \
      --save-plot \
      --seed 42


Anomalies supported:
  - spike: single-point additive spike at --start (ignores --length)
  - level_shift: constant offset from --start to the end (ignores --length)
  - variance_burst: increase variance over a window
  - trend_change: change the linear trend over a window
  - drift: add a gradual linear drift over a window
  - missing_block: mark a window as missing (NaN)
  - amplitude_change: multiply values by a factor over a window
  - deviation: apply y = a*y + b from --start to the end

Notes:
- We keep CSV I/O simple: we rewrite a new CSV with the modified series, add a
  boolean "anomaly_flag" column, and dump a JSON sidecar with the parameters.
- If --save-plot is set, we render an overlay plot of original vs modified series.
- We do not rely on time index; the series is treated as ordered samples.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utility
# -----------------------------

def seed_everything(seed: Optional[int] = None) -> None:
    if seed is None:
        return
    np.random.seed(seed)


def ensure_output_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def window_from_start_length(start: int, length: Optional[int], n: int) -> slice:
    """
    Produce a slice window [start: start+length] or [start:n] if length is None or <=0.
    Clamps within [0, n].
    """
    start = max(0, int(start))
    if length is None or length <= 0:
        stop = n
    else:
        stop = min(n, start + int(length))
    return slice(start, stop)


# -----------------------------
# Anomaly injectors
# -----------------------------

def inject_variance_burst(y: pd.Series, window: slice, factor: float = 3.0) -> pd.Series:
    """
    Increase variance by multiplying the local standard deviation and adding noise.
    """
    y2 = y.copy()
    segment = y2.iloc[window].astype(float).values
    if len(segment) == 0:
        return y2
    mu = np.mean(segment)
    sigma = np.std(segment) if np.std(segment) > 0 else 1.0
    noisy = mu + (segment - mu) * factor + np.random.normal(0, sigma * (factor - 1), size=len(segment))
    y2.iloc[window] = noisy
    return y2


def inject_trend_change(y: pd.Series, window: slice, delta_per_step: float = 0.1) -> pd.Series:
    """
    Add a linear trend change across the window: y[i] += i*delta_per_step, relative to window start.
    """
    y2 = y.copy()
    n = len(y2.iloc[window])
    if n <= 0:
        return y2
    inc = np.arange(n) * delta_per_step
    y2.iloc[window] = y2.iloc[window].astype(float).values + inc
    return y2


def inject_drift(y: pd.Series, window: slice, magnitude: float = 50.0) -> pd.Series:
    """
    Add a drift that ramps from 0 at window start to 'magnitude' at window end.
    """
    y2 = y.copy()
    n = len(y2.iloc[window])
    if n <= 0:
        return y2
    ramp = np.linspace(0.0, magnitude, n)
    y2.iloc[window] = y2.iloc[window].astype(float).values + ramp
    return y2


def inject_missing_block(y: pd.Series, window: slice) -> pd.Series:
    """
    Replace the window with NaNs.
    """
    y2 = y.copy()
    y2.iloc[window] = np.nan
    return y2


def inject_amplitude_change(y: pd.Series, window: slice, factor: float = 1.5) -> pd.Series:
    """
    Multiply the amplitude in a window by a factor (center around the window's mean).
    """
    y2 = y.copy()
    segment = y2.iloc[window].astype(float).values
    if len(segment) == 0:
        return y2
    mu = np.mean(segment)
    y2.iloc[window] = mu + (segment - mu) * factor
    return y2


def inject_spike(y: pd.Series, start_pos: int, magnitude: float) -> pd.Series:
    """Single-point additive spike: y[start] += magnitude (sign of magnitude controls up/down)."""
    y2 = y.copy()
    if start_pos < 0 or start_pos >= len(y2):
        return y2
    y2.iloc[start_pos] = y2.iloc[start_pos] + magnitude
    return y2


def inject_level_shift(y: pd.Series, start_pos: int, shift: float) -> pd.Series:
    """Tail level shift: add 'shift' from start_pos to the end (ignores --length)."""
    y2 = y.copy()
    y2.iloc[start_pos:] = y2.iloc[start_pos:].astype(float).values + shift
    return y2


def inject_deviation(y: pd.Series, start_pos: int, a: float, b: float) -> pd.Series:
    """Apply y = a*y + b from start_pos to the end."""
    y2 = y.copy()
    tail = y2.iloc[start_pos:].astype(float).values
    y2.iloc[start_pos:] = a * tail + b
    return y2


# -----------------------------
# Plotting
# -----------------------------

def plot_overlay(
    x: np.ndarray,
    y: np.ndarray,
    y2: np.ndarray,
    title: str,
    out_path: Optional[Path] = None
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="original", linewidth=2)
    plt.plot(x, y2, label="with anomaly", linestyle="--", linewidth=2)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=120)
    plt.close()


# -----------------------------
# CLI / Main
# -----------------------------

@dataclass
class Args:
    input: str
    ts_col: str
    anomaly: str
    start: int
    length: Optional[int]
    magnitude: float
    factor: float
    mode: str
    output_dir: str
    save_plot: bool
    seed: Optional[int]
    a: float
    b: float


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inject anomalies into a time series column of a CSV.")
    p.add_argument("--input", required=True, help="Path to input CSV.")
    p.add_argument("--ts-col", required=True, help="Name of the time-series column to modify.")
    p.add_argument(
        "--anomaly",
        required=True,
        choices=["spike", "level_shift", "variance_burst", "trend_change", "drift", "missing_block", "amplitude_change", "deviation"],
        help="Type of anomaly to inject."
    )
    p.add_argument("--start", type=int, required=True, help="Start index (0-based) for the anomaly or transform.")
    p.add_argument("--length", type=int, default=0, help="Window length (used by windowed anomalies). <=0 means to the end.")
    p.add_argument("--magnitude", type=float, default=50.0, help="Magnitude for spike/level_shift/drift (additive, default: 50.0)")
    p.add_argument("--a", type=float, default=1.0, help="Scale for deviation: y = a*y + b (default 1.0)")
    p.add_argument("--b", type=float, default=0.0, help="Offset for deviation: y = a*y + b (default 0.0)")
    p.add_argument("--factor", type=float, default=1.5, help="Factor for amplitude_change or variance_burst (default: 1.5)")
    p.add_argument("--mode", choices=["add"], default="add", help="Deprecated: spike is now single-point additive; --mode is ignored.")
    p.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    p.add_argument("--save-plot", action="store_true", help="If set, save an overlay plot.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return p


def main() -> None:
    parser = build_argparser()
    args_ns = parser.parse_args()
    args = Args(
        input=args_ns.input,
        ts_col=args_ns.ts_col,
        anomaly=args_ns.anomaly,
        start=args_ns.start,
        length=args_ns.length,
        magnitude=args_ns.magnitude,
        factor=args_ns.factor,
        mode=args_ns.mode,
        output_dir=args_ns.output_dir,
        save_plot=args_ns.save_plot,
        seed=args_ns.seed,
        a=args_ns.a,
        b=args_ns.b,
    )

    seed_everything(args.seed)
    out_dir = ensure_output_dir(args.output_dir)

    # Read CSV
    df = pd.read_csv(args.input)
    if args.ts_col not in df.columns:
        raise SystemExit(f"Column '{args.ts_col}' not found in {args.input}. Available: {list(df.columns)}")

    y = df[args.ts_col].copy()
    n = len(y)
    pos = max(0, min(n - 1, int(args.start)))
    window = window_from_start_length(args.start, args.length, n)

    # Inject
    if args.anomaly == "spike":
        y_new = inject_spike(y, pos, magnitude=args.magnitude)
    elif args.anomaly == "level_shift":
        y_new = inject_level_shift(y, pos, shift=args.magnitude)
    elif args.anomaly == "variance_burst":
        y_new = inject_variance_burst(y, window, factor=args.factor)
    elif args.anomaly == "trend_change":
        y_new = inject_trend_change(y, window, delta_per_step=args.magnitude / max(1, len(y.iloc[window])))
    elif args.anomaly == "drift":
        y_new = inject_drift(y, window, magnitude=args.magnitude)
    elif args.anomaly == "missing_block":
        y_new = inject_missing_block(y, window)
    elif args.anomaly == "amplitude_change":
        y_new = inject_amplitude_change(y, window, factor=args.factor)
    elif args.anomaly == "deviation":
        y_new = inject_deviation(y, pos, a=args.a, b=args.b)
    else:
        raise SystemExit(f"Unsupported anomaly: {args.anomaly}")

    # Flag vector
    flag = pd.Series(0, index=y.index, name="anomaly_flag", dtype=int)
    flag.iloc[window] = 1
    if args.anomaly == 'spike':
        flag[:] = 0
        if 0 <= pos < len(flag):
            flag.iloc[pos] = 1
    elif args.anomaly in ('level_shift','deviation'):
        flag[:] = 0
        flag.iloc[pos:] = 1

    # Prepare output names
    base_in = Path(args.input).stem
    safe_col = args.ts_col.replace(" ", "_").replace("/", "_")
    out_csv = out_dir / f"{base_in}__{safe_col}__{args.anomaly}.csv"
    out_json = out_dir / f"{base_in}__{safe_col}__{args.anomaly}.json"
    out_png = out_dir / f"{base_in}__{safe_col}__{args.anomaly}.png" if args.save_plot else None

    # Write CSV
    df_out = df.copy()
    df_out[args.ts_col] = y_new
    # Removed anomaly_flag column to prevent data leakage
    df_out.to_csv(out_csv, index=False)

    # Write metadata JSON
    meta = {
        "input": str(Path(args.input).resolve()),
        "output_csv": str(out_csv.resolve()),
        "ts_col": args.ts_col,
        "anomaly": args.anomaly,
        "start": int(args.start),
        "length": int(args.length),
        "magnitude": float(args.magnitude),
        "factor": float(args.factor),
        "a": args.a,
        "b": args.b,
        "mode": args.mode,
        "seed": args.seed,
        "rows": int(n),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Plot if asked
    if args.save_plot:
        x = np.arange(n)
        title = f"{args.ts_col} — {args.anomaly} [start={pos}]"
        plot_overlay(x, y.values, y_new.values, title, out_png)

    print(f"Wrote: {out_csv}")
    print(f"Meta:  {out_json}")
    if out_png is not None:
        print(f"Plot:  {out_png}")


if __name__ == "__main__":
    main()
