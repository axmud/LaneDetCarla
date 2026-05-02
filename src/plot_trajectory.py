"""
Plot recorded reference path vs. one or more actual ego trajectories.

Usage:
    # Single log (default file)
    python src/plot_trajectory.py

    # Single log, explicit
    python src/plot_trajectory.py --logs src/ego_trajectory.csv

    # Compare multiple PID tunings on one figure
    python src/plot_trajectory.py --logs src/run_a.csv src/run_b.csv src/run_c.csv

    # Custom labels (one per log) and save to file
    python src/plot_trajectory.py \\
        --logs src/run_a.csv src/run_b.csv \\
        --labels "Kp=0.25" "Kp=0.50" \\
        --save plots/comparison.png

    # Back-compat: --log still works (singular)
    python src/plot_trajectory.py --log src/run1.csv
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Distinct colors for overlaying multiple runs (matplotlib tab10 minus gray/orange,
# which we reserve for the reference path and the error fill respectively).
PALETTE = [
    "tab:blue",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
]


def default_label(path):
    """Derive a readable label from a CSV path."""
    return os.path.splitext(os.path.basename(path))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", default=None,
                        help="One or more trajectory CSVs to overlay")
    parser.add_argument("--log", default=None,
                        help="(Back-compat) single log CSV. Equivalent to --logs <file>.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Custom labels, one per log (defaults to filename)")
    parser.add_argument("--ref", default="src/carla_data.csv",
                        help="Reference path CSV (waypoint_x, waypoint_y, ...)")
    parser.add_argument("--save", default=None,
                        help="If set, save figure to this path instead of showing")
    args = parser.parse_args()

    # Resolve which logs to plot (precedence: --logs, then --log, then default)
    if args.logs:
        log_paths = args.logs
    elif args.log:
        log_paths = [args.log]
    else:
        log_paths = ["src/ego_trajectory_in_pid_lane_shift.csv"]

    # Resolve labels
    if args.labels:
        if len(args.labels) != len(log_paths):
            parser.error(
                f"--labels has {len(args.labels)} entries but --logs has {len(log_paths)}"
            )
        labels = args.labels
    else:
        labels = [default_label(p) for p in log_paths]

    # Load all logs up front so we can fail fast on bad paths
    logs = [pd.read_csv(p) for p in log_paths]
    ref = pd.read_csv(args.ref, index_col=0)

    fig, (ax_xy, ax_err) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}
    )

    # =========================================================================
    # XY plot: reference + every logged trajectory
    # =========================================================================
    ax_xy.plot(ref["waypoint_x"], ref["waypoint_y"],
               "--", color="tab:gray", linewidth=1.5,
               label="Reference path", zorder=1)

    for i, (log, label) in enumerate(zip(logs, labels)):
        color = PALETTE[i % len(PALETTE)]
        ax_xy.plot(log["x"], log["y"],
                   "-", color=color, linewidth=1.8,
                   label=label, zorder=2)

        # Start / end markers per run, color-matched but with distinct shapes
        ax_xy.scatter(log["x"].iloc[0], log["y"].iloc[0],
                      s=70, color=color, marker="o",
                      edgecolors="black", linewidths=0.8, zorder=3)
        ax_xy.scatter(log["x"].iloc[-1], log["y"].iloc[-1],
                      s=80, color=color, marker="X",
                      edgecolors="black", linewidths=0.8, zorder=3)

    # Single legend entry for "Start" and "End" markers (shape only, neutral color)
    ax_xy.scatter([], [], s=70, color="white", marker="o",
                  edgecolors="black", linewidths=0.8, label="Start")
    ax_xy.scatter([], [], s=80, color="white", marker="X",
                  edgecolors="black", linewidths=0.8, label="End")

    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_title("Trajectory (top-down)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best", fontsize=9)

    # =========================================================================
    # Cross-track error vs. time
    # =========================================================================
    ax_err.axhline(0, color="k", linewidth=0.8, alpha=0.5)

    # Single-log behaviour: keep the filled area for visual emphasis.
    # Multi-log: skip the fill (it would just become muddy overlap) and
    # rely on colored lines + a stats table.
    fill = len(logs) == 1

    stats = []
    for i, (log, label) in enumerate(zip(logs, labels)):
        color = PALETTE[i % len(PALETTE)]
        t = log["time_s"] - log["time_s"].iloc[0]
        rmse = float(np.sqrt(np.mean(log["lane_shift"] ** 2)))
        max_abs = float(log["lane_shift"].abs().max())
        stats.append((label, rmse, max_abs))

        legend_label = f"{label}  (RMSE={rmse:.3f}, max|e|={max_abs:.3f})"
        ax_err.plot(t, log["lane_shift"],
                    color=color, linewidth=1.2,
                    label=legend_label)

        if fill:
            ax_err.fill_between(t, log["lane_shift"], 0,
                                alpha=0.2, color=color)

    if len(logs) == 1:
        ax_err.set_title(
            f"Cross-track error\nRMSE={stats[0][1]:.3f} m | max|e|={stats[0][2]:.3f} m"
        )
    else:
        ax_err.set_title("Cross-track error (multi-run comparison)")

    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Lane shift [m]")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best", fontsize=8)

    # Print a concise comparison table to stdout — handy when sweeping gains
    if len(logs) > 1:
        print("\nCross-track error summary")
        print(f"  {'label':<30s}  {'RMSE [m]':>10s}  {'max|e| [m]':>12s}")
        print(f"  {'-'*30}  {'-'*10}  {'-'*12}")
        for label, rmse, max_abs in stats:
            print(f"  {label:<30s}  {rmse:>10.4f}  {max_abs:>12.4f}")
        print()

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()