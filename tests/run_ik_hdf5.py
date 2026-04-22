"""Comprehensive IK validation on a real Astribot HDF5 trajectory.

Reports five categories of metrics:

1. End-effector position error (mm)
2. End-effector orientation error (deg)
3. Solver success rate (convergence flag + tolerance checks)
4. Constraint satisfaction rate (joint position limits)
5. Solver efficiency (iterations, per-frame time, throughput)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise ImportError("h5py is required. Install with `pip install h5py`.") from exc


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from astribot_kinematics import (  # noqa: E402
    AstribotFK,
    AstribotIK,
    CHAIN_ARM_LEFT,
    CHAIN_ARM_RIGHT,
    CHAIN_TORSO,
)


DEFAULT_HDF5 = "/home/yuqi/astribot_kinematics/tests/data/0710_Microwave_S8_episode_0.hdf5"

TORSO_SLICE = slice(3, 7)
ARM_LEFT_SLICE = slice(7, 14)
ARM_RIGHT_SLICE = slice(15, 22)

ARM_KEYS = (CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT)
EEF_KEYS = (CHAIN_TORSO, CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryData:
    q_user: np.ndarray
    gt: Dict[str, np.ndarray]


def _load_trajectory(path: str) -> TrajectoryData:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "r") as f:
        joints_cmd = f["joints_dict/joints_position_command"][()]
        gt = {key: f[f"command_poses_dict/{key}"][()] for key in EEF_KEYS}
    q_user = np.concatenate(
        [joints_cmd[:, TORSO_SLICE], joints_cmd[:, ARM_LEFT_SLICE], joints_cmd[:, ARM_RIGHT_SLICE]],
        axis=1,
    )
    return TrajectoryData(q_user=q_user, gt=gt)


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------
def _position_error_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3] - b[:3]) * 1000.0)


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    qa = a[3:] / max(np.linalg.norm(a[3:]), 1e-12)
    qb = b[3:] / max(np.linalg.norm(b[3:]), 1e-12)
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _summary(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {k: float("nan") for k in ("mean", "median", "rmse", "max", "p95", "p99", "std")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "max": float(values.max()),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "std": float(values.std()),
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    bar = "=" * 82
    print(f"\n{bar}\n {title}\n{bar}")


def _sub(title: str) -> None:
    print(f"\n-- {title} --")


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    cols = list(zip(*([headers] + list(rows))))
    widths = [max(len(str(c)) for c in col) for col in cols]

    def fmt(row: Sequence[str]) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(row, widths))

    sep = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))


def _fmt_stats_row(label: str, stats: Dict[str, float], count: int, unit: str) -> List[str]:
    return [
        label,
        str(count),
        f"{stats['mean']:.4f} {unit}",
        f"{stats['median']:.4f} {unit}",
        f"{stats['rmse']:.4f} {unit}",
        f"{stats['p95']:.4f} {unit}",
        f"{stats['p99']:.4f} {unit}",
        f"{stats['max']:.4f} {unit}",
        f"{stats['std']:.4f} {unit}",
    ]


STATS_HEADERS = ("frame", "N", "mean", "median", "rmse", "p95", "p99", "max", "std")


# ---------------------------------------------------------------------------
# Trajectory replay
# ---------------------------------------------------------------------------
@dataclass
class IKRunResult:
    pos_err_mm: Dict[str, np.ndarray]
    rot_err_deg: Dict[str, np.ndarray]
    converged: np.ndarray
    iterations: np.ndarray
    per_frame_time_s: np.ndarray
    q_solutions: np.ndarray


def replay_trajectory(
    fk: AstribotFK,
    ik_torso: AstribotIK,
    ik_arms: AstribotIK,
    data: TrajectoryData,
    frame_indices: np.ndarray,
    max_iters: int,
    tol_pos: float,
    tol_rot: float,
    damping: float,
) -> IKRunResult:
    """Two-step solve per frame: first torso, then both arms on top of the torso
    solution (used as warm start, with torso joints held fixed)."""
    n = frame_indices.size
    pos_err_mm = {k: np.empty(n, dtype=np.float64) for k in EEF_KEYS}
    rot_err_deg = {k: np.empty(n, dtype=np.float64) for k in EEF_KEYS}
    converged = np.zeros(n, dtype=bool)
    iterations = np.zeros(n, dtype=np.int64)
    per_frame_time = np.zeros(n, dtype=np.float64)
    q_solutions = np.empty((n, fk.nq), dtype=np.float64)

    q_init = data.q_user[frame_indices[0]].copy()
    for i, idx in enumerate(frame_indices):
        target_torso = data.gt[CHAIN_TORSO][idx]
        target_left = data.gt[CHAIN_ARM_LEFT][idx]
        target_right = data.gt[CHAIN_ARM_RIGHT][idx]

        t0 = time.perf_counter()
        q1, ok1, info1 = ik_torso.solve(
            {CHAIN_TORSO: target_torso},
            q_init=q_init,
            max_iters=max_iters, tol_pos=tol_pos, tol_rot=tol_rot, damping=damping,
        )
        q_sol, ok2, info2 = ik_arms.solve(
            {CHAIN_ARM_LEFT: target_left, CHAIN_ARM_RIGHT: target_right},
            q_init=q1,
            max_iters=max_iters, tol_pos=tol_pos, tol_rot=tol_rot, damping=damping,
        )
        per_frame_time[i] = time.perf_counter() - t0

        converged[i] = bool(ok1 and ok2)
        iterations[i] = int(info1.get("iterations", 0)) + int(info2.get("iterations", 0))
        q_solutions[i] = q_sol

        pred = fk.forward(q_sol, links=list(EEF_KEYS))
        for key, tgt in ((CHAIN_TORSO, target_torso),
                         (CHAIN_ARM_LEFT, target_left),
                         (CHAIN_ARM_RIGHT, target_right)):
            pos_err_mm[key][i] = _position_error_mm(pred[key], tgt)
            rot_err_deg[key][i] = _quat_angle_deg(pred[key], tgt)

        q_init = q_sol

    return IKRunResult(
        pos_err_mm=pos_err_mm,
        rot_err_deg=rot_err_deg,
        converged=converged,
        iterations=iterations,
        per_frame_time_s=per_frame_time,
        q_solutions=q_solutions,
    )


# ---------------------------------------------------------------------------
# Metric blocks
# ---------------------------------------------------------------------------
def report_pose_errors(result: IKRunResult) -> None:
    _section("1. End-effector position error (mm)")
    rows = [
        _fmt_stats_row(key, _summary(result.pos_err_mm[key]), result.pos_err_mm[key].size, "mm")
        for key in EEF_KEYS
    ]
    _print_table(STATS_HEADERS, rows)

    _section("2. End-effector orientation error (deg)")
    rows = [
        _fmt_stats_row(key, _summary(result.rot_err_deg[key]), result.rot_err_deg[key].size, "deg")
        for key in EEF_KEYS
    ]
    _print_table(STATS_HEADERS, rows)


def report_success_rate(result: IKRunResult, tol_pos_mm: float, tol_rot_deg: float) -> None:
    _section("3. Solver success rate")

    n = result.converged.size
    solver_flag_rate = float(result.converged.mean())

    # Independent tolerance checks per frame, across all three end-effectors.
    pos_ok = np.stack(
        [result.pos_err_mm[k] < tol_pos_mm for k in EEF_KEYS], axis=0
    ).all(axis=0)
    rot_ok = np.stack(
        [result.rot_err_deg[k] < tol_rot_deg for k in EEF_KEYS], axis=0
    ).all(axis=0)
    pose_ok = pos_ok & rot_ok

    rows = [
        ["two-step solver converged (step1 & step2)", f"{int(result.converged.sum())}/{n}", f"{solver_flag_rate * 100:.2f}%"],
        [f"pos_err < {tol_pos_mm} mm (torso+arms)", f"{int(pos_ok.sum())}/{n}", f"{pos_ok.mean() * 100:.2f}%"],
        [f"rot_err < {tol_rot_deg} deg (torso+arms)", f"{int(rot_ok.sum())}/{n}", f"{rot_ok.mean() * 100:.2f}%"],
        ["pose_err ok (all 3 frames)", f"{int(pose_ok.sum())}/{n}", f"{pose_ok.mean() * 100:.2f}%"],
    ]
    _print_table(("criterion", "count", "rate"), rows)


def report_constraints(fk: AstribotFK, ik: AstribotIK, result: IKRunResult) -> None:
    _section("4. Constraint satisfaction (joint position limits)")

    lower, upper = fk.joint_limits()
    q = result.q_solutions  # (N, 18)
    active_mask = np.zeros(fk.nq, dtype=bool)
    for chain in ik.active_chains:
        for jn in fk.joint_names:
            if jn in set(_chain_joints(chain)):
                active_mask[fk.joint_names.index(jn)] = True

    lo_violation = np.maximum(lower - q, 0.0)
    hi_violation = np.maximum(q - upper, 0.0)
    violation = np.maximum(lo_violation, hi_violation)

    per_frame_ok = (violation <= 1e-9).all(axis=1)
    per_joint_max = violation.max(axis=0)

    rows = [
        ["all 18 joints within limits (per frame)", f"{int(per_frame_ok.sum())}/{q.shape[0]}", f"{per_frame_ok.mean() * 100:.2f}%"],
        ["active-chain joints within limits", f"{int(((violation[:, active_mask] <= 1e-9).all(axis=1)).sum())}/{q.shape[0]}", "-"],
        ["global max violation", f"{per_joint_max.max():.3e} rad", "-"],
    ]
    _print_table(("criterion", "count / value", "rate"), rows)

    _sub("per-joint margin to limits (rad, over all frames)")
    margin_lo = q - lower
    margin_hi = upper - q
    margin = np.minimum(margin_lo, margin_hi)
    rows = []
    for i, jname in enumerate(fk.joint_names):
        if not active_mask[i]:
            continue
        rows.append(
            [
                jname,
                f"[{lower[i]:+.4f}, {upper[i]:+.4f}]",
                f"{margin[:, i].min():.4f}",
                f"{margin[:, i].mean():.4f}",
                "ok" if per_joint_max[i] <= 1e-9 else f"violated (max {per_joint_max[i]:.3e})",
            ]
        )
    _print_table(
        ("joint", "limits (rad)", "min margin", "mean margin", "status"),
        rows,
    )


def _chain_joints(chain: str) -> List[str]:
    from astribot_kinematics.constants import CHAIN_JOINTS
    return CHAIN_JOINTS[chain]


def report_efficiency(result: IKRunResult) -> None:
    _section("5. Solver efficiency")

    ms = result.per_frame_time_s * 1000.0
    iters = result.iterations.astype(np.float64)
    total_s = result.per_frame_time_s.sum()

    rows = [
        [
            "per-frame time (ms)",
            f"{ms.size}",
            f"{ms.mean():.3f}",
            f"{np.median(ms):.3f}",
            f"{np.percentile(ms, 95):.3f}",
            f"{np.percentile(ms, 99):.3f}",
            f"{ms.max():.3f}",
            f"{ms.std():.3f}",
        ],
        [
            "iterations",
            f"{iters.size}",
            f"{iters.mean():.2f}",
            f"{np.median(iters):.2f}",
            f"{np.percentile(iters, 95):.2f}",
            f"{np.percentile(iters, 99):.2f}",
            f"{iters.max():.0f}",
            f"{iters.std():.2f}",
        ],
    ]
    _print_table(
        ("metric", "N", "mean", "median", "p95", "p99", "max", "std"),
        rows,
    )

    print(
        f"\n total time          : {total_s:.3f} s"
        f"\n throughput          : {result.per_frame_time_s.size / total_s:,.1f} frames / s"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive IK validation on an HDF5 trajectory.")
    parser.add_argument("--hdf5", default=DEFAULT_HDF5, help="Path to trajectory HDF5 file.")
    parser.add_argument("--step", type=int, default=1, help="Replay every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit number of replay frames (0 = no limit).")
    parser.add_argument("--max-iters", type=int, default=100, help="IK max iterations per frame.")
    parser.add_argument("--tol-pos", type=float, default=5e-3, help="Position tolerance (m) passed to the solver.")
    parser.add_argument("--tol-rot", type=float, default=1e-2, help="Rotation tolerance (rad) passed to the solver.")
    parser.add_argument("--damping", type=float, default=1e-4, help="DLS damping factor.")
    parser.add_argument("--pass-pos-mm", type=float, default=5.0, help="Pass threshold for position error (mm).")
    parser.add_argument("--pass-rot-deg", type=float, default=1.0, help="Pass threshold for rotation error (deg).")
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("--step must be > 0.")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")

    data = _load_trajectory(args.hdf5)
    frame_indices = np.arange(0, data.q_user.shape[0], args.step, dtype=np.int64)
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]

    fk = AstribotFK()
    ik_torso = AstribotIK(fk=fk, chains=[CHAIN_TORSO])
    ik_arms = AstribotIK(fk=fk, chains=[CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT])

    _section("Astribot IK validation report (two-step: torso -> arms)")
    print(f" HDF5 file          : {args.hdf5}")
    print(f" Total frames       : {data.q_user.shape[0]}")
    print(f" Replay frames      : {frame_indices.size} (step={args.step})")
    print(f" Step1 chains       : {ik_torso.active_chains}")
    print(f" Step2 chains       : {ik_arms.active_chains}")
    print(f" Solver params      : max_iters={args.max_iters}, tol_pos={args.tol_pos} m, "
          f"tol_rot={args.tol_rot} rad, damping={args.damping}")
    print(f" Pass thresholds    : pos<{args.pass_pos_mm} mm, rot<{args.pass_rot_deg} deg")

    result = replay_trajectory(
        fk=fk,
        ik_torso=ik_torso,
        ik_arms=ik_arms,
        data=data,
        frame_indices=frame_indices,
        max_iters=args.max_iters,
        tol_pos=args.tol_pos,
        tol_rot=args.tol_rot,
        damping=args.damping,
    )

    report_pose_errors(result)
    report_success_rate(result, tol_pos_mm=args.pass_pos_mm, tol_rot_deg=args.pass_rot_deg)
    # Use a 3-chain instance only for the constraint report so that torso
    # joints (moved in step 1) are counted as active.
    ik_all = AstribotIK(fk=fk, chains=[CHAIN_TORSO, CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT])
    report_constraints(fk, ik_all, result)
    report_efficiency(result)

    print("\n[Done]")


if __name__ == "__main__":
    main()
