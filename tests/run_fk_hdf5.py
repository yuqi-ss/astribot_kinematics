"""Comprehensive FK validation on a real Astribot HDF5 trajectory.

Reports five categories of metrics:

1. Position accuracy (mm) per end-effector
2. Orientation accuracy (deg) per end-effector
3. Frame / tool-frame consistency
4. Jacobian correctness (analytical vs central finite difference)
5. Computational efficiency (latency + throughput)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

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
    CHAIN_ARM_LEFT,
    CHAIN_ARM_RIGHT,
    CHAIN_TORSO,
)


DEFAULT_HDF5 = "/home/yuqi/astribot_kinematics/tests/data/0710_Microwave_S8_episode_0.hdf5"

# SDK recorder layout (25 dims); see tests/test_against_sdk_hdf5.py.
TORSO_SLICE = slice(3, 7)
ARM_LEFT_SLICE = slice(7, 14)
ARM_RIGHT_SLICE = slice(15, 22)

EEF_KEYS = (CHAIN_TORSO, CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryData:
    q_user: np.ndarray
    gt: Dict[str, np.ndarray]
    v: Dict[str, np.ndarray]


def _load_trajectory(path: str) -> TrajectoryData:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "r") as f:
        joints_cmd = f["joints_dict/joints_position_command"][()]
        joints_vel = f["joints_dict/joints_velocity_state"][()]
        gt = {key: f[f"command_poses_dict/{key}"][()] for key in EEF_KEYS}

    q_user = np.concatenate(
        [joints_cmd[:, TORSO_SLICE], joints_cmd[:, ARM_LEFT_SLICE], joints_cmd[:, ARM_RIGHT_SLICE]],
        axis=1,
    )
    v = {
        CHAIN_TORSO: np.linalg.norm(joints_vel[:, TORSO_SLICE], axis=1),
        CHAIN_ARM_LEFT: np.linalg.norm(joints_vel[:, ARM_LEFT_SLICE], axis=1),
        CHAIN_ARM_RIGHT: np.linalg.norm(joints_vel[:, ARM_RIGHT_SLICE], axis=1),
    }
    return TrajectoryData(q_user=q_user, gt=gt, v=v)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
def _position_error_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3] - b[:3]) * 1000.0)


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two unit quaternions, in degrees."""
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
    bar = "=" * 78
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
# Metric 1 + 2: Position & orientation accuracy
# ---------------------------------------------------------------------------
def evaluate_pose_accuracy(
    fk: AstribotFK,
    data: TrajectoryData,
    frame_indices: np.ndarray,
    quasi_static_threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return per-frame position (mm) and orientation (deg) errors per EEF."""
    pos_err_mm = {key: np.empty(frame_indices.size, dtype=np.float64) for key in EEF_KEYS}
    rot_err_deg = {key: np.empty(frame_indices.size, dtype=np.float64) for key in EEF_KEYS}

    for i, idx in enumerate(frame_indices):
        predictions = fk.forward(data.q_user[idx], links=list(EEF_KEYS))
        for key in EEF_KEYS:
            pred = predictions[key]
            gt = data.gt[key][idx]
            pos_err_mm[key][i] = _position_error_mm(pred, gt)
            rot_err_deg[key][i] = _quat_angle_deg(pred, gt)

    _section("1. Position accuracy (mm)")
    rows: List[List[str]] = []
    for key in EEF_KEYS:
        rows.append(_fmt_stats_row(f"{key} (all)", _summary(pos_err_mm[key]), pos_err_mm[key].size, "mm"))
        qs_mask = data.v[key][frame_indices] < quasi_static_threshold
        if qs_mask.any():
            rows.append(
                _fmt_stats_row(
                    f"{key} (|v|<{quasi_static_threshold})",
                    _summary(pos_err_mm[key][qs_mask]),
                    int(qs_mask.sum()),
                    "mm",
                )
            )
    _print_table(STATS_HEADERS, rows)

    _section("2. Orientation accuracy (deg)")
    rows = []
    for key in EEF_KEYS:
        rows.append(_fmt_stats_row(f"{key} (all)", _summary(rot_err_deg[key]), rot_err_deg[key].size, "deg"))
        qs_mask = data.v[key][frame_indices] < quasi_static_threshold
        if qs_mask.any():
            rows.append(
                _fmt_stats_row(
                    f"{key} (|v|<{quasi_static_threshold})",
                    _summary(rot_err_deg[key][qs_mask]),
                    int(qs_mask.sum()),
                    "deg",
                )
            )
    _print_table(STATS_HEADERS, rows)

    _sub("error vs joint-speed correlation (sanity check against timestamp jitter)")
    corr_headers = ("frame", "corr(pos_mm, |v|)", "corr(rot_deg, |v|)")
    corr_rows: List[List[str]] = []
    for key in EEF_KEYS:
        v_sel = data.v[key][frame_indices]
        if np.std(v_sel) < 1e-9:
            corr_pos = corr_rot = float("nan")
        else:
            corr_pos = float(np.corrcoef(pos_err_mm[key], v_sel)[0, 1])
            corr_rot = float(np.corrcoef(rot_err_deg[key], v_sel)[0, 1])
        corr_rows.append([key, f"{corr_pos:+.4f}", f"{corr_rot:+.4f}"])
    _print_table(corr_headers, corr_rows)

    return pos_err_mm, rot_err_deg


# ---------------------------------------------------------------------------
# Metric 3: Frame / tool-frame consistency
# ---------------------------------------------------------------------------
def evaluate_frame_consistency(fk: AstribotFK, data: TrajectoryData, frame_indices: np.ndarray) -> None:
    """Check that tool frames are offset from EEF frames exactly by the URDF-baked
    tool offset, and that the SDK world-base offset is applied consistently."""
    _section("3. Frame / tool-frame consistency")

    tool_pairs = (
        (CHAIN_ARM_LEFT, "astribot_arm_left_tool"),
        (CHAIN_ARM_RIGHT, "astribot_arm_right_tool"),
    )

    tool_headers = (
        "pair",
        "N",
        "mean offset (mm)",
        "std (mm)",
        "max (mm)",
        "direction (xyz, mean)",
    )
    tool_rows: List[List[str]] = []
    for eef_key, tool_key in tool_pairs:
        deltas = np.empty((frame_indices.size, 3), dtype=np.float64)
        for i, idx in enumerate(frame_indices):
            poses = fk.forward(data.q_user[idx], links=[eef_key, tool_key])
            deltas[i] = poses[eef_key][:3] - poses[tool_key][:3]
        norms_mm = np.linalg.norm(deltas, axis=1) * 1000.0
        mean_dir = deltas.mean(axis=0)
        tool_rows.append(
            [
                f"{eef_key} vs {tool_key}",
                str(norms_mm.size),
                f"{norms_mm.mean():.4f}",
                f"{norms_mm.std():.4f}",
                f"{norms_mm.max():.4f}",
                f"[{mean_dir[0]:+.5f}, {mean_dir[1]:+.5f}, {mean_dir[2]:+.5f}] m",
            ]
        )
    _print_table(tool_headers, tool_rows)

    _sub("world <-> URDF base offset (SDK weld_to_base_pose)")
    fk_world = AstribotFK(apply_world_base_offset=True)
    fk_urdf = AstribotFK(apply_world_base_offset=False)
    q0 = np.zeros(fk_world.nq)
    rows: List[List[str]] = []
    for key in EEF_KEYS:
        world_pose = fk_world.forward(q0, links=[key])[key]
        urdf_pose = fk_urdf.forward(q0, links=[key])[key]
        delta = world_pose[:3] - urdf_pose[:3]
        rows.append(
            [
                key,
                f"[{world_pose[0]:+.4f}, {world_pose[1]:+.4f}, {world_pose[2]:+.4f}]",
                f"[{urdf_pose[0]:+.4f}, {urdf_pose[1]:+.4f}, {urdf_pose[2]:+.4f}]",
                f"[{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}] m",
            ]
        )
    _print_table(
        ("frame", "world xyz (m)", "urdf xyz (m)", "delta (m)"),
        rows,
    )


# ---------------------------------------------------------------------------
# Metric 4: Jacobian correctness
# ---------------------------------------------------------------------------
def evaluate_jacobian(fk: AstribotFK, data: TrajectoryData, n_samples: int, eps: float) -> None:
    """Compare the analytical 3xN translational Jacobian against central FD."""
    _section("4. Jacobian correctness (analytical vs central finite difference)")

    rng = np.random.default_rng(12345)
    total = data.q_user.shape[0]
    sample_idx = rng.choice(total, size=min(n_samples, total), replace=False)

    rows: List[List[str]] = []
    for key in (CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT, CHAIN_TORSO):
        abs_errs: List[float] = []
        rel_errs: List[float] = []
        max_elem: List[float] = []
        for idx in sample_idx:
            q = data.q_user[idx].astype(np.float64)
            J = fk.jacobian(q, link=key, reference="local_world_aligned")[:3]

            J_fd = np.zeros((3, fk.nq), dtype=np.float64)
            for j in range(fk.nq):
                qp = q.copy(); qp[j] += eps
                qm = q.copy(); qm[j] -= eps
                pp = fk.forward(qp, links=[key])[key][:3]
                pm = fk.forward(qm, links=[key])[key][:3]
                J_fd[:, j] = (pp - pm) / (2.0 * eps)

            diff = J - J_fd
            abs_errs.append(float(np.linalg.norm(diff)))
            denom = max(float(np.linalg.norm(J_fd)), 1e-9)
            rel_errs.append(abs_errs[-1] / denom)
            max_elem.append(float(np.max(np.abs(diff))))

        rows.append(
            [
                key,
                str(sample_idx.size),
                f"{np.mean(abs_errs):.3e}",
                f"{np.max(abs_errs):.3e}",
                f"{np.mean(rel_errs):.3e}",
                f"{np.max(rel_errs):.3e}",
                f"{np.max(max_elem):.3e}",
            ]
        )
    _print_table(
        (
            "frame",
            "N",
            "||J-J_fd||_F mean",
            "||J-J_fd||_F max",
            "rel mean",
            "rel max",
            "max |elem|",
        ),
        rows,
    )


# ---------------------------------------------------------------------------
# Metric 5: Computational efficiency
# ---------------------------------------------------------------------------
def evaluate_efficiency(fk: AstribotFK, data: TrajectoryData, n_calls: int) -> None:
    _section("5. Computational efficiency")

    total = data.q_user.shape[0]
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(total, size=min(n_calls, total), replace=False)
    qs = data.q_user[sample_idx]

    # Warm-up
    for q in qs[: min(5, qs.shape[0])]:
        fk.forward(q)

    def _bench_forward(links: Sequence[str]) -> Tuple[np.ndarray, float]:
        times = np.empty(qs.shape[0], dtype=np.float64)
        t0 = time.perf_counter()
        for i, q in enumerate(qs):
            ts = time.perf_counter()
            fk.forward(q, links=list(links))
            times[i] = time.perf_counter() - ts
        total_s = time.perf_counter() - t0
        return times, total_s

    t_all, total_all = _bench_forward(list(EEF_KEYS))
    t_left, total_left = _bench_forward([CHAIN_ARM_LEFT])

    rows: List[List[str]] = []

    def _row(label: str, times_s: np.ndarray, total_s: float) -> List[str]:
        ms = times_s * 1000.0
        return [
            label,
            str(times_s.size),
            f"{ms.mean():.4f}",
            f"{np.median(ms):.4f}",
            f"{np.percentile(ms, 95):.4f}",
            f"{np.percentile(ms, 99):.4f}",
            f"{ms.max():.4f}",
            f"{times_s.size / total_s:,.1f}",
        ]

    rows.append(_row("forward(3 EEF keys)", t_all, total_all))
    rows.append(_row("forward(1 EEF key)", t_left, total_left))

    # Jacobian timing
    t_jac = np.empty(qs.shape[0], dtype=np.float64)
    t0 = time.perf_counter()
    for i, q in enumerate(qs):
        ts = time.perf_counter()
        fk.jacobian(q, link=CHAIN_ARM_LEFT)
        t_jac[i] = time.perf_counter() - ts
    total_jac = time.perf_counter() - t0
    rows.append(_row("jacobian(arm_left)", t_jac, total_jac))

    # Batched FK
    q_batch = qs
    t_batch_start = time.perf_counter()
    fk.forward_batch(q_batch, links=list(EEF_KEYS))
    t_batch = time.perf_counter() - t_batch_start
    rows.append(
        [
            "forward_batch(3 EEF keys)",
            str(q_batch.shape[0]),
            f"{(t_batch / q_batch.shape[0]) * 1000:.4f}",
            "-",
            "-",
            "-",
            "-",
            f"{q_batch.shape[0] / t_batch:,.1f}",
        ]
    )

    _print_table(
        (
            "operation",
            "N",
            "mean (ms)",
            "median (ms)",
            "p95 (ms)",
            "p99 (ms)",
            "max (ms)",
            "throughput (calls/s)",
        ),
        rows,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive FK validation on an HDF5 trajectory.")
    parser.add_argument("--hdf5", default=DEFAULT_HDF5, help="Path to trajectory HDF5 file.")
    parser.add_argument("--step", type=int, default=1, help="Evaluate every Nth frame.")
    parser.add_argument(
        "--quasi-static-threshold",
        type=float,
        default=0.1,
        help="Joint-speed threshold (rad/s) for the quasi-static subset.",
    )
    parser.add_argument("--jacobian-samples", type=int, default=30, help="Samples used for Jacobian FD check.")
    parser.add_argument("--jacobian-eps", type=float, default=1e-6, help="Finite-difference epsilon.")
    parser.add_argument("--bench-samples", type=int, default=500, help="Samples used for efficiency benchmarks.")
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("--step must be > 0.")

    data = _load_trajectory(args.hdf5)
    fk = AstribotFK()
    frame_indices = np.arange(0, data.q_user.shape[0], args.step, dtype=np.int64)

    _section("Astribot FK validation report")
    print(f" HDF5 file          : {args.hdf5}")
    print(f" Total frames       : {data.q_user.shape[0]}")
    print(f" Sampled frames     : {frame_indices.size} (step={args.step})")
    print(f" Quasi-static v<    : {args.quasi_static_threshold} rad/s")
    print(f" Jacobian FD eps    : {args.jacobian_eps}")
    print(f" Model DoF (user)   : {fk.nq}")

    evaluate_pose_accuracy(fk, data, frame_indices, args.quasi_static_threshold)
    evaluate_frame_consistency(fk, data, frame_indices)
    evaluate_jacobian(fk, data, args.jacobian_samples, args.jacobian_eps)
    evaluate_efficiency(fk, data, args.bench_samples)

    print("\n[Done]")


if __name__ == "__main__":
    main()
