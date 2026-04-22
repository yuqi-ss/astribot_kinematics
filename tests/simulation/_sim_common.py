"""Shared helpers for the simulation-based FK / IK validation scripts.

These utilities are intentionally small and dependency-light so each sim script
(`sim_meshcat_replay.py`, `sim_mujoco_fk.py`, `sim_mujoco_ik.py`) can be run
standalone with `python tests/simulation/<script>.py`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Make the astribot_kinematics package importable when the script is executed
# directly (i.e. without an editable install).
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Canonical defaults.
# ---------------------------------------------------------------------------
DEFAULT_HDF5 = "/home/yuqi/astribot_kinematics/tests/data/0710_Microwave_S8_episode_0.hdf5"

# Mesh directory bundled with this package (beside the URDF). URDF mesh paths
# (``s1_torso/...``, ``s1_arm/...``, ``s1_head/...``) resolve relative to this
# dir, so Meshcat / MuJoCo can display the robot without astribot_sdk.
# Override via the ``ASTRIBOT_MESHES_DIR`` env var or a ``--meshes-dir`` flag.
DEFAULT_MESHES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "astribot_kinematics",
    "assets",
    "meshes",
)

# SDK recorder layout (25-dim whole-body vector), same as elsewhere.
TORSO_SLICE = slice(3, 7)
ARM_LEFT_SLICE = slice(7, 14)
ARM_RIGHT_SLICE = slice(15, 22)

EEF_KEYS = ("astribot_torso", "astribot_arm_left", "astribot_arm_right")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryData:
    q_user: np.ndarray                      # (T, 18)
    gt: Dict[str, np.ndarray]               # key -> (T, 7)


def load_trajectory(path: str):
    """Load the 18-DoF user joint trajectory and ground-truth SDK poses.

    Requires ``h5py``. The returned ``q_user`` follows the canonical layout
    ``[torso(4), arm_left(7), arm_right(7)]``.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("h5py is required. Install with `pip install h5py`.") from exc

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


def resolve_meshes_dir(cli_value: str | None = None) -> str:
    """Pick the meshes directory (CLI arg > env var > hardcoded default)."""
    path = cli_value or os.environ.get("ASTRIBOT_MESHES_DIR") or DEFAULT_MESHES_DIR
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Meshes directory not found: {path}. "
            "Pass --meshes-dir <path>, set $ASTRIBOT_MESHES_DIR, or install astribot_sdk."
        )
    return path


# ---------------------------------------------------------------------------
# Pose math helpers
# ---------------------------------------------------------------------------
def position_error_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3] - b[:3]) * 1000.0)


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle (deg) between two xyzw unit quaternions."""
    qa = a[3:] / max(np.linalg.norm(a[3:]), 1e-12)
    qb = b[3:] / max(np.linalg.norm(b[3:]), 1e-12)
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def mat3_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a [x, y, z, w] unit quaternion."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = float(R[0, 0] + R[1, 1] + R[2, 2])
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def summary(values: np.ndarray) -> Dict[str, float]:
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
def section(title: str, width: int = 82) -> None:
    bar = "=" * width
    print(f"\n{bar}\n {title}\n{bar}")


def sub(title: str) -> None:
    print(f"\n-- {title} --")


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    if not rows:
        print("(no data)")
        return
    cols = list(zip(*([headers] + list(rows))))
    widths = [max(len(str(c)) for c in col) for col in cols]

    def fmt(row: Sequence[str]) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(row, widths))

    sep = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))


STATS_HEADERS = ("frame", "N", "mean", "median", "rmse", "p95", "p99", "max", "std")


def fmt_stats_row(label: str, stats: Dict[str, float], count: int, unit: str) -> List[str]:
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
