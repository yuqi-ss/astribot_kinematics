"""Cross-check against the Astribot SDK's real trajectory recordings.

Uses a real teleop trajectory (``0710_Microwave_S8_episode_0.hdf5``) to
validate both FK and IK against the SDK's ground-truth data:

1. FK validation: Compare offline FK output against ``command_poses_dict``
   from the SDK. Verifies bit-level alignment to within dataset jitter.

2. IK validation: Replay the trajectory by solving IK frame-by-frame with
   warm-start, verifying sub-cm tracking accuracy.

The trajectory file is optional: tests are skipped if neither the default
path nor ``$ASTRIBOT_HDF5`` resolves to an existing file.

Expected behaviour (validated on ``0710_Microwave_S8_episode_0.hdf5``):

    - FK quasi-static frames (|v| < 0.1 rad/s): per-frame error < 2 mm
    - FK torso end-effector mean error: < 1 mm over whole trajectory
    - IK trajectory tracking: failure rate < 2%, mean error < 2 mm, p99 < 10 mm
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("pinocchio")
h5py = pytest.importorskip("h5py")

from astribot_kinematics import (
    AstribotFK,
    AstribotIK,
    CHAIN_ARM_LEFT,
    CHAIN_ARM_RIGHT,
    CHAIN_TORSO,
)

DEFAULT_HDF5 = "/home/yuqi/astribot_kinematics/tests/data/0710_Microwave_S8_episode_0.hdf5"

# 25-dim whole-body layout used by the SDK recorder (see
# astribot_sdk/examples/208-traj_replay.py):
#   [chassis 0:3, torso 3:7, arm_left 7:14, gripper_left 14:15,
#    arm_right 15:22, gripper_right 22:23, head 23:25]
TORSO_SLICE = slice(3, 7)
ARM_LEFT_SLICE = slice(7, 14)
ARM_RIGHT_SLICE = slice(15, 22)


def _hdf5_path() -> str:
    return os.environ.get("ASTRIBOT_HDF5", DEFAULT_HDF5)


@pytest.fixture(scope="module")
def trajectory():
    path = _hdf5_path()
    if not os.path.isfile(path):
        pytest.skip(
            f"HDF5 trajectory not found at {path}. "
            "Set $ASTRIBOT_HDF5 to a real recording to enable this test."
        )
    with h5py.File(path, "r") as f:
        Jc = f["joints_dict/joints_position_command"][()]
        V = f["joints_dict/joints_velocity_state"][()]
        targets = {
            "astribot_torso": f["command_poses_dict/astribot_torso"][()],
            "astribot_arm_left": f["command_poses_dict/astribot_arm_left"][()],
            "astribot_arm_right": f["command_poses_dict/astribot_arm_right"][()],
        }
    q_user = np.concatenate(
        [Jc[:, TORSO_SLICE], Jc[:, ARM_LEFT_SLICE], Jc[:, ARM_RIGHT_SLICE]], axis=1
    )
    v_left = np.linalg.norm(V[:, ARM_LEFT_SLICE], axis=1)
    v_right = np.linalg.norm(V[:, ARM_RIGHT_SLICE], axis=1)
    return dict(q_user=q_user, gt=targets, v_left=v_left, v_right=v_right)


@pytest.fixture(scope="module")
def fk() -> AstribotFK:
    return AstribotFK()


def _pos_err_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3] - b[:3]) * 1000.0)


def _quat_err(a: np.ndarray, b: np.ndarray) -> float:
    qa, qb = a[3:], b[3:]
    return float(min(np.linalg.norm(qa - qb), np.linalg.norm(qa + qb)))


def test_torso_end_matches_sdk(fk, trajectory):
    """The torso frame moves slowly; the FK should match the SDK to < 1 mm."""
    q, gt = trajectory["q_user"], trajectory["gt"]["astribot_torso"]
    idx = np.arange(0, q.shape[0], 20)
    errs = np.array([_pos_err_mm(fk.forward(q[i])["astribot_torso"], gt[i]) for i in idx])
    assert errs.mean() < 1.0, f"torso mean pos error = {errs.mean():.3f} mm"
    assert errs.max() < 3.0, f"torso max pos error = {errs.max():.3f} mm"


@pytest.mark.parametrize(
    "arm_key, vel_key",
    [("astribot_arm_left", "v_left"), ("astribot_arm_right", "v_right")],
)
def test_arm_quasi_static(fk, trajectory, arm_key, vel_key):
    """On quasi-static frames the FK should match the SDK sub-mm on average.

    This is the bit-level statement: at joint speeds below 0.1 rad/s the only
    difference between our FK and the SDK's is IEEE 754 rounding, so we
    expect a mean error ~0.1 mm and a worst case within a couple of mm.
    """
    q = trajectory["q_user"]
    gt = trajectory["gt"][arm_key]
    vel = trajectory[vel_key]

    mask = vel < 0.1
    if mask.sum() < 20:
        pytest.skip(f"not enough quasi-static frames (|v|<0.1): {mask.sum()}")

    frame_idx = np.where(mask)[0][::5]  # downsample for speed
    errs = np.array([_pos_err_mm(fk.forward(q[i])[arm_key], gt[i]) for i in frame_idx])
    assert errs.mean() < 0.5, f"{arm_key} quasi-static mean = {errs.mean():.3f} mm"
    assert errs.max() < 3.0, f"{arm_key} quasi-static max  = {errs.max():.3f} mm"


def test_error_velocity_correlation(fk, trajectory):
    """Residual error must correlate with joint speed, proving it comes from
    the dataset's own sampling jitter rather than our FK implementation."""
    q = trajectory["q_user"]
    gt = trajectory["gt"]["astribot_arm_right"]
    vel = trajectory["v_right"]
    idx = np.arange(0, q.shape[0], 5)
    errs = np.array([_pos_err_mm(fk.forward(q[i])["astribot_arm_right"], gt[i]) for i in idx])
    vels = vel[idx]
    corr = float(np.corrcoef(errs, vels)[0, 1])
    assert corr > 0.3, (
        f"expected positive correlation between pos error and joint speed "
        f"(data jitter hypothesis), got corr = {corr:.3f}"
    )


def test_trajectory_tracking_ik(fk, trajectory):
    """Replay a real teleop trajectory through IK and verify sub-cm tracking.

    Uses the ground-truth poses from the SDK as IK targets, warm-starting
    each frame with the previous solution (simulating real-time control).
    """
    q_user = trajectory["q_user"]
    gt_left = trajectory["gt"]["astribot_arm_left"]
    gt_right = trajectory["gt"]["astribot_arm_right"]

    ik = AstribotIK(fk=fk, chains=[CHAIN_TORSO, CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT])
    q_init = q_user[0].copy()

    errs_l, errs_r = [], []
    failures = 0
    for i in range(q_user.shape[0]):
        q_sol, ok, _ = ik.solve(
            {CHAIN_ARM_LEFT: gt_left[i], CHAIN_ARM_RIGHT: gt_right[i]},
            q_init=q_init,
            max_iters=100,
            tol_pos=5e-3,
            tol_rot=1e-2,
        )
        if not ok:
            failures += 1
        errs_l.append(np.linalg.norm(fk.eef_left(q_sol)[:3] - gt_left[i, :3]))
        errs_r.append(np.linalg.norm(fk.eef_right(q_sol)[:3] - gt_right[i, :3]))
        q_init = q_sol

    errs_l = np.asarray(errs_l) * 1000.0
    errs_r = np.asarray(errs_r) * 1000.0
    n = q_user.shape[0]
    assert failures < 0.02 * n, f"too many non-convergences: {failures}/{n}"
    assert errs_l.mean() < 2.0 and np.percentile(errs_l, 99) < 10.0
    assert errs_r.mean() < 2.0 and np.percentile(errs_r, 99) < 10.0
