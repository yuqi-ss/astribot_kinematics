"""Closed-loop IK validation: AstribotIK solves, MuJoCo verifies.

For each frame of the HDF5 trajectory:
1. Read the SDK-recorded target pose (``command_poses_dict/astribot_*``).
2. Solve IK with :class:`astribot_kinematics.AstribotIK` warm-started from the
   previous solution.
3. Push the solution ``q`` into MuJoCo (``mj_kinematics``) and read back the
   achieved end-effector poses.
4. Report the end-effector position / orientation tracking error against the
   original target, plus solver convergence rate and timing.

This tests not just the IK residual reported by the solver, but also whether
the solution is faithful once the *same* joint vector is placed into an
independent simulator (MuJoCo), which is the strongest form of FK / IK
closed-loop verification.

Usage:
    python tests/simulation/sim_mujoco_ik.py
    python tests/simulation/sim_mujoco_ik.py --step 1 --max-frames 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from _sim_common import (  # noqa: E402
    DEFAULT_HDF5,
    STATS_HEADERS,
    fmt_stats_row,
    load_trajectory,
    mat3_to_quat_xyzw,
    position_error_mm,
    print_table,
    quat_angle_deg,
    resolve_meshes_dir,
    section,
    summary,
)
from sim_mujoco_fk import (  # noqa: E402
    EEF_LINK_NAMES,
    _build_joint_map,
    _find_body_id,
    _inject_mujoco_compiler,
    _require_mujoco,
)

from astribot_kinematics import (  # noqa: E402
    AstribotFK,
    AstribotIK,
    CHAIN_ARM_LEFT,
    CHAIN_ARM_RIGHT,
    CHAIN_TORSO,
)
from astribot_kinematics.constants import FULL_JOINT_ORDER  # noqa: E402


ARM_KEYS = (CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT)


@dataclass
class ClosedLoopResult:
    pos_err_mm: Dict[str, np.ndarray]
    rot_err_deg: Dict[str, np.ndarray]
    converged: np.ndarray
    iterations: np.ndarray
    per_frame_time_s: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop IK validation via MuJoCo.")
    parser.add_argument("--hdf5", default=DEFAULT_HDF5, help="HDF5 trajectory file.")
    parser.add_argument("--meshes-dir", default=None, help="Mesh directory (s1_* subfolders).")
    parser.add_argument("--step", type=int, default=5, help="Replay every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit replay frames (0 = no limit).")
    parser.add_argument("--max-iters", type=int, default=100, help="IK max iterations per frame.")
    parser.add_argument("--tol-pos", type=float, default=5e-3, help="IK position tolerance (m).")
    parser.add_argument("--tol-rot", type=float, default=1e-2, help="IK rotation tolerance (rad).")
    parser.add_argument("--damping", type=float, default=1e-4, help="DLS damping factor.")
    parser.add_argument("--pass-pos-mm", type=float, default=5.0, help="Pass threshold for pos error (mm).")
    parser.add_argument("--pass-rot-deg", type=float, default=1.0, help="Pass threshold for rot error (deg).")
    args = parser.parse_args()

    _require_mujoco()
    import mujoco

    meshes_dir = resolve_meshes_dir(args.meshes_dir)

    # AstribotFK used for the IK solver is in the SDK world frame (to match the
    # target pose convention in HDF5); a second pure-URDF instance is used
    # only to help sanity-check MuJoCo if needed.
    fk = AstribotFK()
    ik = AstribotIK(fk=fk, chains=[CHAIN_TORSO, CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT])

    urdf_str = _inject_mujoco_compiler(fk.urdf_path, meshes_dir)
    mj_model = mujoco.MjModel.from_xml_string(urdf_str)
    mj_data = mujoco.MjData(mj_model)
    qpos_idx = _build_joint_map(mj_model, FULL_JOINT_ORDER)
    eef_body_ids = {key: _find_body_id(mj_model, link) for key, link in EEF_LINK_NAMES.items()}

    # MuJoCo reports poses in the URDF root frame; HDF5 targets are in the
    # SDK world frame. We translate MuJoCo xyz by the world-base offset so
    # both live in the same frame for comparison.
    z_offset = fk._world_T_base.translation

    traj = load_trajectory(args.hdf5)
    frame_indices = np.arange(0, traj.q_user.shape[0], max(args.step, 1), dtype=np.int64)
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]

    section("MuJoCo closed-loop IK validation")
    print(f" URDF file          : {fk.urdf_path}")
    print(f" HDF5 file          : {args.hdf5}")
    print(f" Replay frames      : {frame_indices.size} / {traj.q_user.shape[0]} (step={args.step})")
    print(f" Active chains      : {ik.active_chains}")
    print(f" IK params          : max_iters={args.max_iters}, tol_pos={args.tol_pos} m, "
          f"tol_rot={args.tol_rot} rad, damping={args.damping}")
    print(f" Pass thresholds    : pos<{args.pass_pos_mm} mm, rot<{args.pass_rot_deg} deg")

    n = frame_indices.size
    result = ClosedLoopResult(
        pos_err_mm={k: np.empty(n, dtype=np.float64) for k in ARM_KEYS},
        rot_err_deg={k: np.empty(n, dtype=np.float64) for k in ARM_KEYS},
        converged=np.zeros(n, dtype=bool),
        iterations=np.zeros(n, dtype=np.int64),
        per_frame_time_s=np.zeros(n, dtype=np.float64),
    )

    q_init = traj.q_user[frame_indices[0]].copy()
    for i, idx in enumerate(frame_indices):
        target_left = traj.gt[CHAIN_ARM_LEFT][idx]
        target_right = traj.gt[CHAIN_ARM_RIGHT][idx]

        t0 = time.perf_counter()
        q_sol, ok, info = ik.solve(
            {CHAIN_ARM_LEFT: target_left, CHAIN_ARM_RIGHT: target_right},
            q_init=q_init,
            max_iters=args.max_iters,
            tol_pos=args.tol_pos,
            tol_rot=args.tol_rot,
            damping=args.damping,
        )
        result.per_frame_time_s[i] = time.perf_counter() - t0
        result.converged[i] = bool(ok)
        result.iterations[i] = int(info.get("iterations", 0))

        mj_data.qpos[:] = 0.0
        mj_data.qpos[qpos_idx] = q_sol
        mujoco.mj_kinematics(mj_model, mj_data)

        for key in ARM_KEYS:
            bid = eef_body_ids[key]
            mj_xyz_world = np.asarray(mj_data.xpos[bid]) + z_offset
            mj_mat = np.asarray(mj_data.xmat[bid]).reshape(3, 3)
            mj_quat = mat3_to_quat_xyzw(mj_mat)
            mj_pose_world = np.concatenate([mj_xyz_world, mj_quat])

            target = target_left if key == CHAIN_ARM_LEFT else target_right
            result.pos_err_mm[key][i] = position_error_mm(mj_pose_world, target)
            result.rot_err_deg[key][i] = quat_angle_deg(mj_pose_world, target)

        q_init = q_sol

    # --- reports -----------------------------------------------------------
    section("1. Position error (MuJoCo achieved vs SDK target, mm)")
    rows = [fmt_stats_row(key, summary(result.pos_err_mm[key]), result.pos_err_mm[key].size, "mm") for key in ARM_KEYS]
    print_table(STATS_HEADERS, rows)

    section("2. Orientation error (MuJoCo achieved vs SDK target, deg)")
    rows = [fmt_stats_row(key, summary(result.rot_err_deg[key]), result.rot_err_deg[key].size, "deg") for key in ARM_KEYS]
    print_table(STATS_HEADERS, rows)

    section("3. Solver success rate")
    pos_ok = np.stack([result.pos_err_mm[k] < args.pass_pos_mm for k in ARM_KEYS], axis=0).all(axis=0)
    rot_ok = np.stack([result.rot_err_deg[k] < args.pass_rot_deg for k in ARM_KEYS], axis=0).all(axis=0)
    pose_ok = pos_ok & rot_ok
    solver_flag = float(result.converged.mean())
    rows = [
        ["solver `converged=True`", f"{int(result.converged.sum())}/{n}", f"{solver_flag * 100:.2f}%"],
        [f"MuJoCo pos < {args.pass_pos_mm} mm", f"{int(pos_ok.sum())}/{n}", f"{pos_ok.mean() * 100:.2f}%"],
        [f"MuJoCo rot < {args.pass_rot_deg} deg", f"{int(rot_ok.sum())}/{n}", f"{rot_ok.mean() * 100:.2f}%"],
        ["MuJoCo pose ok (pos & rot)", f"{int(pose_ok.sum())}/{n}", f"{pose_ok.mean() * 100:.2f}%"],
    ]
    print_table(("criterion", "count", "rate"), rows)

    section("4. Solver efficiency")
    ms = result.per_frame_time_s * 1000.0
    iters = result.iterations.astype(np.float64)
    rows = [
        ["per-frame time (ms)",
         f"{ms.size}",
         f"{ms.mean():.3f}",
         f"{np.median(ms):.3f}",
         f"{np.percentile(ms, 95):.3f}",
         f"{np.percentile(ms, 99):.3f}",
         f"{ms.max():.3f}",
         f"{ms.std():.3f}"],
        ["iterations",
         f"{iters.size}",
         f"{iters.mean():.2f}",
         f"{np.median(iters):.2f}",
         f"{np.percentile(iters, 95):.2f}",
         f"{np.percentile(iters, 99):.2f}",
         f"{iters.max():.0f}",
         f"{iters.std():.2f}"],
    ]
    print_table(("metric", "N", "mean", "median", "p95", "p99", "max", "std"), rows)
    total_s = result.per_frame_time_s.sum()
    print(f"\n total IK time  : {total_s:.3f} s")
    print(f" throughput     : {ms.size / max(total_s, 1e-9):,.1f} frames / s")

    print("\n[Done]")


if __name__ == "__main__":
    main()
