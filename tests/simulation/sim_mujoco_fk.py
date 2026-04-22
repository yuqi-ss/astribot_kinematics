"""Cross-check AstribotFK against MuJoCo's own forward kinematics.

Loads the bundled URDF into MuJoCo, replays the HDF5 trajectory, and at each
frame compares the body pose reported by MuJoCo (after ``mj_kinematics``)
against the pose returned by :class:`astribot_kinematics.AstribotFK` with the
SDK world-base offset disabled (both are expressed in the URDF root frame).

Passing this test confirms that:
- Joint-name / joint-order mapping is consistent between our library and MuJoCo
- The URDF is interpreted identically by pinocchio and MuJoCo
- End-effector link transforms agree down to numerical precision

Usage:
    python tests/simulation/sim_mujoco_fk.py
    python tests/simulation/sim_mujoco_fk.py --step 1 --list-bodies
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from _sim_common import (  # noqa: E402
    DEFAULT_HDF5,
    EEF_KEYS,
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

from astribot_kinematics import AstribotFK  # noqa: E402
from astribot_kinematics.constants import FULL_JOINT_ORDER  # noqa: E402


def _require_mujoco():
    try:
        import mujoco  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mujoco is required. Install with `pip install mujoco` "
            "or `bash install.sh --sim`."
        ) from exc


MUJOCO_COMPILER_TAG = (
    '  <mujoco>\n'
    '    <compiler meshdir="{meshdir}" strippath="false" '
    'balanceinertia="true" discardvisual="false" fusestatic="false"/>\n'
    '  </mujoco>\n'
)


def _inject_mujoco_compiler(urdf_path: str, meshes_dir: str) -> str:
    """Return the URDF as a string with a <mujoco> compiler block inserted.

    MuJoCo understands this extension block inside URDFs to resolve mesh
    paths relative to ``meshdir``.
    """
    with open(urdf_path, "r", encoding="utf-8") as f:
        text = f.read()
    block = MUJOCO_COMPILER_TAG.format(meshdir=os.path.abspath(meshes_dir))

    match = re.search(r"<robot\b[^>]*>", text)
    if match is None:
        raise ValueError(f"Could not find <robot> tag in URDF: {urdf_path}")
    insert_pos = match.end()
    return text[:insert_pos] + "\n" + block + text[insert_pos:]


def _build_joint_map(model, joint_names: List[str]) -> np.ndarray:
    """qpos index in MuJoCo for each user-layout joint."""
    import mujoco
    out = np.empty(len(joint_names), dtype=np.int64)
    for i, jname in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"Joint '{jname}' not found in MuJoCo model.")
        out[i] = model.jnt_qposadr[jid]
    return out


def _find_body_id(model, name: str) -> int:
    import mujoco
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return int(bid)


# Link names exposed by AstribotFK frame keys (subset in EEF_LINKS).
EEF_LINK_NAMES = {
    "astribot_torso": "astribot_torso_end_effector",
    "astribot_arm_left": "astribot_arm_left_end_effector",
    "astribot_arm_right": "astribot_arm_right_end_effector",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo vs AstribotFK cross-validation on HDF5 trajectory.")
    parser.add_argument("--hdf5", default=DEFAULT_HDF5, help="HDF5 trajectory file.")
    parser.add_argument("--meshes-dir", default=None, help="Mesh directory (s1_* subfolders).")
    parser.add_argument("--step", type=int, default=1, help="Evaluate every Nth frame.")
    parser.add_argument("--list-bodies", action="store_true", help="Print all MuJoCo body names and exit.")
    args = parser.parse_args()

    _require_mujoco()
    import mujoco

    meshes_dir = resolve_meshes_dir(args.meshes_dir)

    # AstribotFK without the SDK world-base offset so we compare against
    # MuJoCo in the same URDF root frame.
    fk = AstribotFK(apply_world_base_offset=False)
    urdf_path = fk.urdf_path

    section("MuJoCo FK cross-validation")
    print(f" URDF file          : {urdf_path}")
    print(f" Meshes dir         : {meshes_dir}")
    print(f" HDF5 file          : {args.hdf5}")

    urdf_str = _inject_mujoco_compiler(urdf_path, meshes_dir)
    try:
        model = mujoco.MjModel.from_xml_string(urdf_str)
    except Exception as exc:
        raise RuntimeError(f"MuJoCo failed to load URDF: {exc}") from exc
    data = mujoco.MjData(model)

    print(f" MuJoCo nq / nv     : {model.nq} / {model.nv}")
    print(f" MuJoCo #bodies     : {model.nbody}")

    if args.list_bodies:
        section("MuJoCo bodies")
        for i in range(model.nbody):
            print(f"  [{i:3d}] {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)}")
        return

    qpos_idx = _build_joint_map(model, FULL_JOINT_ORDER)
    eef_body_ids: Dict[str, int] = {}
    for key, link_name in EEF_LINK_NAMES.items():
        bid = _find_body_id(model, link_name)
        if bid < 0:
            raise RuntimeError(
                f"Body '{link_name}' not found in MuJoCo model - it may have been "
                f"fused with its parent. Re-run with --list-bodies to inspect."
            )
        eef_body_ids[key] = bid

    traj = load_trajectory(args.hdf5)
    frame_indices = np.arange(0, traj.q_user.shape[0], max(args.step, 1), dtype=np.int64)

    pos_err = {key: np.empty(frame_indices.size, dtype=np.float64) for key in EEF_KEYS}
    rot_err = {key: np.empty(frame_indices.size, dtype=np.float64) for key in EEF_KEYS}

    for i, idx in enumerate(frame_indices):
        q_user = traj.q_user[idx]
        data.qpos[:] = 0.0
        data.qpos[qpos_idx] = q_user
        mujoco.mj_kinematics(model, data)

        fk_poses = fk.forward(q_user, links=list(EEF_KEYS))
        for key in EEF_KEYS:
            bid = eef_body_ids[key]
            mj_xyz = np.asarray(data.xpos[bid])
            mj_mat = np.asarray(data.xmat[bid]).reshape(3, 3)
            mj_quat = mat3_to_quat_xyzw(mj_mat)
            mj_pose = np.concatenate([mj_xyz, mj_quat])
            pos_err[key][i] = position_error_mm(mj_pose, fk_poses[key])
            rot_err[key][i] = quat_angle_deg(mj_pose, fk_poses[key])

    section(f"Position error vs MuJoCo (mm), {frame_indices.size} frames (step={args.step})")
    rows = [fmt_stats_row(key, summary(pos_err[key]), pos_err[key].size, "mm") for key in EEF_KEYS]
    print_table(STATS_HEADERS, rows)

    section(f"Orientation error vs MuJoCo (deg), {frame_indices.size} frames")
    rows = [fmt_stats_row(key, summary(rot_err[key]), rot_err[key].size, "deg") for key in EEF_KEYS]
    print_table(STATS_HEADERS, rows)

    global_pos = np.concatenate([pos_err[k] for k in EEF_KEYS])
    global_rot = np.concatenate([rot_err[k] for k in EEF_KEYS])
    section("Overall verdict")
    print(f" max position error (mm) : {global_pos.max():.4e}")
    print(f" max rotation error (deg): {global_rot.max():.4e}")
    if global_pos.max() < 1e-3 and global_rot.max() < 1e-3:
        print(" Status                   : PASS (numerical agreement down to 1e-3 mm / deg)")
    else:
        print(" Status                   : FAIL or non-trivial disagreement - investigate URDF parsing.")

    print("\n[Done]")


if __name__ == "__main__":
    main()
