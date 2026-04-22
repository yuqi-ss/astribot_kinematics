"""Interactive IK driver with Meshcat visualisation.

Type commands in the terminal to nudge the end-effector targets of the Astribot
S1 and watch the IK solution appear in the Meshcat viewer in real time. The
solver runs in *two stages* per command:

1. Torso IK:  ``AstribotIK(chains=[CHAIN_TORSO]).solve`` warm-started from the
   current joint vector updates only the 4 torso joints to reach the torso
   target.
2. Arms IK:   ``AstribotIK(chains=[CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT]).solve``
   warm-started from the post-torso joint vector updates only the 14 arm
   joints to reach the left / right arm targets, leaving the torso frozen.

This mirrors a common controller pattern (move the torso first, then track
the hands) and keeps the IK problem well-conditioned.

Command reference (type ``help`` inside the REPL for the live copy):

    Targets are indexed by chain:  ``torso|t`` | ``left|l`` | ``right|r``

    <chain> +x|+y|+z|-x|-y|-z <meters>     relative translation (world frame)
    <chain> xyz <x> <y> <z>                absolute position  (keep orientation)
    <chain> rx|ry|rz <degrees>             relative rotation around the end-
                                           effector's own local axis
    show                                   print current targets / achieved
    reset                                  back to q=0 and the initial targets
    undo                                   revert the last accepted command
    help | ?                               print command help
    quit | q | exit                        leave the REPL

Usage::

    python tests/simulation/sim_interactive_ik.py
    python tests/simulation/sim_interactive_ik.py --no-open   # do not auto-open browser
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from _sim_common import (  # noqa: E402
    position_error_mm,
    quat_angle_deg,
    resolve_meshes_dir,
    section,
)

from astribot_kinematics import (  # noqa: E402
    AstribotFK,
    AstribotIK,
    CHAIN_ARM_LEFT,
    CHAIN_ARM_RIGHT,
    CHAIN_TORSO,
)


# =============================================================================
# Configuration defaults
# =============================================================================
DEFAULT_MARKER_RADIUS = 0.025
DEFAULT_ACHIEVED_OPACITY = 0.55
DEFAULT_IK_MAX_ITERS = 200
DEFAULT_IK_TOL_POS = 1e-4       # m
DEFAULT_IK_TOL_ROT = 1e-3       # rad
DEFAULT_IK_DAMPING = 1e-4

MARKER_COLOR_ACHIEVED = 0x2ECC71   # green
MARKER_COLOR_TARGET = 0xE74C3C     # red

CHAIN_ALIASES = {
    "t": CHAIN_TORSO,    "torso": CHAIN_TORSO,
    "l": CHAIN_ARM_LEFT, "left":  CHAIN_ARM_LEFT,
    "r": CHAIN_ARM_RIGHT, "right": CHAIN_ARM_RIGHT,
}
ALL_KEYS = (CHAIN_TORSO, CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT)

HELP_TEXT = """\
 Commands (chain = torso|t | left|l | right|r):

   <chain> +x|+y|+z|-x|-y|-z <meters>     relative translation (world frame)
   <chain> xyz <x> <y> <z>                absolute position (orientation kept)
   <chain> rx|ry|rz <degrees>             relative rotation around local axis
   show                                   print current targets / achieved
   reset                                  back to q=0 and initial targets
   undo                                   revert the last accepted command
   help | ?                               show this help
   quit | q | exit                        leave the REPL

 Examples:
   left +x 0.05                           move left hand 5 cm in +X
   right xyz 0.35 -0.25 1.10              snap right hand to absolute position
   torso rz 15                            rotate torso EEF 15 deg around local Z
"""


# =============================================================================
# Quaternion helpers (xyzw convention, matches the rest of the project)
# =============================================================================
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _axis_angle_quat(axis: str, deg: float) -> np.ndarray:
    axis_vec = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis]
    half = np.deg2rad(deg) * 0.5
    s, c = np.sin(half), np.cos(half)
    return np.array([axis_vec[0] * s, axis_vec[1] * s, axis_vec[2] * s, c], dtype=np.float64)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


# =============================================================================
# Meshcat helpers
# =============================================================================
def _require_imports():
    try:
        import pinocchio as pin  # noqa: F401
        from pinocchio.visualize import MeshcatVisualizer  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pinocchio is required. `pip install pin`.") from exc
    try:
        import meshcat  # noqa: F401
        import meshcat.geometry  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("meshcat is required. `pip install meshcat`.") from exc


def _build_geom_parallel(pin_module, model, urdf_path: str, meshes_dir: str):
    def _build(geom_type):
        return pin_module.buildGeomFromUrdf(
            model, urdf_path, geom_type, package_dirs=[meshes_dir]
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_v = pool.submit(_build, pin_module.GeometryType.VISUAL)
        fut_c = pool.submit(_build, pin_module.GeometryType.COLLISION)
        return fut_v.result(), fut_c.result()


def _marker_transform(xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64).ravel()
    return T


# =============================================================================
# Command parsing
# =============================================================================
class CommandError(ValueError):
    """Raised when a user command cannot be parsed."""


def _parse_chain(token: str) -> str:
    key = CHAIN_ALIASES.get(token.lower())
    if key is None:
        raise CommandError(f"unknown chain '{token}' (use torso|left|right)")
    return key


def _parse_float(token: str, name: str) -> float:
    try:
        return float(token)
    except ValueError as exc:
        raise CommandError(f"expected number for {name}, got '{token}'") from exc


# =============================================================================
# Driver
# =============================================================================
class IKSession:
    """Owns the robot state, IK solvers, and the live Meshcat scene."""

    def __init__(self, args: argparse.Namespace):
        _require_imports()
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer
        import meshcat.geometry as meshcat_geom

        self.pin = pin
        self.meshcat_geom = meshcat_geom

        self.args = args
        self.fk_world = AstribotFK()                                  # SDK world frame
        self.fk_urdf = AstribotFK(apply_world_base_offset=False)      # for Meshcat display
        self.ik_torso = AstribotIK(fk=self.fk_world, chains=[CHAIN_TORSO])
        self.ik_arms = AstribotIK(
            fk=self.fk_world, chains=[CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT]
        )

        urdf_path = self.fk_world.urdf_path
        meshes_dir = resolve_meshes_dir(args.meshes_dir)
        model = pin.buildModelFromUrdf(urdf_path)
        try:
            visual_model, collision_model = _build_geom_parallel(
                pin, model, urdf_path, meshes_dir
            )
        except Exception:  # pragma: no cover
            visual_model = pin.buildGeomFromUrdf(
                model, urdf_path, pin.GeometryType.VISUAL, package_dirs=[meshes_dir]
            )
            collision_model = pin.buildGeomFromUrdf(
                model, urdf_path, pin.GeometryType.COLLISION, package_dirs=[meshes_dir]
            )
        self.viz = MeshcatVisualizer(model, collision_model, visual_model)
        self.viz.initViewer(open=not args.no_open)
        self.viz.loadViewerModel()
        self.z_offset = self.fk_world._world_T_base.translation

        self.q = np.zeros(self.fk_world.nq, dtype=np.float64)
        init_poses = self.fk_world.forward(self.q, links=list(ALL_KEYS))
        self.initial_targets: Dict[str, np.ndarray] = {
            k: init_poses[k].copy() for k in ALL_KEYS
        }
        self.targets: Dict[str, np.ndarray] = copy.deepcopy(self.initial_targets)
        self.history: List[Tuple[Dict[str, np.ndarray], np.ndarray]] = []

        self._install_markers()
        self._display_current_state(initial=True)

        section("Interactive IK ready")
        print(f" URDF file    : {urdf_path}")
        print(f" Viewer URL   : {self.viz.viewer.url()}")
        print(f" IK params    : max_iters={args.ik_max_iters}, tol_pos={args.ik_tol_pos} m,"
              f" tol_rot={args.ik_tol_rot} rad, damping={args.ik_damping}")
        print(" Type `help` for the command list.\n")

    # ------------------------------------------------------------------ viz
    def _install_markers(self) -> None:
        mg = self.meshcat_geom
        for key in ALL_KEYS:
            self.viz.viewer[f"markers/achieved/{key}"].set_object(
                mg.Sphere(self.args.marker_radius),
                mg.MeshLambertMaterial(
                    color=MARKER_COLOR_ACHIEVED,
                    reflectivity=0.5,
                    opacity=float(self.args.achieved_opacity),
                    transparent=True,
                ),
            )
            self.viz.viewer[f"markers/target/{key}"].set_object(
                mg.Sphere(self.args.marker_radius * 0.65),
                mg.MeshLambertMaterial(color=MARKER_COLOR_TARGET, reflectivity=0.5),
            )

    def _display_current_state(self, initial: bool = False) -> None:
        q_pin = self.fk_urdf.to_pin_q(self.q)
        self.viz.display(q_pin)
        achieved = self.fk_world.forward(self.q, links=list(ALL_KEYS))
        # Meshcat receives poses in the URDF root frame (viewer's own frame),
        # so we subtract the world->base offset that the SDK-world FK applies.
        for key in ALL_KEYS:
            self.viz.viewer[f"markers/achieved/{key}"].set_transform(
                _marker_transform(achieved[key][:3] - self.z_offset)
            )
            self.viz.viewer[f"markers/target/{key}"].set_transform(
                _marker_transform(self.targets[key][:3] - self.z_offset)
            )
        if initial and (not self.args.no_confirm_start) and sys.stdin.isatty():
            print(" Confirm the full robot mesh is visible, then press Enter to continue...")
            try:
                input()
            except EOFError:
                pass

    # ------------------------------------------------------------------ IK
    def _solve(self, proposed_targets: Dict[str, np.ndarray]) -> Tuple[bool, Dict[str, dict]]:
        """Two-stage IK: torso first, then both arms with torso frozen."""
        q_init = self.q.copy()
        q_mid, ok_t, info_t = self.ik_torso.solve(
            {CHAIN_TORSO: proposed_targets[CHAIN_TORSO]},
            q_init=q_init,
            max_iters=self.args.ik_max_iters,
            tol_pos=self.args.ik_tol_pos,
            tol_rot=self.args.ik_tol_rot,
            damping=self.args.ik_damping,
        )
        q_new, ok_a, info_a = self.ik_arms.solve(
            {
                CHAIN_ARM_LEFT: proposed_targets[CHAIN_ARM_LEFT],
                CHAIN_ARM_RIGHT: proposed_targets[CHAIN_ARM_RIGHT],
            },
            q_init=q_mid,
            max_iters=self.args.ik_max_iters,
            tol_pos=self.args.ik_tol_pos,
            tol_rot=self.args.ik_tol_rot,
            damping=self.args.ik_damping,
        )
        ok = bool(ok_t and ok_a)
        infos = {"torso": info_t, "arms": info_a}
        if ok:
            self.history.append((copy.deepcopy(self.targets), self.q.copy()))
            self.targets = proposed_targets
            self.q = q_new
            self._display_current_state()
        return ok, infos

    # ------------------------------------------------------------------ cmd
    def _cmd_show(self) -> None:
        achieved = self.fk_world.forward(self.q, links=list(ALL_KEYS))
        print("\n  end-effector state (SDK world frame):")
        for key in ALL_KEYS:
            t = self.targets[key]
            a = achieved[key]
            pos_mm = position_error_mm(t, a)
            rot_deg = quat_angle_deg(t, a)
            print(
                f"   [{key:20s}] target xyz=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})  "
                f"quat=({t[3]:+.3f}, {t[4]:+.3f}, {t[5]:+.3f}, {t[6]:+.3f})"
            )
            print(
                f"   {'':22s} achv.  xyz=({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f})  "
                f"err pos={pos_mm:.2f} mm, rot={rot_deg:.3f} deg"
            )
        print()

    def _cmd_reset(self) -> None:
        self.history.append((copy.deepcopy(self.targets), self.q.copy()))
        self.targets = copy.deepcopy(self.initial_targets)
        self.q = np.zeros(self.fk_world.nq, dtype=np.float64)
        self._display_current_state()
        print("  -> reset to q=0 and initial targets.")

    def _cmd_undo(self) -> None:
        if not self.history:
            print("  (nothing to undo)")
            return
        self.targets, self.q = self.history.pop()
        self._display_current_state()
        print("  -> reverted to previous state.")

    def _propose_translation(self, key: str, delta_xyz: np.ndarray) -> Dict[str, np.ndarray]:
        proposed = {k: v.copy() for k, v in self.targets.items()}
        proposed[key][:3] += delta_xyz
        return proposed

    def _propose_abs_xyz(self, key: str, xyz: np.ndarray) -> Dict[str, np.ndarray]:
        proposed = {k: v.copy() for k, v in self.targets.items()}
        proposed[key][:3] = xyz
        return proposed

    def _propose_rotation(self, key: str, axis: str, deg: float) -> Dict[str, np.ndarray]:
        proposed = {k: v.copy() for k, v in self.targets.items()}
        cur_quat = proposed[key][3:]
        # Right-multiply applies the delta in the end-effector's local frame.
        new_quat = _normalize_quat(_quat_mul(cur_quat, _axis_angle_quat(axis, deg)))
        proposed[key][3:] = new_quat
        return proposed

    # ------------------------------------------------------------------ repl
    def handle(self, line: str) -> bool:
        """Return False when the REPL should exit."""
        tokens = line.strip().split()
        if not tokens:
            return True
        head = tokens[0].lower()

        if head in {"q", "quit", "exit"}:
            return False
        if head in {"help", "?", "h"}:
            print(HELP_TEXT)
            return True
        if head == "show":
            self._cmd_show()
            return True
        if head == "reset":
            self._cmd_reset()
            return True
        if head == "undo":
            self._cmd_undo()
            return True

        # chain-targeted commands
        try:
            key = _parse_chain(head)
        except CommandError as exc:
            print(f"  ! {exc}. Type `help` for syntax.")
            return True
        if len(tokens) < 2:
            print("  ! missing sub-command. Type `help` for syntax.")
            return True
        op = tokens[1].lower()

        try:
            proposed = self._dispatch(key, op, tokens[2:])
        except CommandError as exc:
            print(f"  ! {exc}")
            return True

        t0 = time.perf_counter()
        ok, infos = self._solve(proposed)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        it_t = int(infos["torso"].get("iterations", 0))
        it_a = int(infos["arms"].get("iterations", 0))
        status = "OK " if ok else "MISS"
        achieved = self.fk_world.forward(self.q, links=list(ALL_KEYS))
        tgt = self.targets[key] if ok else proposed[key]
        pos_mm = position_error_mm(tgt, achieved[key])
        rot_deg = quat_angle_deg(tgt, achieved[key])
        print(
            f"  [{status}] {dt_ms:6.1f} ms  iters(torso/arms)={it_t}/{it_a}  "
            f"{key} residual: pos={pos_mm:.2f} mm, rot={rot_deg:.3f} deg"
        )
        if not ok:
            print("         -> IK did not converge; targets unchanged.")
        return True

    def _dispatch(self, key: str, op: str, rest: List[str]) -> Dict[str, np.ndarray]:
        # relative translation: +x / -x / +y / -y / +z / -z <meters>
        if len(op) == 2 and op[0] in "+-" and op[1] in "xyz":
            if len(rest) != 1:
                raise CommandError(f"'{op}' expects 1 number (meters)")
            sign = 1.0 if op[0] == "+" else -1.0
            amount = sign * _parse_float(rest[0], "meters")
            delta = np.zeros(3)
            delta["xyz".index(op[1])] = amount
            return self._propose_translation(key, delta)

        if op == "xyz":
            if len(rest) != 3:
                raise CommandError("'xyz' expects 3 numbers: x y z")
            xyz = np.array([_parse_float(v, ax) for v, ax in zip(rest, ("x", "y", "z"))])
            return self._propose_abs_xyz(key, xyz)

        if op in {"rx", "ry", "rz"}:
            if len(rest) != 1:
                raise CommandError(f"'{op}' expects 1 number (degrees)")
            deg = _parse_float(rest[0], "degrees")
            return self._propose_rotation(key, op[1], deg)

        raise CommandError(f"unknown operation '{op}' for chain. Type `help`.")


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive IK driver with Meshcat visualisation.")
    parser.add_argument("--meshes-dir", default=None, help="Directory containing s1_* mesh folders.")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open the browser tab.")
    parser.add_argument(
        "--no-confirm-start", action="store_true",
        help="Skip the Enter confirmation before the REPL starts.",
    )
    parser.add_argument("--marker-radius", type=float, default=DEFAULT_MARKER_RADIUS,
                        help="Marker sphere radius (m).")
    parser.add_argument("--achieved-opacity", type=float, default=DEFAULT_ACHIEVED_OPACITY,
                        help="Opacity of the green achieved marker in [0, 1].")
    parser.add_argument("--ik-max-iters", type=int, default=DEFAULT_IK_MAX_ITERS,
                        help="IK max iterations per stage.")
    parser.add_argument("--ik-tol-pos", type=float, default=DEFAULT_IK_TOL_POS,
                        help="IK position tolerance (m).")
    parser.add_argument("--ik-tol-rot", type=float, default=DEFAULT_IK_TOL_ROT,
                        help="IK rotation tolerance (rad).")
    parser.add_argument("--ik-damping", type=float, default=DEFAULT_IK_DAMPING,
                        help="DLS damping factor.")
    args = parser.parse_args()
    if not (0.0 <= args.achieved_opacity <= 1.0):
        raise ValueError("--achieved-opacity must be within [0, 1].")

    session = IKSession(args)
    print(HELP_TEXT)

    try:
        while True:
            try:
                line = input(" ik> ")
            except EOFError:
                print()
                break
            if not session.handle(line):
                break
    except KeyboardInterrupt:
        print("\n[sim_interactive_ik] interrupted by user.")
    print("[Done]")


if __name__ == "__main__":
    main()
