"""Replay Astribot trajectories in Meshcat for FK/IK visual validation.

Modes:
1) FK mode (default):
   - Displays commanded robot motion from HDF5 joint trajectory.
   - Green markers: AstribotFK predicted end-effector positions.
   - Red markers: SDK recorded end-effector positions.
   - Overlap indicates FK alignment.

2) IK mode:
   - Uses SDK recorded end-effector poses as targets.
   - Solves IK frame-by-frame (warm-started).
   - Displays solved robot motion.
   - Green markers: IK achieved positions.
   - Red markers: target positions.
   - Prints convergence and tracking error statistics after playback.

Usage:
    python tests/simulation/sim_meshcat_replay.py
    python tests/simulation/sim_meshcat_replay.py --mode fk --step 1 --fps 30
    python tests/simulation/sim_meshcat_replay.py --mode ik --step 1 --fps 24
"""

from __future__ import annotations

# =============================================================================
# CONFIGURATION - Edit these values to change defaults
# =============================================================================

# Input/Output paths
DEFAULT_HDF5_PATH = "/home/yuqi/astribot_kinematics/tests/data/0710_Microwave_S8_episode_0.hdf5"
DEFAULT_MESHES_DIR = None  # None = use bundled meshes under astribot_kinematics/assets/meshes

# Playback mode and timing
DEFAULT_MODE = "ik"        # "fk" or "ik"
DEFAULT_STEP = 2           # Play every Nth frame
DEFAULT_FPS = 30.0         # Playback rate (frames/s)

# Visual appearance
DEFAULT_MARKER_RADIUS = 0.02       # Marker sphere radius in meters
DEFAULT_ACHIEVED_OPACITY = 0.45    # Green ball opacity [0, 1]
DEFAULT_SHOW_ERROR_LINES = False   # Draw yellow error lines in IK mode

# IK solver parameters (only used when MODE = "ik")
DEFAULT_IK_MAX_ITERS = 100
DEFAULT_IK_TOL_POS = 5e-3          # meters
DEFAULT_IK_TOL_ROT = 1e-2          # radians
DEFAULT_IK_DAMPING = 1e-4
DEFAULT_IK_TARGET_TORSO = True     # Constrain torso in IK

# Logging and interaction
DEFAULT_PRINT_EVERY = 24           # Print IK stats every N frames (0 = disable)
DEFAULT_NO_CONFIRM_START = False   # Skip Enter confirmation before playback
DEFAULT_LOOP = False               # Auto-loop without waiting for Enter
DEFAULT_NO_OPEN = False            # Do not auto-open browser tab

# =============================================================================

import argparse

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import select
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from _sim_common import (  # noqa: E402
    DEFAULT_HDF5,
    EEF_KEYS,
    fmt_stats_row,
    load_trajectory,
    position_error_mm,
    print_table,
    quat_angle_deg,
    resolve_meshes_dir,
    section,
    STATS_HEADERS,
    summary,
)

from astribot_kinematics import (  # noqa: E402
    AstribotFK,
    AstribotIK,
    CHAIN_ARM_LEFT,
    CHAIN_ARM_RIGHT,
    CHAIN_TORSO,
)


def _require_imports():
    try:
        import pinocchio as pin  # noqa: F401
        from pinocchio.visualize import MeshcatVisualizer  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pinocchio is required. Install with `pip install pin` or "
            "`conda install -c conda-forge pinocchio`."
        ) from exc
    try:
        import meshcat  # noqa: F401
        import meshcat.geometry  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "meshcat is required. Install with `pip install meshcat` "
            "or `bash install.sh --sim`."
        ) from exc


MARKER_COLOR_FK = 0x2ECC71   # green
MARKER_COLOR_GT = 0xE74C3C   # red


def _str2bool(value: str | bool) -> bool:
    """Parse typical CLI boolean strings."""
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _marker_transform(position_xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(position_xyz, dtype=np.float64).ravel()
    return T


def _build_geom_models_parallel(pin_module, model, urdf_path: str, meshes_dir: str):
    """Build visual/collision geometry models in parallel for faster startup.

    In practice, STL parsing is IO-heavy and this can reduce startup latency on
    larger models. If parallel build fails for any reason, caller should fall
    back to sequential build to keep behavior robust.
    """

    def _build(geom_type):
        return pin_module.buildGeomFromUrdf(
            model,
            urdf_path,
            geom_type,
            package_dirs=[meshes_dir],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_visual = pool.submit(_build, pin_module.GeometryType.VISUAL)
        fut_collision = pool.submit(_build, pin_module.GeometryType.COLLISION)
        visual_model = fut_visual.result()
        collision_model = fut_collision.result()
    return visual_model, collision_model


def _print_ik_summary(
    pos_err_mm: dict[str, np.ndarray],
    rot_err_deg: dict[str, np.ndarray],
    converged_flags: np.ndarray,
    iterations: np.ndarray,
) -> None:
    section("IK summary (visual replay)")
    rows = []
    for key in (CHAIN_TORSO, CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT):
        rows.append(
            fmt_stats_row(
                f"{key} pos",
                summary(pos_err_mm[key]),
                int(pos_err_mm[key].size),
                "mm",
            )
        )
        rows.append(
            fmt_stats_row(
                f"{key} rot",
                summary(rot_err_deg[key]),
                int(rot_err_deg[key].size),
                "deg",
            )
        )
    print_table(STATS_HEADERS, rows)

    section("IK convergence (visual replay)")
    n = int(converged_flags.size)
    n_ok = int(converged_flags.sum())
    print(f" converged frames : {n_ok}/{n} ({(n_ok / max(n, 1)) * 100:.2f}%)")
    if n > 0:
        print(f" mean iterations  : {float(iterations.mean()):.2f}")
        print(f" max iterations   : {int(iterations.max())}")


def _print_fk_summary(
    pos_err_mm: dict[str, np.ndarray],
    rot_err_deg: dict[str, np.ndarray],
) -> None:
    section("FK summary (visual replay)")
    rows = []
    for key in (CHAIN_TORSO, CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT):
        rows.append(
            fmt_stats_row(
                f"{key} pos",
                summary(pos_err_mm[key]),
                int(pos_err_mm[key].size),
                "mm",
            )
        )
        rows.append(
            fmt_stats_row(
                f"{key} rot",
                summary(rot_err_deg[key]),
                int(rot_err_deg[key].size),
                "deg",
            )
        )
    print_table(STATS_HEADERS, rows)


def _poll_playback_control(paused: bool) -> tuple[bool, bool]:
    """Read non-blocking stdin control commands.

    Controls (TTY only):
    - Enter: toggle pause/resume
    - q + Enter: quit playback
    """
    if not sys.stdin.isatty():
        return paused, False
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    except (OSError, ValueError):
        return paused, False
    if not readable:
        return paused, False

    line = sys.stdin.readline()
    if line == "":  # EOF
        return paused, False
    cmd = line.strip().lower()
    if cmd == "":
        paused = not paused
        state = "paused" if paused else "resumed"
        print(f" [control] playback {state}")
        return paused, False
    if cmd in {"q", "quit", "exit"}:
        print(" [control] quit requested")
        return paused, True
    return paused, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinocchio + Meshcat trajectory replay.")
    parser.add_argument("--hdf5", default=DEFAULT_HDF5_PATH, help="HDF5 trajectory file.")
    parser.add_argument("--meshes-dir", default=DEFAULT_MESHES_DIR, help="Directory containing s1_* mesh folders.")
    parser.add_argument(
        "--mode",
        choices=("fk", "ik"),
        default=DEFAULT_MODE,
        help=f"Replay mode: fk or ik (default: {DEFAULT_MODE}).",
    )
    parser.add_argument("--step", type=int, default=DEFAULT_STEP, help="Play every Nth frame.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Playback rate (frames/s).")
    parser.add_argument(
        "--no-confirm-start",
        action="store_true",
        default=DEFAULT_NO_CONFIRM_START,
        help="Skip interactive Enter confirmation before playback starts.",
    )
    parser.add_argument("--marker-radius", type=float, default=DEFAULT_MARKER_RADIUS, help="Marker sphere radius in meters.")
    parser.add_argument(
        "--achieved-opacity",
        type=float,
        default=DEFAULT_ACHIEVED_OPACITY,
        help="Opacity of the green achieved marker in [0, 1].",
    )
    parser.add_argument("--ik-max-iters", type=int, default=DEFAULT_IK_MAX_ITERS, help="IK max iterations per frame (ik mode).")
    parser.add_argument("--ik-tol-pos", type=float, default=DEFAULT_IK_TOL_POS, help="IK position tolerance in meters (ik mode).")
    parser.add_argument("--ik-tol-rot", type=float, default=DEFAULT_IK_TOL_ROT, help="IK orientation tolerance in radians (ik mode).")
    parser.add_argument("--ik-damping", type=float, default=DEFAULT_IK_DAMPING, help="IK damping factor (ik mode).")
    parser.add_argument(
        "--ik-target-torso",
        type=_str2bool,
        nargs="?",
        const=True,
        default=DEFAULT_IK_TARGET_TORSO,
        help=f"In ik mode, whether to constrain torso end-effector target (default: {DEFAULT_IK_TARGET_TORSO}).",
    )
    parser.add_argument(
        "--show-error-lines",
        action="store_true",
        default=DEFAULT_SHOW_ERROR_LINES,
        help="In ik mode, draw yellow line segments from target to achieved points.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=DEFAULT_PRINT_EVERY,
        help="Print per-frame errors every N replayed frames (0 disables).",
    )
    parser.add_argument("--loop", action="store_true", default=DEFAULT_LOOP, help="Loop playback forever (Ctrl-C to quit).")
    parser.add_argument("--no-open", action="store_true", default=DEFAULT_NO_OPEN, help="Do not auto-open the browser tab.")
    args = parser.parse_args()
    if not (0.0 <= args.achieved_opacity <= 1.0):
        raise ValueError("--achieved-opacity must be within [0, 1].")
    if args.print_every < 0:
        raise ValueError("--print-every must be >= 0.")

    _require_imports()
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer
    import meshcat.geometry as meshcat_geom

    meshes_dir = resolve_meshes_dir(args.meshes_dir)

    fk_world = AstribotFK()                                   # SDK-world frame
    fk_urdf = AstribotFK(apply_world_base_offset=False)       # URDF root frame for Meshcat
    ik = AstribotIK(fk=fk_world, chains=[CHAIN_TORSO, CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT])
    urdf_path = fk_world.urdf_path
    data = load_trajectory(args.hdf5)

    section("Meshcat trajectory replay")
    print(f" Mode            : {args.mode}")
    print(f" URDF file       : {urdf_path}")
    print(f" Meshes dir      : {meshes_dir}")
    print(f" HDF5 file       : {args.hdf5}")
    print(f" Total frames    : {data.q_user.shape[0]}")
    print(f" Step            : {args.step} (play every Nth frame)")
    print(f" Playback rate   : {args.fps:.1f} fps")

    model = pin.buildModelFromUrdf(urdf_path)
    t_geom_start = time.perf_counter()
    try:
        visual_model, collision_model = _build_geom_models_parallel(
            pin,
            model,
            urdf_path,
            meshes_dir,
        )
        print(f" Geom load mode  : parallel (2 threads), {time.perf_counter() - t_geom_start:.2f}s")
    except Exception as exc:  # pragma: no cover
        print(f" Geom load mode  : parallel failed ({exc}), fallback to sequential")
        t_fallback = time.perf_counter()
        visual_model = pin.buildGeomFromUrdf(
            model, urdf_path, pin.GeometryType.VISUAL, package_dirs=[meshes_dir]
        )
        collision_model = pin.buildGeomFromUrdf(
            model, urdf_path, pin.GeometryType.COLLISION, package_dirs=[meshes_dir]
        )
        print(f" Geom load mode  : sequential fallback, {time.perf_counter() - t_fallback:.2f}s")

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=not args.no_open)
    viz.loadViewerModel()

    z_offset = fk_world._world_T_base.translation

    # In IK mode we always render torso + both arms to avoid hidden defaults.
    display_keys = list(EEF_KEYS) if args.mode == "fk" else [
        CHAIN_ARM_LEFT,
        CHAIN_ARM_RIGHT,
        CHAIN_TORSO,
    ]

    for key in display_keys:
        viz.viewer[f"markers/fk/{key}"].set_object(
            meshcat_geom.Sphere(args.marker_radius),
            meshcat_geom.MeshLambertMaterial(
                color=MARKER_COLOR_FK,
                reflectivity=0.5,
                opacity=float(args.achieved_opacity),
                transparent=True,
            ),
        )
        viz.viewer[f"markers/gt/{key}"].set_object(
            meshcat_geom.Sphere(args.marker_radius * 0.6),
            meshcat_geom.MeshLambertMaterial(color=MARKER_COLOR_GT, reflectivity=0.5),
        )
        if args.show_error_lines:
            viz.viewer[f"markers/error/{key}"].set_object(
                meshcat_geom.Line(
                    meshcat_geom.PointsGeometry(np.zeros((3, 2), dtype=np.float64)),
                    meshcat_geom.LineBasicMaterial(color=0xF1C40F, linewidth=2),
                )
            )

    if args.mode == "fk":
        print("\n Green sphere  = AstribotFK predicted position")
        print(" Red sphere    = SDK ground-truth position (from HDF5)")
    else:
        print("\n Green sphere  = IK achieved position")
        print(" Red sphere    = IK target position (from HDF5)")
        print(
            " IK params     : "
            f"max_iters={args.ik_max_iters}, "
            f"tol_pos={args.ik_tol_pos}, "
            f"tol_rot={args.ik_tol_rot}, "
            f"damping={args.ik_damping}"
        )
    print(" Viewer URL    :", viz.viewer.url())
    print(" Press Ctrl-C to stop.\n")
    if sys.stdin.isatty():
        print(" Controls       : Enter = pause/resume, q + Enter = quit")

    frame_indices = np.arange(0, data.q_user.shape[0], max(args.step, 1), dtype=np.int64)
    dt = 1.0 / max(args.fps, 1e-3)

    ik_pos_err = {
        CHAIN_TORSO: np.empty(frame_indices.size, dtype=np.float64),
        CHAIN_ARM_LEFT: np.empty(frame_indices.size, dtype=np.float64),
        CHAIN_ARM_RIGHT: np.empty(frame_indices.size, dtype=np.float64),
    }
    ik_rot_err = {
        CHAIN_TORSO: np.empty(frame_indices.size, dtype=np.float64),
        CHAIN_ARM_LEFT: np.empty(frame_indices.size, dtype=np.float64),
        CHAIN_ARM_RIGHT: np.empty(frame_indices.size, dtype=np.float64),
    }
    ik_ok = np.zeros(frame_indices.size, dtype=bool)
    ik_iters = np.zeros(frame_indices.size, dtype=np.int64)
    fk_pos_err = {
        CHAIN_TORSO: np.empty(frame_indices.size, dtype=np.float64),
        CHAIN_ARM_LEFT: np.empty(frame_indices.size, dtype=np.float64),
        CHAIN_ARM_RIGHT: np.empty(frame_indices.size, dtype=np.float64),
    }
    fk_rot_err = {
        CHAIN_TORSO: np.empty(frame_indices.size, dtype=np.float64),
        CHAIN_ARM_LEFT: np.empty(frame_indices.size, dtype=np.float64),
        CHAIN_ARM_RIGHT: np.empty(frame_indices.size, dtype=np.float64),
    }
    q_init = data.q_user[frame_indices[0]].copy() if frame_indices.size > 0 else np.zeros(fk_world.nq)

    def _play_once() -> bool:
        """Play one pass. Return True if user requested quit."""
        nonlocal q_init
        paused = False
        for i, idx in enumerate(frame_indices):
            paused, should_quit = _poll_playback_control(paused)
            if should_quit:
                return True
            while paused:
                time.sleep(0.05)
                paused, should_quit = _poll_playback_control(paused)
                if should_quit:
                    return True

            q_user = data.q_user[idx]
            if args.mode == "fk":
                q_show = q_user
            else:
                target_left = data.gt[CHAIN_ARM_LEFT][idx]
                target_right = data.gt[CHAIN_ARM_RIGHT][idx]
                targets = {
                    CHAIN_ARM_LEFT: target_left,
                    CHAIN_ARM_RIGHT: target_right,
                }
                if args.ik_target_torso:
                    targets[CHAIN_TORSO] = data.gt[CHAIN_TORSO][idx]
                q_sol, ok, info = ik.solve(
                    targets,
                    q_init=q_init,
                    max_iters=args.ik_max_iters,
                    tol_pos=args.ik_tol_pos,
                    tol_rot=args.ik_tol_rot,
                    damping=args.ik_damping,
                )
                q_show = q_sol
                q_init = q_sol
                ik_ok[i] = bool(ok)
                ik_iters[i] = int(info.get("iterations", 0))

            q_pin = fk_urdf.to_pin_q(q_show)
            viz.display(q_pin)

            fk_poses_urdf = fk_urdf.forward(q_show, links=list(EEF_KEYS))
            fk_poses_world = fk_world.forward(q_show, links=list(EEF_KEYS))
            for key in display_keys:
                viz.viewer[f"markers/fk/{key}"].set_transform(
                    _marker_transform(fk_poses_urdf[key][:3])
                )
                gt_urdf = data.gt[key][idx, :3] - z_offset
                viz.viewer[f"markers/gt/{key}"].set_transform(_marker_transform(gt_urdf))

            if args.mode == "ik":
                ik_pos_err[CHAIN_TORSO][i] = position_error_mm(
                    fk_poses_world[CHAIN_TORSO], data.gt[CHAIN_TORSO][idx]
                )
                ik_pos_err[CHAIN_ARM_LEFT][i] = position_error_mm(
                    fk_poses_world[CHAIN_ARM_LEFT], data.gt[CHAIN_ARM_LEFT][idx]
                )
                ik_pos_err[CHAIN_ARM_RIGHT][i] = position_error_mm(
                    fk_poses_world[CHAIN_ARM_RIGHT], data.gt[CHAIN_ARM_RIGHT][idx]
                )
                ik_rot_err[CHAIN_TORSO][i] = quat_angle_deg(
                    fk_poses_world[CHAIN_TORSO], data.gt[CHAIN_TORSO][idx]
                )
                ik_rot_err[CHAIN_ARM_LEFT][i] = quat_angle_deg(
                    fk_poses_world[CHAIN_ARM_LEFT], data.gt[CHAIN_ARM_LEFT][idx]
                )
                ik_rot_err[CHAIN_ARM_RIGHT][i] = quat_angle_deg(
                    fk_poses_world[CHAIN_ARM_RIGHT], data.gt[CHAIN_ARM_RIGHT][idx]
                )
                if args.show_error_lines:
                    for key in (CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT):
                        achieved_urdf = fk_poses_urdf[key][:3]
                        target_urdf = data.gt[key][idx, :3] - z_offset
                        seg = np.stack([target_urdf, achieved_urdf], axis=1)
                        viz.viewer[f"markers/error/{key}"].set_object(
                            meshcat_geom.Line(
                                meshcat_geom.PointsGeometry(seg),
                                meshcat_geom.LineBasicMaterial(color=0xF1C40F, linewidth=2),
                            )
                        )
                if args.print_every > 0 and (i % args.print_every == 0):
                    print(
                        f"[ik frame {i:04d}/{frame_indices.size:04d}] "
                        f"T: {ik_pos_err[CHAIN_TORSO][i]:.2f} mm, {ik_rot_err[CHAIN_TORSO][i]:.3f} deg | "
                        f"L: {ik_pos_err[CHAIN_ARM_LEFT][i]:.2f} mm, {ik_rot_err[CHAIN_ARM_LEFT][i]:.3f} deg | "
                        f"R: {ik_pos_err[CHAIN_ARM_RIGHT][i]:.2f} mm, {ik_rot_err[CHAIN_ARM_RIGHT][i]:.3f} deg | "
                        f"iters={ik_iters[i]} ok={ik_ok[i]}"
                    )
            else:
                fk_pos_err[CHAIN_TORSO][i] = position_error_mm(
                    fk_poses_world[CHAIN_TORSO], data.gt[CHAIN_TORSO][idx]
                )
                fk_pos_err[CHAIN_ARM_LEFT][i] = position_error_mm(
                    fk_poses_world[CHAIN_ARM_LEFT], data.gt[CHAIN_ARM_LEFT][idx]
                )
                fk_pos_err[CHAIN_ARM_RIGHT][i] = position_error_mm(
                    fk_poses_world[CHAIN_ARM_RIGHT], data.gt[CHAIN_ARM_RIGHT][idx]
                )
                fk_rot_err[CHAIN_TORSO][i] = quat_angle_deg(
                    fk_poses_world[CHAIN_TORSO], data.gt[CHAIN_TORSO][idx]
                )
                fk_rot_err[CHAIN_ARM_LEFT][i] = quat_angle_deg(
                    fk_poses_world[CHAIN_ARM_LEFT], data.gt[CHAIN_ARM_LEFT][idx]
                )
                fk_rot_err[CHAIN_ARM_RIGHT][i] = quat_angle_deg(
                    fk_poses_world[CHAIN_ARM_RIGHT], data.gt[CHAIN_ARM_RIGHT][idx]
                )
                if args.print_every > 0 and (i % args.print_every == 0):
                    print(
                        f"[fk frame {i:04d}/{frame_indices.size:04d}] "
                        f"T: {fk_pos_err[CHAIN_TORSO][i]:.2f} mm, {fk_rot_err[CHAIN_TORSO][i]:.3f} deg | "
                        f"L: {fk_pos_err[CHAIN_ARM_LEFT][i]:.2f} mm, {fk_rot_err[CHAIN_ARM_LEFT][i]:.3f} deg | "
                        f"R: {fk_pos_err[CHAIN_ARM_RIGHT][i]:.2f} mm, {fk_rot_err[CHAIN_ARM_RIGHT][i]:.3f} deg"
                    )

            time.sleep(dt)
        return False

    try:
        q0 = np.zeros(fk_urdf.nq)
        viz.display(fk_urdf.to_pin_q(q0))
        # Browser-side mesh loading is asynchronous; the most reliable way to
        # avoid starting too early is an explicit user confirmation gate.
        if (not args.no_confirm_start) and sys.stdin.isatty():
            print(" Confirm the full robot mesh is visible, then press Enter to start playback...")
            input()
        while True:
            should_quit = _play_once()
            if should_quit:
                break
            if args.mode == "ik" and frame_indices.size > 0:
                _print_ik_summary(ik_pos_err, ik_rot_err, ik_ok, ik_iters)
            if args.mode == "fk" and frame_indices.size > 0:
                _print_fk_summary(fk_pos_err, fk_rot_err)
            if args.loop:
                continue
            if sys.stdin.isatty():
                print(" Playback finished. Press Enter to replay (Ctrl-C to exit)...")
                input()
                continue
            # Non-interactive session: avoid infinite loop.
            break
    except KeyboardInterrupt:
        print("\n[sim_meshcat_replay] interrupted by user.")

    print("[Done]")


if __name__ == "__main__":
    main()
