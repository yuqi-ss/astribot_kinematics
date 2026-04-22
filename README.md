# astribot_kinematics

Offline **forward** and **inverse** kinematics for the Astribot **S1** humanoid
robot (torso + dual 7-DoF arm, 18 DoF in total). Self-contained, pure-Python,
no ROS / SDK runtime required — you only need `numpy` and `pinocchio`.

The bundled URDF is the official `astribot_whole_body.urdf` shipped with
`astribot_sdk`; the joint vector layout and frame keys mirror the SDK's
`Astribot.get_forward_kinematics` / `get_inverse_kinematics`, so downstream
code can switch between live-robot FK/IK and offline FK/IK without any
refactor.

Primary use cases:

- Converting recorded / generated joint trajectories to end-effector
  trajectories for **manipulation-ability evaluation** (reach, workspace
  coverage, manipulability ellipsoids, …).
- Solving IK as a pre-processing step for motion-planning / imitation
  datasets.
- Running entirely offline on a headless machine, on CPU, without any ROS
  master.

---

## Requirements

- Linux (tested on Ubuntu 20.04 / 22.04)
- Python ≥ 3.8
- [`pinocchio`](https://github.com/stack-of-tasks/pinocchio) ≥ 3.0
  (installed automatically by `install.sh`; package name on PyPI is `pin`,
  import name is `pinocchio`)
- `numpy`

---

## Installation

Requires Python >= 3.8. All dependencies are installed via pip.

```bash
cd astribot_kinematics
bash install.sh              # install all dependencies + package
bash install.sh --create-env # create new conda env 'kinematics' first
bash install.sh --sim        # also install meshcat + mujoco for sim tests
```

Or manual installation:

```bash
# 1. Core dependencies (numpy + pinocchio)
pip install "numpy>=1.20" "pin>=3.0"

# 2. Test dependencies
pip install "pytest>=7" h5py

# 3. (optional) Simulation backends
pip install "meshcat>=0.3" "mujoco>=3.0"

# 4. Install this package (editable mode)
pip install -e .
```

**Note**: The PyPI package name is `pin` (imported as `pinocchio`), not `pinocchio`.

Run the test suite to verify the install:

```bash
pytest -q tests        # 20 tests
pytest tests/ -v       # verbose output
```

---

## Package layout

```
astribot_kinematics/
├── astribot_kinematics/
│   ├── __init__.py           # public API
│   ├── constants.py          # joint names, frame keys, URDF locator
│   ├── fk.py                 # AstribotFK: forward kinematics + Jacobian
│   ├── ik.py                 # AstribotIK: damped least-squares CLIK
│   └── assets/
│       ├── astribot_whole_body.urdf  # bundled S1 URDF (18 DoF)
│       └── meshes/                   # bundled STL meshes (torso/head/arm)
├── tests/
│   ├── test_fk.py                    # pytest: FK smoke tests
│   ├── test_ik.py                    # pytest: IK smoke tests
│   ├── test_against_sdk_hdf5.py      # pytest: cross-check vs real SDK recording
│   ├── run_fk_hdf5.py                # standalone FK validation report (HDF5)
│   ├── run_ik_hdf5.py                # standalone IK validation report (HDF5)
│   └── simulation/
│       ├── sim_meshcat_replay.py     # Pinocchio + Meshcat browser replay
│       ├── sim_mujoco_fk.py          # MuJoCo vs AstribotFK cross-check
│       └── sim_mujoco_ik.py          # MuJoCo closed-loop IK verification
├── install.sh
├── pyproject.toml
└── README.md
```

The 18-DoF user joint vector is **always ordered**:

```
q = [ torso_1 .. torso_4 ,         # 4
      arm_left_1 .. arm_left_7 ,   # 7
      arm_right_1 .. arm_right_7 ] # 7
```

---

## Frame keys

| key                         | URDF link                            | SDK equivalent                         |
| --------------------------- | ------------------------------------ | -------------------------------------- |
| `astribot_torso`            | `astribot_torso_end_effector`        | `Astribot.torso_name`                  |
| `astribot_arm_left`         | `astribot_arm_left_end_effector`     | `Astribot.arm_left_name`               |
| `astribot_arm_right`        | `astribot_arm_right_end_effector`    | `Astribot.arm_right_name`              |
| `astribot_arm_left_tool`    | `astribot_arm_left_tool_link`        | wrist flange (before tool offset)      |
| `astribot_arm_right_tool`   | `astribot_arm_right_tool_link`       | wrist flange (before tool offset)      |

The SDK's `effector_to_tool_pose = [0, -0.15, 0]` is already baked into the
URDF's `astribot_arm_*_ee_joint`, so the `astribot_arm_*` keys return poses
that match the SDK's `get_forward_kinematics` output bit-for-bit.

---

## Forward kinematics

```python
import numpy as np
from astribot_kinematics import AstribotFK

fk = AstribotFK()

# --- single-sample, per-chain dict input --------------------------------
q = fk.build_q(
    torso    =[0.275, -0.55, 0.275, 0.0],
    arm_left =[-0.09622, -0.4218, -1.1273, 1.6168, -0.4149, 0.0645, 0.4225],
    arm_right=[ 0.09622, -0.4218,  1.1273, 1.6168,  0.4149, 0.0645,-0.4225],
)

print(fk.eef_left(q))       # (7,)  [x, y, z, qx, qy, qz, qw]
print(fk.eef_right(q, format="matrix"))  # (4, 4)
print(fk.torso_end(q))

# --- batched FK over any leading shape (..., 18) ------------------------
q_batch = np.random.randn(1000, 100, 18) * 0.1   # 1000 trajectories × 100 frames
poses = fk.forward_batch(q_batch)
left_traj = poses["astribot_arm_left"]           # (1000, 100, 7)
```

### Jacobian & manipulability

```python
# 6 x 18 Jacobian in the user joint order
J = fk.jacobian(q, link="astribot_arm_left", reference="local_world_aligned")

# Yoshikawa manipulability index (translation-only) — a scalar proxy for
# how "dexterous" the configuration is.
m = fk.manipulability(q, link="astribot_arm_left", translation_only=True)
```

---

## Inverse kinematics

`AstribotIK` is a damped least-squares CLIK solver that operates on
pinocchio's Lie-group integration. You explicitly choose which chains
participate (torso / left arm / right arm); the remaining joints stay
fixed at `q_init`, and limits from the URDF are enforced at every step.

```python
from astribot_kinematics import AstribotFK, AstribotIK

fk = AstribotFK()
ik = AstribotIK(fk=fk, chains=["astribot_torso", "astribot_arm_left"])

target = [0.35, 0.25, 1.10, 0.0, 0.0, 0.0, 1.0]   # xyz + quat (x,y,z,w)

q_sol, ok, info = ik.solve(
    targets   ={"astribot_arm_left": target},
    q_init    =None,                              # None -> zeros
    max_iters =200,
    tol_pos   =1e-4,     # m
    tol_rot   =1e-3,     # rad
    damping   =1e-4,
    respect_limits=True,
)
print("converged:", ok, "iters:", info["iterations"])
print("achieved pose:", fk.eef_left(q_sol))
```

Convenience helpers:

```python
ik.solve_arm_left(target)                                       # single arm
ik.solve_arm_right(target)                                      # single arm
ik.solve_bimanual(target_left, target_right, q_init=q0)         # both arms
ik.solve({"astribot_arm_left": t_l,
          "astribot_arm_right": t_r}, position_only=True)       # 3-DoF task
```

Parameters of note:

| argument          | meaning                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| `chains`          | which chains the solver may move; others stay fixed at `q_init`         |
| `position_only`   | drive (x, y, z) only, ignore orientation                                |
| `respect_limits`  | clip active joints to the URDF's `lower/upper_position_limit` each step |
| `damping`         | Levenberg-Marquardt λ — increase near singularities                     |
| `step_size`       | multiplier on the Newton step (default 1.0)                             |

---

## A realistic example: manipulation-ability metric

Compute a per-frame "reach index" (distance from the left EEF to the torso
end-effector) over a batch of motion trajectories:

```python
import numpy as np
from astribot_kinematics import AstribotFK

fk = AstribotFK()
q_batch = np.load("trajectories.npy")             # (N, T, 18)

poses = fk.forward_batch(q_batch)
torso = poses["astribot_torso"][..., :3]
left  = poses["astribot_arm_left"][..., :3]

reach = np.linalg.norm(left - torso, axis=-1)     # (N, T)
print("mean reach:", reach.mean(),
      "coverage  :", reach.std())
```

Combine with `fk.manipulability(..., translation_only=True)` for a more
principled dexterity score.

---

## API reference (summary)

### `AstribotFK`

| method                             | signature                                                         | purpose                                                        |
| ---------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------- |
| `AstribotFK(urdf_path=None, extra_frames=None)` |                                                         | construct solver (URDF defaults to bundled copy)               |
| `forward(q, links=None, format="xyzquat")`      | `q: (18,)` or dict → `{link: (7,) or (4,4)}`             | single-sample FK                                               |
| `forward_batch(q_batch, links=None, format=…)`  | `q: (..., 18)` → `{link: (..., 7)}`                      | vectorised FK                                                  |
| `eef_left / eef_right / torso_end`              | `q → (7,) or (4,4)`                                       | per-end-effector shortcuts                                     |
| `jacobian(q, link, reference, columns)`         | → `(6, 18)` or `(6, model.nv)`                            | analytical Jacobian                                            |
| `manipulability(q, link, mode, translation_only)` | → `float`                                              | Yoshikawa or inverse-condition number                          |
| `build_q(torso=…, arm_left=…, arm_right=…)`     | → `(18,)`                                                 | assemble user-layout q from chain arrays                       |
| `joint_limits()`                                | → `(lower(18,), upper(18,))`                              | URDF position limits                                           |

### `AstribotIK`

| method                                          | purpose                                                |
| ----------------------------------------------- | ------------------------------------------------------ |
| `AstribotIK(fk=None, urdf_path=None, chains=…)` | damped least-squares CLIK solver                       |
| `solve(targets, q_init, …)`                     | multi-frame IK, returns `(q_user, converged, info)`    |
| `solve_arm_left / solve_arm_right`              | single-arm shortcut                                    |
| `solve_bimanual(target_left, target_right, …)`  | both arms at once (chains must include both arms)      |

---

## Validating against the live SDK (bit-level cross-check)

This module reports poses in the SDK's **world frame**, i.e. with the
`weld_to_base_pose = [0, 0, 0.097, 0, 0, 0]` offset from
`astribot_torso.yaml` pre-applied. Disable via
`AstribotFK(apply_world_base_offset=False)` to get pure URDF-frame output.

Cross-checked against a real SDK recording
(`0710_Microwave_S8_episode_0.hdf5`, 1831 frames, dual-arm manipulation
task). Comparing `command_poses_dict/astribot_*` (SDK's online FK output)
against this package's offline FK of `joints_dict/joints_position_command`:

| frame set                 | torso end    | arm_left     | arm_right    |
| ------------------------- | -----------: | -----------: | -----------: |
| quasi-static (\|v\| < 0.1 rad/s) mean | 0.08 mm | 0.15 mm | 0.17 mm |
| quasi-static max          | 0.65 mm     | 1.4 mm       | 0.8 mm       |
| all frames mean           | 0.08 mm     | 0.32 mm      | 1.05 mm      |
| all frames max            | 0.65 mm     | 3.1 mm       | 7.0 mm       |

Key observation: on quasi-static frames the residual is at IEEE-754
rounding level (sub-mm). On fast frames the error correlates with joint
speed (Pearson r ≈ +0.54 for the left arm, +0.63 for the right arm),
confirming the residual comes from the dataset's own
command-vs-pose timestamp jitter, **not** from our FK implementation.

The check is automated in `tests/test_against_sdk_hdf5.py`; point it to
your own recording through `ASTRIBOT_HDF5=/path/to/file.hdf5 pytest -q`.

For a more comprehensive, human-readable validation report (position /
orientation accuracy, frame & tool-frame consistency, Jacobian correctness,
solver success rate, constraint satisfaction, and timing statistics), run
the standalone scripts:

```bash
python tests/run_fk_hdf5.py      # FK metrics (5 blocks)
python tests/run_ik_hdf5.py      # IK metrics (5 blocks)

# optional: stricter / fuller sweep
python tests/run_fk_hdf5.py --step 1 --jacobian-samples 50 --bench-samples 1000
python tests/run_ik_hdf5.py --step 1 --tol-pos 1e-4 --tol-rot 1e-3 \
                            --pass-pos-mm 2.0 --pass-rot-deg 0.5
```

Both scripts default to `tests/data/0710_Microwave_S8_episode_0.hdf5`; pass
`--hdf5 /path/to/file.hdf5` to target any other recording.

---

## Simulation-based validation (Meshcat + MuJoCo)

The repo also ships three **standalone simulation scripts** under
`tests/simulation/` that work out-of-the-box (the required STL meshes are
bundled under `astribot_kinematics/assets/meshes/`, so you do **not** need
`astribot_sdk` installed).

Install the optional backends first:

```bash
bash install.sh --sim                            # meshcat + mujoco
# or: pip install "meshcat>=0.3" "mujoco>=3.0"
```

| Script                              | What it does                                                                 |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| `sim_meshcat_replay.py`             | Pinocchio + Meshcat browser replay in two modes: `--mode fk` (FK vs SDK)      |
|                                     | and `--mode ik` (IK achieved vs target, plus convergence stats)               |
| `sim_mujoco_fk.py`                  | Loads the same URDF into MuJoCo, compares link poses vs `AstribotFK`         |
| `sim_mujoco_ik.py`                  | Closed-loop: `AstribotIK` solves, MuJoCo reads back achieved EEF pose        |

```bash
python tests/simulation/sim_meshcat_replay.py --mode fk
python tests/simulation/sim_meshcat_replay.py --mode ik
python tests/simulation/sim_mujoco_fk.py
python tests/simulation/sim_mujoco_ik.py

# all scripts accept --hdf5, --step, and --meshes-dir (defaults to the
# bundled meshes under astribot_kinematics/assets/meshes/).
# Meshcat replay waits for an Enter confirmation by default after the
# robot is first rendered (to ensure meshes are fully visible) - use
# --no-confirm-start to skip that gate.
# Defaults: --mode ik --step 1 --fps 20 --ik-target-torso true
```

The MuJoCo scripts print the same structured statistics tables as the other
validation scripts (position / orientation error, success rate, timing).

---

## License

BSD 3-Clause (matches the upstream `astribot_sdk` URDF license).
