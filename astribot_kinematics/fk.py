"""Offline forward kinematics for Astribot S1 based on pinocchio.

This module deliberately avoids any ROS / Astribot SDK runtime dependency: it
only needs the bundled URDF and pinocchio. The joint vector layout mirrors what
the Astribot SDK exposes:

    q = [torso_1..torso_4, arm_left_1..arm_left_7, arm_right_1..arm_right_7]  # 18

The same three chain keys ("astribot_torso", "astribot_arm_left",
"astribot_arm_right") used by the SDK's ``get_forward_kinematics`` are
recognised here so downstream code can swap between live-robot and offline FK
painlessly.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pinocchio is required for astribot_kinematics. Install with "
        "`pip install pin` or `conda install -c conda-forge pinocchio`."
    ) from exc

from .constants import (
    ARM_LEFT_JOINTS,
    ARM_RIGHT_JOINTS,
    CHAIN_ARM_LEFT,
    CHAIN_ARM_RIGHT,
    CHAIN_JOINTS,
    CHAIN_TORSO,
    EEF_LINKS,
    FULL_JOINT_ORDER,
    TORSO_JOINTS,
    WORLD_TO_TORSO_BASE_RPY,
    WORLD_TO_TORSO_BASE_XYZ,
    default_urdf_path,
)

ArrayLike = Union[np.ndarray, Sequence[float]]
QInput = Union[ArrayLike, Dict[str, ArrayLike]]
PoseFormat = str          # "xyzquat" or "matrix"
JacobianFrame = str       # "local", "world", or "local_world_aligned"

_JAC_REF = {
    "local": pin.ReferenceFrame.LOCAL,
    "world": pin.ReferenceFrame.WORLD,
    "local_world_aligned": pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
}


def _mat_to_xyzquat(T: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogeneous transform into [x, y, z, qx, qy, qz, qw]."""
    quat = pin.Quaternion(T[:3, :3])
    quat.normalize()
    return np.array(
        [T[0, 3], T[1, 3], T[2, 3], quat.x, quat.y, quat.z, quat.w],
        dtype=np.float64,
    )


def _pose_to_se3(pose: ArrayLike) -> pin.SE3:
    """Accept (4,4) matrix or (7,) xyzquat and return a pinocchio SE3."""
    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape == (4, 4):
        return pin.SE3(arr[:3, :3].copy(), arr[:3, 3].copy())
    if arr.shape == (7,):
        quat = pin.Quaternion(arr[6], arr[3], arr[4], arr[5])  # (w, x, y, z)
        quat.normalize()
        return pin.SE3(quat.toRotationMatrix(), arr[:3].copy())
    raise ValueError(
        f"Target pose must be shape (4,4) or (7,) xyzquat, got {arr.shape}."
    )


class AstribotFK:
    """Forward kinematics solver for the Astribot S1 (torso + dual arm).

    Parameters
    ----------
    urdf_path:
        Path to ``astribot_whole_body.urdf``. Defaults to the copy bundled
        inside the package.
    extra_frames:
        Optional mapping ``{key: link_name}`` to expose additional URDF frames
        alongside :data:`EEF_LINKS`.
    apply_world_base_offset:
        When ``True`` (default), poses are reported in the Astribot SDK's
        *world* frame, which applies a ``z += 0.097`` offset coming from
        ``astribot_torso.yaml`` (`weld_to_base_pose`). Disable to get poses in
        the pure URDF frame (``astribot_torso_base`` at the origin).
    world_base_xyz / world_base_rpy:
        Override the default ``weld_to_base_pose`` if needed.
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        extra_frames: Optional[Dict[str, str]] = None,
        apply_world_base_offset: bool = True,
        world_base_xyz: Optional[Sequence[float]] = None,
        world_base_rpy: Optional[Sequence[float]] = None,
    ) -> None:
        self.urdf_path: str = urdf_path or default_urdf_path()
        self._model: pin.Model = pin.buildModelFromUrdf(self.urdf_path)
        self._data: pin.Data = self._model.createData()

        # Pre-compute the constant world -> torso_base transform (SE3).
        xyz = np.asarray(world_base_xyz if world_base_xyz is not None
                         else WORLD_TO_TORSO_BASE_XYZ, dtype=np.float64)
        rpy = np.asarray(world_base_rpy if world_base_rpy is not None
                         else WORLD_TO_TORSO_BASE_RPY, dtype=np.float64)
        R = pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        self._world_T_base: pin.SE3 = pin.SE3(R, xyz.copy())
        self._apply_base_offset: bool = bool(apply_world_base_offset)

        # Cache per-joint indices in pinocchio's q/v vectors, keyed by joint
        # name, so the user-facing 18-DoF layout is decoupled from however
        # pinocchio happens to parse the URDF.
        self._idx_q: Dict[str, int] = {}
        self._idx_v: Dict[str, int] = {}
        for jname in FULL_JOINT_ORDER:
            jid = self._model.getJointId(jname)
            if jid == 0 or jid >= self._model.njoints:
                raise ValueError(
                    f"Joint '{jname}' not found in URDF: {self.urdf_path}"
                )
            joint = self._model.joints[jid]
            self._idx_q[jname] = joint.idx_q
            self._idx_v[jname] = joint.idx_v

        # Resolve URDF frames for every exposed link.
        frames: Dict[str, str] = dict(EEF_LINKS)
        if extra_frames:
            frames.update(extra_frames)
        self._frame_ids: Dict[str, int] = {}
        for key, link_name in frames.items():
            fid = self._model.getFrameId(link_name)
            if fid >= self._model.nframes:
                raise ValueError(
                    f"Link '{link_name}' not found in URDF: {self.urdf_path}"
                )
            self._frame_ids[key] = fid

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def model(self) -> pin.Model:
        return self._model

    @property
    def data(self) -> pin.Data:
        return self._data

    @property
    def nq(self) -> int:
        """Size of the user-facing joint vector (always 18 for S1)."""
        return len(FULL_JOINT_ORDER)

    @property
    def joint_names(self) -> List[str]:
        return list(FULL_JOINT_ORDER)

    @property
    def frame_keys(self) -> List[str]:
        return list(self._frame_ids.keys())

    def frame_id(self, key: str) -> int:
        """Return pinocchio's internal frame id for a user-facing key."""
        if key not in self._frame_ids:
            raise KeyError(f"Unknown frame key '{key}'. Available: {self.frame_keys}")
        return self._frame_ids[key]

    def joint_idx_v(self, joint_name: str) -> int:
        """Column index of a given joint within pinocchio's ``v`` / Jacobian."""
        if joint_name not in self._idx_v:
            raise KeyError(f"Unknown joint name: {joint_name}")
        return self._idx_v[joint_name]

    def chain_v_indices(self, chain_key: str) -> np.ndarray:
        """Jacobian column indices for a chain (``torso`` / ``arm_left`` / ``arm_right``)."""
        if chain_key not in CHAIN_JOINTS:
            raise KeyError(
                f"Unknown chain '{chain_key}'. Expected one of {list(CHAIN_JOINTS)}."
            )
        return np.array(
            [self._idx_v[jn] for jn in CHAIN_JOINTS[chain_key]], dtype=np.int64
        )

    def user_to_v_indices(self) -> np.ndarray:
        """Mapping from the 18-DoF user layout to pinocchio ``v`` indices."""
        return np.array([self._idx_v[jn] for jn in FULL_JOINT_ORDER], dtype=np.int64)

    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` position limits aligned with :attr:`joint_names`."""
        lower = np.empty(self.nq, dtype=np.float64)
        upper = np.empty(self.nq, dtype=np.float64)
        for i, jname in enumerate(FULL_JOINT_ORDER):
            iq = self._idx_q[jname]
            lower[i] = self._model.lowerPositionLimit[iq]
            upper[i] = self._model.upperPositionLimit[iq]
        return lower, upper

    # ------------------------------------------------------------------
    # q-vector helpers
    # ------------------------------------------------------------------
    def to_pin_q(self, q_input: QInput) -> np.ndarray:
        """Map a user 18-D vector / SDK-style dict into pinocchio's q layout."""
        q_pin = pin.neutral(self._model)

        if isinstance(q_input, dict):
            for chain_key, joint_names in CHAIN_JOINTS.items():
                if chain_key not in q_input:
                    continue
                values = np.asarray(q_input[chain_key], dtype=np.float64).ravel()
                if values.size != len(joint_names):
                    raise ValueError(
                        f"Chain '{chain_key}' expects {len(joint_names)} values, "
                        f"got {values.size}."
                    )
                for jname, v in zip(joint_names, values):
                    q_pin[self._idx_q[jname]] = v
            return q_pin

        q_arr = np.asarray(q_input, dtype=np.float64).ravel()
        if q_arr.size != self.nq:
            raise ValueError(
                f"Expected a flat q of length {self.nq} "
                f"(torso 4 + arm_left 7 + arm_right 7), got {q_arr.size}."
            )
        for jname, v in zip(FULL_JOINT_ORDER, q_arr):
            q_pin[self._idx_q[jname]] = v
        return q_pin

    def from_pin_q(self, q_pin: np.ndarray) -> np.ndarray:
        """Extract the 18-DoF user vector from a pinocchio ``q`` array."""
        return np.array(
            [q_pin[self._idx_q[jn]] for jn in FULL_JOINT_ORDER], dtype=np.float64
        )

    def build_q(
        self,
        torso: Optional[ArrayLike] = None,
        arm_left: Optional[ArrayLike] = None,
        arm_right: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Assemble a flat 18-DoF q vector from per-chain arrays."""
        q = np.zeros(self.nq, dtype=np.float64)
        slots: Dict[str, Tuple[int, int, Optional[ArrayLike]]] = {
            CHAIN_TORSO: (0, len(TORSO_JOINTS), torso),
            CHAIN_ARM_LEFT: (
                len(TORSO_JOINTS),
                len(TORSO_JOINTS) + len(ARM_LEFT_JOINTS),
                arm_left,
            ),
            CHAIN_ARM_RIGHT: (
                len(TORSO_JOINTS) + len(ARM_LEFT_JOINTS),
                self.nq,
                arm_right,
            ),
        }
        for _, (start, end, values) in slots.items():
            if values is None:
                continue
            arr = np.asarray(values, dtype=np.float64).ravel()
            if arr.size != end - start:
                raise ValueError(
                    f"Chain slice [{start}:{end}] expects {end - start} values, "
                    f"got {arr.size}."
                )
            q[start:end] = arr
        return q

    # ------------------------------------------------------------------
    # FK core
    # ------------------------------------------------------------------
    def forward(
        self,
        q: QInput,
        links: Optional[Iterable[str]] = None,
        format: PoseFormat = "xyzquat",
    ) -> Dict[str, np.ndarray]:
        """Single-sample forward kinematics.

        Parameters
        ----------
        q:
            Flat ``(18,)`` array following :attr:`joint_names`, or an SDK-style
            dict ``{"astribot_torso": [...], "astribot_arm_left": [...],
            "astribot_arm_right": [...]}`` (missing chains default to 0).
        links:
            Iterable of frame keys to return. Defaults to :attr:`frame_keys`.
        format:
            ``"xyzquat"`` (default) -> ``(7,)`` ``[x,y,z,qx,qy,qz,qw]``;
            ``"matrix"`` -> ``(4,4)`` homogeneous transforms.
        """
        if format not in ("xyzquat", "matrix"):
            raise ValueError("format must be 'xyzquat' or 'matrix'.")

        q_pin = self.to_pin_q(q)
        pin.framesForwardKinematics(self._model, self._data, q_pin)

        keys = list(links) if links is not None else list(self._frame_ids.keys())
        out: Dict[str, np.ndarray] = {}
        for key in keys:
            oMf = self._data.oMf[self.frame_id(key)]
            if self._apply_base_offset:
                oMf = self._world_T_base * oMf
            T = np.asarray(oMf.homogeneous)
            out[key] = _mat_to_xyzquat(T) if format == "xyzquat" else T.copy()
        return out

    def forward_batch(
        self,
        q_batch: ArrayLike,
        links: Optional[Iterable[str]] = None,
        format: PoseFormat = "xyzquat",
    ) -> Dict[str, np.ndarray]:
        """Vectorised FK over arbitrary leading batch dimensions.

        ``q_batch`` of shape ``(..., 18)`` yields ``(..., 7)`` (xyzquat) or
        ``(..., 4, 4)`` (matrix) per link.
        """
        q_arr = np.asarray(q_batch, dtype=np.float64)
        if q_arr.ndim == 0 or q_arr.shape[-1] != self.nq:
            raise ValueError(
                f"q_batch's last dim must equal {self.nq}, got shape {q_arr.shape}."
            )
        if q_arr.ndim == 1:
            return self.forward(q_arr, links=links, format=format)

        batch_shape = q_arr.shape[:-1]
        q_flat = q_arr.reshape(-1, self.nq)
        keys = list(links) if links is not None else list(self._frame_ids.keys())

        buffers: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
        for q in q_flat:
            step = self.forward(q, links=keys, format=format)
            for k in keys:
                buffers[k].append(step[k])

        per_sample_shape = (7,) if format == "xyzquat" else (4, 4)
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            stacked = np.stack(buffers[k], axis=0)
            out[k] = stacked.reshape(*batch_shape, *per_sample_shape)
        return out

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------
    def eef_left(self, q: QInput, format: PoseFormat = "xyzquat") -> np.ndarray:
        """Pose of ``astribot_arm_left_end_effector`` (SDK-equivalent)."""
        return self.forward(q, links=[CHAIN_ARM_LEFT], format=format)[CHAIN_ARM_LEFT]

    def eef_right(self, q: QInput, format: PoseFormat = "xyzquat") -> np.ndarray:
        """Pose of ``astribot_arm_right_end_effector`` (SDK-equivalent)."""
        return self.forward(q, links=[CHAIN_ARM_RIGHT], format=format)[CHAIN_ARM_RIGHT]

    def torso_end(self, q: QInput, format: PoseFormat = "xyzquat") -> np.ndarray:
        """Pose of ``astribot_torso_end_effector`` (SDK-equivalent)."""
        return self.forward(q, links=[CHAIN_TORSO], format=format)[CHAIN_TORSO]

    # ------------------------------------------------------------------
    # Jacobian & manipulability (operated-ability metrics)
    # ------------------------------------------------------------------
    def jacobian(
        self,
        q: QInput,
        link: str,
        reference: JacobianFrame = "local_world_aligned",
        columns: str = "user",
    ) -> np.ndarray:
        """6 x N Jacobian of a given frame.

        Parameters
        ----------
        q:
            Joint configuration (flat or dict).
        link:
            Frame key (see :attr:`frame_keys`).
        reference:
            Reference frame for the twist: ``"local"``, ``"world"``, or
            ``"local_world_aligned"`` (default, i.e. expressed at the frame's
            origin with world axes, which is the most intuitive for task-space
            control).
        columns:
            ``"user"`` (default) returns a ``(6, 18)`` matrix whose columns
            follow :attr:`joint_names`. ``"pinocchio"`` returns the raw
            ``(6, model.nv)`` Jacobian in pinocchio's layout.
        """
        if reference not in _JAC_REF:
            raise ValueError(
                f"reference must be one of {list(_JAC_REF)}, got '{reference}'."
            )
        if columns not in ("user", "pinocchio"):
            raise ValueError("columns must be 'user' or 'pinocchio'.")

        q_pin = self.to_pin_q(q)
        J = pin.computeFrameJacobian(
            self._model, self._data, q_pin, self.frame_id(link), _JAC_REF[reference]
        )
        if columns == "pinocchio":
            return np.asarray(J)
        return np.asarray(J[:, self.user_to_v_indices()])

    def manipulability(
        self,
        q: QInput,
        link: str,
        reference: JacobianFrame = "local_world_aligned",
        mode: str = "yoshikawa",
        translation_only: bool = False,
    ) -> float:
        """Scalar manipulability index for a frame.

        Parameters
        ----------
        mode:
            ``"yoshikawa"`` returns ``sqrt(det(J J^T))``;
            ``"inverse_condition"`` returns ``sigma_min / sigma_max`` of ``J``.
        translation_only:
            When ``True`` use only the first 3 rows (linear velocity) of the
            Jacobian, which is the relevant manipulability metric for pure
            positional reach evaluation.
        """
        J = self.jacobian(q, link=link, reference=reference, columns="user")
        if translation_only:
            J = J[:3]
        if mode == "yoshikawa":
            M = J @ J.T
            det = np.linalg.det(M)
            return float(np.sqrt(max(det, 0.0)))
        if mode == "inverse_condition":
            s = np.linalg.svd(J, compute_uv=False)
            if s.size == 0 or s[0] == 0:
                return 0.0
            return float(s[-1] / s[0])
        raise ValueError("mode must be 'yoshikawa' or 'inverse_condition'.")
