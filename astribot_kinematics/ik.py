"""Offline inverse kinematics for Astribot S1 based on pinocchio.

The solver follows a damped least-squares (Levenberg-Marquardt) CLIK scheme
operating on pinocchio's Lie-group integration:

    twist e = log6( oMf_current^{-1} * oMf_target )
    J      = 6 x nv Jacobian of the frame expressed in LOCAL
    v      = J^T (J J^T + lambda^2 I)^{-1} * e            (damped pseudo-inverse)
    q     <- integrate(q, v * step)
    clamp q within [lower, upper] if requested

Multiple frame targets are stacked into a single (6 m) x nv system, and only
the columns corresponding to the user-chosen *active chains* are kept - this
is how you choose whether to solve IK for the arm alone, arm + torso, or both
arms + torso. The other joints (plus any fixed ones) are held at ``q_init``.

Example
-------
>>> ik = AstribotIK(chains=["astribot_torso", "astribot_arm_left"])
>>> target_left = [0.30, 0.25, 1.10, 0.0, 0.0, 0.0, 1.0]  # xyz + quat
>>> q, ok, info = ik.solve({"astribot_arm_left": target_left})
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pinocchio is required for astribot_kinematics. Install with "
        "`pip install pin` or `conda install -c conda-forge pinocchio`."
    ) from exc

from .constants import (
    CHAIN_ARM_LEFT,
    CHAIN_ARM_RIGHT,
    CHAIN_JOINTS,
    CHAIN_TORSO,
)
from .fk import AstribotFK, ArrayLike, _pose_to_se3


class AstribotIK:
    """Whole-body CLIK solver for the Astribot S1.

    Parameters
    ----------
    fk:
        Optional existing :class:`AstribotFK` instance (reuses its pinocchio
        model). If ``None``, a new one is created with the default URDF.
    urdf_path:
        Forwarded to :class:`AstribotFK` when ``fk`` is not provided.
    chains:
        Active chain keys to optimise over. Order-insensitive. Defaults to the
        single left arm. Valid keys:
        ``"astribot_torso"``, ``"astribot_arm_left"``, ``"astribot_arm_right"``.
    """

    def __init__(
        self,
        fk: Optional[AstribotFK] = None,
        urdf_path: Optional[str] = None,
        chains: Optional[Iterable[str]] = None,
    ) -> None:
        self.fk: AstribotFK = fk if fk is not None else AstribotFK(urdf_path=urdf_path)

        chain_list = list(chains) if chains is not None else [CHAIN_ARM_LEFT]
        for c in chain_list:
            if c not in CHAIN_JOINTS:
                raise ValueError(
                    f"Unknown chain '{c}'. Expected subset of {list(CHAIN_JOINTS)}."
                )
        # Preserve a canonical ordering: torso, then left arm, then right arm.
        canonical = [CHAIN_TORSO, CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT]
        self.chains: List[str] = [c for c in canonical if c in chain_list]

        # Column indices in pinocchio's nv-wide Jacobian that belong to the
        # active joints.
        self._active_v_idx = np.concatenate(
            [self.fk.chain_v_indices(c) for c in self.chains]
        ).astype(np.int64)
        self._nv_active = self._active_v_idx.size

        # Pre-compute the (model.nv,) boolean mask for fast scatter-integration.
        self._active_mask = np.zeros(self.fk.model.nv, dtype=bool)
        self._active_mask[self._active_v_idx] = True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def active_chains(self) -> List[str]:
        return list(self.chains)

    @property
    def n_active_dofs(self) -> int:
        return int(self._nv_active)

    # ------------------------------------------------------------------
    # Main solver
    # ------------------------------------------------------------------
    def solve(
        self,
        targets: Dict[str, ArrayLike],
        q_init: Optional[ArrayLike] = None,
        *,
        max_iters: int = 200,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
        damping: float = 1e-4,
        step_size: float = 1.0,
        respect_limits: bool = True,
        position_only: bool = False,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, bool, Dict]:
        """Solve IK.

        Parameters
        ----------
        targets:
            Dict mapping a frame key (any key accepted by :meth:`AstribotFK.forward`)
            to a target pose, given either as ``(7,) [x,y,z,qx,qy,qz,qw]`` or
            ``(4, 4)`` homogeneous matrix.
        q_init:
            Initial 18-DoF user-layout joint vector. Defaults to zeros. Joints
            outside the active chains stay fixed at this value.
        max_iters:
            Maximum CLIK iterations.
        tol_pos, tol_rot:
            Convergence tolerances in metres and radians (applied to each
            target's ``log6`` error).
        damping:
            DLS damping ``lambda``. Larger = more robust near singularities at
            the cost of slower convergence.
        step_size:
            Line-search scalar for each iteration (1.0 is the DLS default).
        respect_limits:
            Clip the active joints to their URDF position limits after every
            step.
        position_only:
            If ``True``, drive the position residual only and ignore
            orientation (useful for 3-DoF reach evaluation).
        verbose:
            Print per-iteration diagnostics.

        Returns
        -------
        q_user, converged, info:
            ``q_user``: (18,) user-layout joint vector (always returned, even
            if IK did not converge - contains the best-effort solution).
            ``converged``: bool - whether all per-target tolerances were met.
            ``info``: dict with ``iterations``, ``per_target_errors``,
            ``task_size``, ``final_cost``.
        """
        if not targets:
            raise ValueError("`targets` must contain at least one frame key.")

        # Pull the FK base offset into an inverse transform so user-supplied
        # targets (in SDK "world" frame) are mapped back into the pure URDF
        # frame that the CLIK loop operates in.
        base_T_world = (
            self.fk._world_T_base.inverse() if self.fk._apply_base_offset else pin.SE3.Identity()
        )

        task_items: List[Tuple[int, pin.SE3]] = []
        task_labels: List[str] = []
        for key, pose in targets.items():
            target_world = _pose_to_se3(pose)
            target_urdf = base_T_world * target_world
            task_items.append((self.fk.frame_id(key), target_urdf))
            task_labels.append(key)

        task_rows = 3 if position_only else 6
        task_size = task_rows * len(task_items)
        if task_size > self._nv_active:
            if verbose:
                print(
                    f"[AstribotIK] Over-constrained: {task_size} task dims vs "
                    f"{self._nv_active} active DoFs. DLS will return a "
                    f"least-squares solution."
                )

        # Build initial q in pinocchio layout.
        if q_init is None:
            q_pin = pin.neutral(self.fk.model)
        else:
            q_pin = self.fk.to_pin_q(q_init)

        lower_user, upper_user = self.fk.joint_limits()
        user_to_v = self.fk.user_to_v_indices()

        errors_last = np.zeros(6)
        per_target_err = np.zeros(len(task_items))

        it = 0
        converged = False
        for it in range(1, max_iters + 1):
            pin.framesForwardKinematics(self.fk.model, self.fk.data, q_pin)
            pin.computeJointJacobians(self.fk.model, self.fk.data, q_pin)
            pin.updateFramePlacements(self.fk.model, self.fk.data)

            e_stack = np.zeros(task_size, dtype=np.float64)
            J_stack = np.zeros((task_size, self._nv_active), dtype=np.float64)

            per_tgt_pos = np.zeros(len(task_items))
            per_tgt_rot = np.zeros(len(task_items))

            for i, (fid, target) in enumerate(task_items):
                current: pin.SE3 = self.fk.data.oMf[fid]
                err6 = pin.log6(current.inverse() * target).vector  # 6-vec in LOCAL

                # Jacobian in LOCAL frame (matches log6 residual above).
                J_local = pin.getFrameJacobian(
                    self.fk.model, self.fk.data, fid, pin.ReferenceFrame.LOCAL
                )
                J_active = J_local[:, self._active_v_idx]

                if position_only:
                    e_stack[i * 3 : (i + 1) * 3] = err6[:3]
                    J_stack[i * 3 : (i + 1) * 3, :] = J_active[:3, :]
                else:
                    e_stack[i * 6 : (i + 1) * 6] = err6
                    J_stack[i * 6 : (i + 1) * 6, :] = J_active

                per_tgt_pos[i] = np.linalg.norm(err6[:3])
                per_tgt_rot[i] = np.linalg.norm(err6[3:])

            per_target_err = per_tgt_pos if position_only else (per_tgt_pos + per_tgt_rot)

            # Convergence check (each target independently).
            if position_only:
                conv = np.all(per_tgt_pos < tol_pos)
            else:
                conv = np.all((per_tgt_pos < tol_pos) & (per_tgt_rot < tol_rot))
            if conv:
                converged = True
                errors_last = e_stack
                if verbose:
                    print(f"[AstribotIK] converged in {it} iters")
                break

            # DLS solve: v_active = J^T (J J^T + lambda^2 I)^-1 e
            JJt = J_stack @ J_stack.T
            JJt_reg = JJt + (damping ** 2) * np.eye(task_size)
            try:
                rhs = np.linalg.solve(JJt_reg, e_stack)
            except np.linalg.LinAlgError:
                rhs = np.linalg.lstsq(JJt_reg, e_stack, rcond=None)[0]
            v_active = J_stack.T @ rhs * step_size

            # Scatter into a full-model velocity vector and integrate in Lie group.
            v_full = np.zeros(self.fk.model.nv)
            v_full[self._active_v_idx] = v_active
            q_pin = pin.integrate(self.fk.model, q_pin, v_full)

            if respect_limits:
                q_user = self.fk.from_pin_q(q_pin)
                # Only clamp active chains; inactive joints are untouched by v_full.
                active_user_mask = np.zeros(self.fk.nq, dtype=bool)
                for c in self.chains:
                    joint_names = CHAIN_JOINTS[c]
                    for jn in joint_names:
                        active_user_mask[self.fk.joint_names.index(jn)] = True
                q_user[active_user_mask] = np.clip(
                    q_user[active_user_mask],
                    lower_user[active_user_mask],
                    upper_user[active_user_mask],
                )
                q_pin = self.fk.to_pin_q(q_user)

            errors_last = e_stack
            if verbose:
                print(
                    f"[AstribotIK] iter {it:3d}  "
                    f"pos_err={per_tgt_pos.max():.4e} m  "
                    f"rot_err={per_tgt_rot.max():.4e} rad"
                )

        q_user = self.fk.from_pin_q(q_pin)
        info = {
            "iterations": it,
            "task_size": task_size,
            "per_target_errors": dict(zip(task_labels, per_target_err.tolist())),
            "final_cost": float(np.linalg.norm(errors_last)),
            "converged": converged,
            "active_chains": list(self.chains),
            "active_dofs": self._nv_active,
        }
        return q_user, converged, info

    # ------------------------------------------------------------------
    # Ergonomic one-liners
    # ------------------------------------------------------------------
    def solve_arm_left(
        self,
        target: ArrayLike,
        q_init: Optional[ArrayLike] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, bool, Dict]:
        """Shortcut: IK for ``astribot_arm_left_end_effector``."""
        return self.solve({CHAIN_ARM_LEFT: target}, q_init=q_init, **kwargs)

    def solve_arm_right(
        self,
        target: ArrayLike,
        q_init: Optional[ArrayLike] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, bool, Dict]:
        """Shortcut: IK for ``astribot_arm_right_end_effector``."""
        return self.solve({CHAIN_ARM_RIGHT: target}, q_init=q_init, **kwargs)

    def solve_bimanual(
        self,
        target_left: ArrayLike,
        target_right: ArrayLike,
        q_init: Optional[ArrayLike] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, bool, Dict]:
        """Shortcut: IK for both arms simultaneously (requires both arm chains active)."""
        missing = {CHAIN_ARM_LEFT, CHAIN_ARM_RIGHT} - set(self.chains)
        if missing:
            raise ValueError(
                f"solve_bimanual requires {sorted(missing)} in active chains; "
                f"got {self.chains}."
            )
        return self.solve(
            {CHAIN_ARM_LEFT: target_left, CHAIN_ARM_RIGHT: target_right},
            q_init=q_init,
            **kwargs,
        )
