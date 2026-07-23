"""UR5e arm over RTDE (ur_rtde) — the teleop/record control path.

Project decision: RTDE for teleop + data collection (in-process, low latency, no rclpy); the
ROS 2 path (Ros2ArmInterface) is reserved for autonomous/mobile-base deployment. The SpaceMouse
EE-pose target is streamed to the UR controller via servoL, which runs IK on-board.

ur_rtde is lazy-imported in connect() so this module loads anywhere; only the Intel robot PC that
talks to the arm needs `pip install ur_rtde`.

UR's TCP pose is [x, y, z, Rx, Ry, Rz] with rotation as an axis-angle (rotvec) — the SAME layout
as this project's action/state EE pose, so poses pass through without conversion.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import ur_kin
from .arm_interface import ArmInterface
from .workspace import clamp_out_of_nogo

DEFAULT_UR5E_IP = "192.168.11.21"


class RtdeArmInterface(ArmInterface):
    """UR5e EE-pose control via RTDE. The EE target is converted to a JOINT target with a
    damped-least-squares step (ur_kin) and streamed via servoJ — avoiding the controller's
    get_inverse_kin, which faults on near-singular poses ("cannot reach the requested pose").

    speed/accel  — servoJ limits (unused by servoJ; kept for signature parity).
    dt           — control period; match the teleop loop rate (12 Hz -> 1/12 s).
    lookahead/gain — servoJ smoothing/stiffness (0.03-0.2 s, 100-2000).
    max_step     — hard cap (m) on per-command TCP translation; a glitchy/large target is
                   clamped toward the current pose so the arm can never jump. SAFETY-CRITICAL.
    max_rot_step — hard cap (rad) on per-command wrist reorientation; same idea for orientation
                   (a noisy/under-trained policy otherwise snaps the wrist). SAFETY-CRITICAL.
    z_floor      — min TCP z (m, base frame); commanded targets are lifted to this so the arm can't be
                   driven into the table. None disables it. SET TO THE MEASURED TABLE HEIGHT to enable.
    ik_mode      — 'servoL' (proven Cartesian teleop: controller IK + servoL) or 'dls' (Python
                   damped-least-squares + servoJ; singularity-robust, for the autonomous policy).
    ik_damping   — DLS damping λ (rad); higher = smoother/safer near singularities, less precise.
    ik_dq_max    — hard cap (rad) on per-command joint motion; final SAFETY bound on the servoJ step.
    """

    def __init__(self, robot_ip: str = DEFAULT_UR5E_IP, *, speed: float = 0.25, accel: float = 1.2,
                 dt: float = 1.0 / 12, lookahead: float = 0.1, gain: float = 300.0,
                 max_step: float = 0.05, max_rot_step: float = 0.15, z_floor: "float | None" = None,
                 ik_mode: str = "servoL", ik_damping: float = 0.08, ik_dq_max: float = 0.2):
        self.robot_ip = robot_ip
        self.speed, self.accel, self.dt = speed, accel, dt
        self.lookahead, self.gain, self.max_step = lookahead, gain, max_step
        self.max_rot_step = max_rot_step  # rad/command (~8.6° at 0.15); 0/None disables the orient clamp
        self.z_floor = z_floor            # min TCP z (m); None disables. Set to the measured table height.
        self.ik_mode = ik_mode            # 'servoL' (proven teleop) | 'dls' (singularity-robust, policy)
        self.ik_damping, self.ik_dq_max = ik_damping, ik_dq_max  # DLS damping + per-joint step cap
        self._c = self._r = None
        self._connected = False
        self._freedrive = False

    def connect(self) -> None:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        self._r = RTDEReceiveInterface(self.robot_ip)   # receive first: read-only, zero risk
        self._c = RTDEControlInterface(self.robot_ip)
        self._connected = True

    def disconnect(self) -> None:
        if self._c is not None:
            try:
                self.stop_freedrive()
                self._c.servoStop()
                self._c.stopScript()
            except Exception:  # noqa: BLE001
                pass
            self._c.disconnect()
        if self._r is not None:
            self._r.disconnect()
        self._connected = False

    def start_freedrive(self) -> bool:
        """Enable gravity-comp freedrive so the arm can be hand-guided (kinesthetic teaching)."""
        self._c.teachMode()
        self._freedrive = True
        return True

    def stop_freedrive(self) -> None:
        if self._freedrive:
            self._c.endTeachMode()
            self._freedrive = False

    def reconnect(self) -> str:
        """Recover after a protective stop / stopped control script (no app restart needed).

        Tears down the stale interfaces, clears a protective stop via the dashboard if present, then
        reconnects (which reuploads the RTDE control script). Returns a status message for the UI.
        """
        import time

        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        try:
            self.stop_freedrive()
        except Exception:  # noqa: BLE001
            pass
        for iface in (self._c, self._r):
            try:
                if iface is not None:
                    iface.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._c = self._r = None
        self._connected = False

        self._r = RTDEReceiveInterface(self.robot_ip)
        safety = self._r.getSafetyMode()  # 3=PROTECTIVE_STOP, 6/7=EMERGENCY_STOP
        note = ""
        if safety == 3:
            self._dashboard("unlock protective stop")  # control can't start until this clears
            time.sleep(0.6)
            note = "cleared protective stop; "
        elif safety in (6, 7):
            return "E-STOP engaged — release it on the robot, then hit Reconnect again."
        try:
            self._c = RTDEControlInterface(self.robot_ip)
            self._connected = True
            return note + "UR reconnected."
        except Exception as e:  # noqa: BLE001
            return note + f"control reconnect failed ({e}); check the pendant."

    def _dashboard(self, cmd: str) -> None:
        import socket
        import time
        try:
            d = socket.create_connection((self.robot_ip, 29999), timeout=3)
            d.recv(4096)
            d.sendall((cmd + "\n").encode())
            time.sleep(0.3)
            try:
                d.recv(4096)
            except Exception:  # noqa: BLE001
                pass
            d.close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    def is_ready(self) -> bool:
        """True if the arm can be servoed right now: not protective/emergency-stopped and the RTDE
        control script is running. Cheap RTDE reads — call each control step to catch a mid-run fault
        (a protective stop halts the control script, after which servoJ is a silent no-op)."""
        try:
            if self._r is not None and (self._r.isProtectiveStopped() or self._r.isEmergencyStopped()):
                return False
        except Exception:  # noqa: BLE001 — older ur_rtde may lack these; fall through to the script check
            pass
        try:
            if self._c is not None and hasattr(self._c, "isProgramRunning") and not self._c.isProgramRunning():
                return False
        except Exception:  # noqa: BLE001
            pass
        return True

    def get_ee_pose(self) -> list[float]:
        return list(self._r.getActualTCPPose())  # [x,y,z, Rx,Ry,Rz] rotvec — matches the schema

    def get_joint_positions(self) -> list[float]:
        return list(self._r.getActualQ())

    def send_ee_pose(self, pose: Sequence[float]) -> list[float]:
        if self._freedrive:  # arm is hand-guided in teach mode — never servo on top of that
            return list(self._r.getActualTCPPose())
        target, _ = clamp_out_of_nogo([float(v) for v in pose])  # keep out of the robot-body zone
        cur = self._r.getActualTCPPose()
        d = np.asarray(target[:3]) - np.asarray(cur[:3])
        dist = float(np.linalg.norm(d))
        if dist > self.max_step:  # clamp the per-step translation
            target[:3] = (np.asarray(cur[:3]) + d * (self.max_step / dist)).tolist()
        if self.max_rot_step:  # clamp the per-step wrist reorientation (bound the geodesic rotation)
            from scipy.spatial.transform import Rotation as _Rot
            _Rc = _Rot.from_rotvec(cur[3:6])
            _rel = (_Rot.from_rotvec(target[3:6]) * _Rc.inv()).as_rotvec()
            _ang = float(np.linalg.norm(_rel))
            if _ang > self.max_rot_step:  # slew toward the target orientation, capped
                target[3:6] = (_Rot.from_rotvec(_rel * (self.max_rot_step / _ang)) * _Rc).as_rotvec().tolist()
        if self.z_floor is not None and target[2] < self.z_floor:  # keep the TCP above the table
            target[2] = self.z_floor
        # Convert the EE-pose target to a JOINT target with a damped-least-squares step, then servoJ.
        # This never calls the controller's get_inverse_kin, so near-singular targets damp gracefully
        # instead of faulting ("cannot reach the requested pose"). Closes the loop on the ACTUAL pose
        # each cycle, so nominal-vs-calibrated DH error is corrected, not accumulated.
        if self.ik_mode == "dls":
            # Python damped-least-squares -> servoJ: never calls the controller's get_inverse_kin, so
            # near-singular targets damp gracefully instead of faulting. For the autonomous policy.
            q_target = ur_kin.dls_ik_step(self._r.getActualQ(), cur, target,
                                          lam=self.ik_damping, dq_max=self.ik_dq_max)
            self._c.servoJ(q_target, self.speed, self.accel, self.dt, self.lookahead, self.gain)
        else:
            # Proven Cartesian teleop path: controller IK reachability gate (hold if none) + servoL.
            try:
                sol = self._c.getInverseKinematics(target, self._r.getActualQ())
            except Exception:  # noqa: BLE001 — ur_rtde raises when no solution
                sol = None
            if not sol:
                return list(cur)  # unreachable — hold
            self._c.servoL(target, self.speed, self.accel, self.dt, self.lookahead, self.gain)
        return list(self._r.getActualTCPPose())
