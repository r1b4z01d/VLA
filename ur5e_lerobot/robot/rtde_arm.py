"""UR5e arm over RTDE (ur_rtde) — the teleop/record control path.

Project decision: RTDE for teleop + data collection (in-process, low latency, no rclpy); the
ROS 2 path (Ros2ArmInterface) is reserved for autonomous/mobile-base deployment. The SpaceMouse
EE-pose target is streamed to the UR controller via servoL, which runs IK on-board.

ur_rtde is lazy-imported in connect() so this module loads anywhere; only the Intel robot PC that
talks to the arm needs `pip install ur_rtde`.

UR's TCP pose is [x, y, z, Rx, Ry, Rz] with rotation as an axis-angle (rotvec) — the SAME layout
as this project's action/state EE pose, so poses pass through without conversion.

STREAMING (2026-10-24): send_ee_pose() no longer talks to the robot itself. It hands the clamped
target to _ServoStreamer, a thread that re-sends servoL at `stream_hz` with a short `servo_time`,
interpolating from the last sent pose toward the newest target over the measured hand-over period.
Why: ur_rtde's control script runs servoj in its own thread and only picks a new target up at each
servo_time boundary; with servo_time = 1/12 s and targets arriving every 110-125 ms (the teleop
loop's real period) the pick-up was quantised to 83/166 ms and the arm converged onto each stale
target before the next one arrived — measured as 7-8 commands/s and a hesitation every cycle. The
reachability gate (controller IK round trip) moved into the streamer too, so the caller's loop never
blocks on the robot. Every ur_rtde control call goes through self._lock: the interface is not
thread-safe.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import numpy as np

from . import ur_kin
from .arm_interface import ArmInterface
from .workspace import clamp_out_of_nogo, load_workspace_calib

DEFAULT_UR5E_IP = "192.168.11.21"
_Z_FLOOR_CALIB = load_workspace_calib().get("z_floor")  # calibrate_workspace.py; None if uncalibrated


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
                 max_step: float = 0.05, max_rot_step: float = 0.15, z_floor: "float | None" = _Z_FLOOR_CALIB,
                 ik_mode: str = "servoL", ik_damping: float = 0.08, ik_dq_max: float = 0.2,
                 stream_hz: float = 50.0, servo_time: float = 0.02):
        self.robot_ip = robot_ip
        self.speed, self.accel, self.dt = speed, accel, dt
        self.lookahead, self.gain, self.max_step = lookahead, gain, max_step
        self.max_rot_step = max_rot_step  # rad/command (~8.6° at 0.15); 0/None disables the orient clamp
        self.z_floor = z_floor            # min TCP z (m); None disables. Set to the measured table height.
        self.ik_mode = ik_mode            # 'servoL' (proven teleop) | 'dls' (singularity-robust, policy)
        self.ik_damping, self.ik_dq_max = ik_damping, ik_dq_max  # DLS damping + per-joint step cap
        self.stream_hz, self.servo_time = stream_hz, servo_time  # servoL re-send rate + servoj t (s)
        self._c = self._r = None
        self._connected = False
        self._freedrive = False
        self._lock = threading.RLock()  # ur_rtde control interface is not thread-safe
        self._stream: "_ServoStreamer | None" = None

    def _start_stream(self) -> None:
        self._stop_stream()
        self._stream = _ServoStreamer(self)
        self._stream.start()

    def _stop_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream = None

    def _pause_stream(self) -> None:
        if self._stream is not None:
            self._stream.pause()

    def stream_summary(self, reset: bool = False) -> str:
        """One line of streamer counters for the panel/eval log (slow cycles, IK/servo maxima)."""
        if self._stream is None:
            return "servo stream: off"
        s = self._stream.summary()
        if reset:
            self._stream.reset_counters()
        return s

    def connect(self) -> None:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        self._r = RTDEReceiveInterface(self.robot_ip)   # receive first: read-only, zero risk
        self._c = RTDEControlInterface(self.robot_ip)
        self._connected = True
        self._start_stream()

    def disconnect(self) -> None:
        self._stop_stream()
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
        self._pause_stream()         # the streamer must not servo while the arm is hand-guided
        with self._lock:
            try:
                self._c.servoStop()  # release any active servoL first, else teachMode can't take control
            except Exception:        # noqa: BLE001 — no-op if nothing was servoing
                pass                 # ("Another thread is already controlling the robot")
            self._c.teachMode()
        self._freedrive = True
        return True

    def stop_freedrive(self) -> None:
        if self._freedrive:
            with self._lock:
                self._c.endTeachMode()
                self._freedrive = False
                try:
                    self._c.servoStop()  # clean control state before the next servoL resumes teleop
                except Exception:        # noqa: BLE001
                    pass

    def home_step(self, q_home, max_dq: float = 0.03) -> bool:
        """servoJ ONE small step from the current joints toward q_home (rad) — call each control cycle
        (hold-to-home). Joint space, so no IK/singularity faults and it's perfectly repeatable. Returns
        True once within ~0.5° of home. `max_dq` caps per-cycle joint motion (speed)."""
        if self._freedrive or self._c is None or self._r is None:
            return False
        self._pause_stream()  # joint-space homing owns the servo until the next send_ee_pose
        q = np.asarray(self._r.getActualQ(), dtype=float)
        qh = np.asarray(q_home, dtype=float)
        dq = np.clip(qh - q, -max_dq, max_dq)
        with self._lock:
            self._c.servoJ((q + dq).tolist(), self.speed, self.accel, self.dt, self.lookahead, self.gain)
        return bool(np.all(np.abs(qh - q) < 0.009))

    def reconnect(self) -> str:
        """Recover after a protective stop / stopped control script (no app restart needed).

        Tears down the stale interfaces, clears a protective stop via the dashboard if present, then
        reconnects (which reuploads the RTDE control script). Returns a status message for the UI.
        """
        import time

        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        self._stop_stream()
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
        safety = self._r.getSafetyMode()  # 3=PROTECTIVE_STOP, 6/7=EMERGENCY_STOP, 8=VIOLATION, 9=FAULT
        note = ""
        if safety == 3:
            self._dashboard("close safety popup")
            self._dashboard("unlock protective stop")  # control can't start until this clears
            for _ in range(25):                         # PolyScope takes ~0.5-2 s to go back to NORMAL
                time.sleep(0.2)
                if self._r.getSafetyMode() != 3:
                    break
            note = "cleared protective stop; "
        elif safety in (6, 7):
            return "E-STOP engaged — release it on the robot, then hit Reconnect again."
        elif safety in (8, 9):
            # VIOLATION / FAULT: the arm powered itself off; the safety system must be restarted and the
            # brakes released again before a control script can run (ur_power.power_up does both).
            from .ur_power import power_up
            try:
                note = power_up(self.robot_ip) + "; "
            except Exception as e:  # noqa: BLE001
                return f"safety {'VIOLATION' if safety == 8 else 'FAULT'} — {e}"
            self._r.disconnect()
            self._r = RTDEReceiveInterface(self.robot_ip)
        try:
            self._c = RTDEControlInterface(self.robot_ip)
            self._connected = True
            self._start_stream()
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
            if self._c is not None and hasattr(self._c, "isProgramRunning"):
                with self._lock:
                    running = self._c.isProgramRunning()
                if not running:
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
            self._pause_stream()
            q_target = ur_kin.dls_ik_step(self._r.getActualQ(), cur, target,
                                          lam=self.ik_damping, dq_max=self.ik_dq_max)
            with self._lock:
                self._c.servoJ(q_target, self.speed, self.accel, self.dt, self.lookahead, self.gain)
        elif self._stream is not None:
            # Cartesian teleop/eval path: hand the target to the streamer (one controller IK per new
            # target, servoJ at stream_hz interpolated in joint space) — no robot round trip here.
            self._stream.set_target(target)
        else:  # not connected through connect() (tests / stubs): legacy synchronous path
            try:
                sol = self._c.getInverseKinematics(target, self._r.getActualQ())
            except Exception:  # noqa: BLE001 — ur_rtde raises when no solution
                sol = None
            if not sol:
                return list(cur)  # unreachable — hold
            self._c.servoL(target, self.speed, self.accel, self.dt, self.lookahead, self.gain)
        return list(self._r.getActualTCPPose())


def _lerp_pose(a, b, alpha: float) -> list:
    """Pose between a and b: linear in position, geodesic (rotvec) in orientation."""
    from scipy.spatial.transform import Rotation as _Rot
    pos = [a[i] + (b[i] - a[i]) * alpha for i in range(3)]
    ra = _Rot.from_rotvec(a[3:6])
    rel = (_Rot.from_rotvec(b[3:6]) * ra.inv()).as_rotvec()
    return pos + (_Rot.from_rotvec(rel * alpha) * ra).as_rotvec().tolist()


class _ServoStreamer(threading.Thread):
    """Streams servoJ at a fixed rate from its own thread, interpolating in joint space toward the
    newest target.

    The caller (teleop tick / eval step) hands over a Cartesian target every ~80-120 ms via
    set_target(). For each NEW target this thread asks the controller for ONE inverse-kinematics
    solution seeded with the current joints (the same call the old reachability gate made), then
    advances the sent joint vector linearly from where the previous segment got to toward that
    solution over the measured hand-over period, re-sending servoJ every 1/stream_hz s with a short
    servo_time. The control script therefore never runs IK in its hot loop (servoL made it run
    get_inverse_kin for every single command), and the controller always sees a moving target; it
    only holds if the next hand-over is late. An unreachable target, or a solution that jumps a joint
    by more than JOINT_JUMP_MAX (an IK branch flip), is dropped and the previous target held.

    pause() stops sending without a servoStop: the script keeps holding the last target, which is
    what hold-to-home (servoJ from the caller) and freedrive (servoStop + teachMode) need before
    they take over. The next set_target() resumes from the arm's actual joints.

    Instrumentation: every cycle is timed (lock wait, IK, servoJ); cycles over SLOW_CYCLE_MS are
    counted and reported at most every 2 s, and summary() gives the totals for the panel/eval log.
    """

    JOINT_JUMP_MAX = 0.5    # rad per hand-over; a Cartesian step <= max_step never needs more
    SLOW_CYCLE_MS = 100.0

    def __init__(self, arm: RtdeArmInterface):
        super().__init__(daemon=True, name="ur-servo-stream")
        self.arm = arm
        self._cv = threading.Condition()
        self._pending = None      # newest Cartesian target from the caller, not yet accepted
        self._goal_pose = None    # accepted target (Cartesian, for change detection)
        self._goal_q = None       # its joint solution
        self._from_q = None       # joints the current segment started from
        self._t_goal = 0.0
        self._period = float(arm.dt)   # EMA of hand-over intervals (s)
        self._t_accept = None
        self._sent_q = None
        self._paused = True
        self._stop_ev = threading.Event()
        self._last_warn = 0.0
        # counters for summary()
        self.sends = self.ik_calls = self.unreachable = self.rejected = self.errors = self.slow_cycles = 0
        self.max_cycle_ms = self.max_ik_ms = self.max_servo_ms = self.max_lock_ms = 0.0

    # --- caller side ----------------------------------------------------------------------------
    def set_target(self, pose) -> None:
        with self._cv:
            self._pending = [float(v) for v in pose]
            self._paused = False
            self._cv.notify()

    def pause(self) -> None:
        # Take the controller lock too: a cycle that is mid-send finishes first, and a cycle that has
        # not sent yet re-checks the flag under that same lock — so nothing goes out after pause()
        # returns (hold-to-home's servoJ / freedrive's servoStop must not interleave with a stale send).
        with self.arm._lock:
            with self._cv:
                self._paused = True
                self._pending = self._goal_pose = self._goal_q = self._from_q = self._sent_q = None

    def stop(self) -> None:
        self._stop_ev.set()
        with self._cv:
            self._cv.notify()
        if self.is_alive():
            self.join(timeout=2.0)

    def summary(self) -> str:
        return (f"servo stream: sends {self.sends}, ik {self.ik_calls}, unreachable {self.unreachable}, "
                f"rejected {self.rejected}, errors {self.errors}, slow cycles {self.slow_cycles} "
                f"(max cycle {self.max_cycle_ms:.0f} ms: lock {self.max_lock_ms:.0f}, ik {self.max_ik_ms:.0f}, "
                f"servo {self.max_servo_ms:.0f}), hand-over period {self._period * 1000:.0f} ms")

    def reset_counters(self) -> None:
        self.sends = self.ik_calls = self.unreachable = self.rejected = self.errors = self.slow_cycles = 0
        self.max_cycle_ms = self.max_ik_ms = self.max_servo_ms = self.max_lock_ms = 0.0

    # --- thread ----------------------------------------------------------------------------------
    def run(self) -> None:
        period = 1.0 / max(float(self.arm.stream_hz), 1.0)
        next_t = time.time()
        while not self._stop_ev.is_set():
            with self._cv:
                while self._paused and not self._stop_ev.is_set():
                    self._cv.wait(0.2)
                    next_t = time.time()
                if self._stop_ev.is_set():
                    break
                pending, self._pending = self._pending, None
            t0 = time.time()
            try:
                self._cycle(pending)
            except Exception as e:  # noqa: BLE001 — script stopped (protective stop) etc.; keep trying
                self.errors += 1
                if time.time() - self._last_warn > 2.0:
                    self._last_warn = time.time()
                    print(f"[servo] stream error: {e}", flush=True)
                time.sleep(0.2)
            cyc = (time.time() - t0) * 1000
            self.max_cycle_ms = max(self.max_cycle_ms, cyc)
            if cyc > self.SLOW_CYCLE_MS:
                self.slow_cycles += 1
                if time.time() - self._last_warn > 2.0:
                    self._last_warn = time.time()
                    print(f"[servo] slow cycle {cyc:.0f} ms (lock {self._t_lock:.0f} ik {self._t_ik:.0f} "
                          f"servo {self._t_servo:.0f} ms) — {self.summary()}", flush=True)
            next_t += period
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.time()

    def _cycle(self, pending) -> None:
        arm = self.arm
        now = time.time()
        self._t_lock = self._t_ik = self._t_servo = 0.0
        if pending is not None and (self._goal_pose is None
                                    or max(abs(pending[i] - self._goal_pose[i]) for i in range(6)) > 1e-6):
            t = time.time()
            with arm._lock:
                self._t_lock = (time.time() - t) * 1000
                t = time.time()
                try:
                    sol = arm._c.getInverseKinematics(pending, arm._r.getActualQ())
                except Exception:  # noqa: BLE001 — ur_rtde raises when no solution
                    sol = None
                self._t_ik = (time.time() - t) * 1000
            self.ik_calls += 1
            self.max_ik_ms = max(self.max_ik_ms, self._t_ik)
            if not sol or len(sol) != 6:
                self.unreachable += 1
                if now - self._last_warn > 2.0:
                    self._last_warn = now
                    print("[servo] target unreachable — holding the previous one", flush=True)
            else:
                q_start = self._sent_q if self._sent_q is not None else [float(v) for v in arm._r.getActualQ()]
                jump = max(abs(sol[j] - q_start[j]) for j in range(6))
                if jump > self.JOINT_JUMP_MAX:
                    self.rejected += 1
                    if now - self._last_warn > 2.0:
                        self._last_warn = now
                        print(f"[servo] IK solution jumps {jump:.2f} rad (branch flip?) — holding", flush=True)
                else:
                    if self._t_accept is not None:
                        iv = min(max(now - self._t_accept, 0.02), 0.3)
                        self._period = 0.7 * self._period + 0.3 * iv
                    self._t_accept = now
                    self._from_q, self._goal_q, self._goal_pose, self._t_goal = q_start, [float(v) for v in sol], pending, now
        if self._goal_q is None:
            return
        alpha = min(1.0, (now - self._t_goal) / max(self._period, 0.02))
        q = self._goal_q if alpha >= 1.0 else [a + (b - a) * alpha for a, b in zip(self._from_q, self._goal_q)]
        t = time.time()
        with arm._lock:
            self._t_lock += (time.time() - t) * 1000
            with self._cv:
                if self._paused:
                    return
            t = time.time()
            arm._c.servoJ(q, arm.speed, arm.accel, arm.servo_time, arm.lookahead, arm.gain)
            self._t_servo = (time.time() - t) * 1000
        self.max_servo_ms = max(self.max_servo_ms, self._t_servo)
        self.max_lock_ms = max(self.max_lock_ms, self._t_lock)
        self._sent_q = q
        self.sends += 1
