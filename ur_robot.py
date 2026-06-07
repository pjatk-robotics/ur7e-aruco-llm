import numpy as np
from robotiq_gripper_control import RobotiqGripper
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from transforms import Transformations


class URRobot:
    def __init__(
        self,
        ip,
        tcp=(0.0, 0.0, 0.25, 0.0, 0.0, 0.0),
        velocity=0.20,
        acceleration=0.20,
        gripper_force=50,
        gripper_speed=100,
        activate_gripper=True,
    ):
        self.ip = ip
        self.tcp = list(tcp)
        self.velocity = velocity
        self.acceleration = acceleration
        self.gripper_force = self._clamp_percent(gripper_force)
        self.gripper_speed = self._clamp_percent(gripper_speed)

        self.rtde_c = RTDEControlInterface(ip)
        self.rtde_r = RTDEReceiveInterface(ip)

        self.set_tcp(self.tcp)

        self.gripper = RobotiqGripper(self.rtde_c)
        if activate_gripper:
            self.activate_gripper()
        self.configure_gripper()

    def set_tcp(self, tcp):
        self.tcp = [float(value) for value in tcp]
        self.rtde_c.setTcp(self.tcp)
        print(f"TCP set to gripper: {self.tcp}")

    def stop(self):
        try:
            self.rtde_c.stopScript()
        except Exception as exc:
            print(f"Robot stop warning: {exc}")

    def get_tcp_pose(self):
        # RTDE returns active TCP -> base.
        return list(self.rtde_r.getActualTCPPose())

    def get_tcp2base(self):
        return Transformations.pose_to_transform(self.get_tcp_pose())

    def get_tcp2flange(self):
        # The configured TCP pose is expressed relative to the flange.
        return Transformations.pose_to_transform(self.tcp)

    def get_flange2base(self):
        T_tcp2base = self.get_tcp2base()
        T_tcp2flange = self.get_tcp2flange()
        return T_tcp2base @ np.linalg.inv(T_tcp2flange)

    def move_l(self, pose, velocity=None, acceleration=None):
        pose = self._validate_pose(pose)
        velocity = self.velocity if velocity is None else velocity
        acceleration = self.acceleration if acceleration is None else acceleration

        print(f"Moving TCP to: {pose}")
        ok = self.rtde_c.moveL(pose, velocity, acceleration)
        if ok is False:
            raise RuntimeError("Robot moveL failed or timed out")
        
    def move_j(self, joints, velocity=None, acceleration=None):
        if len(joints) != 6:
            raise ValueError("Joint vector must have 6 values")

        joints = [float(v) for v in joints]

        if not np.isfinite(joints).all():
            raise ValueError("Joint vector contains invalid values")

        velocity = self.velocity if velocity is None else velocity
        acceleration = self.acceleration if acceleration is None else acceleration

        print(f"Moving joints to: {joints}")

        ok = self.rtde_c.moveJ(
            joints,
            velocity,
            acceleration,
        )

        if ok is False:
            raise RuntimeError(
                "Robot moveJ failed or timed out"
            )

    def move_to_position_rotation(self, position, R_tcp2base, velocity=None, acceleration=None):
        pose = Transformations.pose_from_position_rotation(position, R_tcp2base)
        self.move_l(pose, velocity=velocity, acceleration=acceleration)
        return pose

    def open_gripper(self):
        self._run_gripper_command("Opening Robotiq Hand-E", self.gripper.open)

    def close_gripper(self):
        self._run_gripper_command("Closing Robotiq Hand-E", self.gripper.close)

    def activate_gripper(self):
        self._run_gripper_command("Activating Robotiq Hand-E", self.gripper.activate)

    def configure_gripper(self):
        self._run_gripper_command(
            f"Setting Robotiq force to {self.gripper_force}%",
            lambda: self.gripper.set_force(self.gripper_force),
        )
        self._run_gripper_command(
            f"Setting Robotiq speed to {self.gripper_speed}%",
            lambda: self.gripper.set_speed(self.gripper_speed),
        )

    def move_gripper_mm(self, position_mm):
        position_mm = max(0, int(position_mm))
        self._run_gripper_command(
            f"Moving Robotiq Hand-E to {position_mm} mm",
            lambda: self.gripper.move(position_mm),
        )

    @staticmethod
    def _run_gripper_command(label, command):
        print(label)
        ok = command()
        if not ok:
            raise RuntimeError(f"{label} failed or timed out")
        print(f"{label}: OK")

    @staticmethod
    def _clamp_percent(value):
        return int(max(0, min(100, value)))

    @staticmethod
    def _validate_pose(pose):
        if len(pose) != 6:
            raise ValueError("Pose must have 6 values")

        pose = [float(value) for value in pose]
        if not np.isfinite(pose).all():
            raise ValueError("Pose contains invalid values")

        return pose
