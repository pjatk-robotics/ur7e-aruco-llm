import time

from camera_aruco import ArucoCamera
from planner import Planner
from transforms import Transformations
from ur_robot import URRobot


ROBOT_IP = "10.20.3.136"
CAM_ID = 0
MARKER_LENGTH = 0.028

MARKER_IDS = range(6)
LIFT_PAUSE_S = 1.0

GRIPPER_TCP = [0.0, 0.0, 0.18, 0.0, 0.0, 0.0]
ROBOT_VELOCITY = 0.80
ROBOT_ACCELERATION = 0.80
GRIPPER_FORCE = 50
GRIPPER_SPEED = 100


class LiftOnlyPlanner(Planner):
    def run_marker_sequence(self, marker_ids):
        self.cam.start_preview()
        self.move_home()

        for marker_id in marker_ids:
            print(f"\n========== BOX / MARKER {marker_id} ==========")

            try:
                self.lift_marker_and_put_back(marker_id)
            except RuntimeError as exc:
                print(f"Marker {marker_id} skipped: {exc}")

            print("Moving HOME before next marker")
            self.move_home()

    def lift_marker_and_put_back(self, marker_id):
        print(f"Preparing marker {marker_id}")

        pick_base, T_marker2base = self.calculate_pick_point_base(marker_id)

        camera_center_pose, camera_base = (
            self.camera_center_pose_above_marker(T_marker2base)
        )

        print(
            f"Moving camera {self.approach_height:.2f} m above marker center"
        )
        print(f"Camera base position: {camera_base}")

        self.robot.move_l(camera_center_pose)

        print("\nRecalculating marker pose near target")

        pick_base, T_marker2base = self.calculate_pick_point_base(marker_id)

        R_grasp_tcp2base = Transformations.tcp_opposite_marker_rotation(
            T_marker2base,
            self.grasp_yaw_marker_rad,
        )

        approach_base = pick_base.copy()
        approach_base[2] += self.approach_height

        approach_pose = Transformations.pose_from_position_rotation(
            approach_base,
            R_grasp_tcp2base,
        )

        print(
            f"Moving {self.approach_height:.2f} m above marker grip point"
        )
        print(f"Approach base position: {approach_base}")

        self.robot.move_l(approach_pose)

        self.open_gripper("before moving down to grip point")

        pick_pose = Transformations.pose_from_position_rotation(
            pick_base,
            R_grasp_tcp2base,
        )

        print("Moving opened TCP to grip point")
        print(f"Pick base position: {pick_base}")

        self.robot.move_l(pick_pose)

        saved_pick_pose = self.robot.get_tcp_pose()
        print(f"Saved pick TCP pose in base:\n{saved_pick_pose}")

        self.close_gripper("at grip point")

        lift_pose = list(saved_pick_pose)
        lift_pose[2] += self.approach_height

        print(
            f"Lifting object vertically by {self.approach_height:.2f} m"
        )
        self.robot.move_l(lift_pose)
        time.sleep(LIFT_PAUSE_S)

        print("Putting object back to saved pick pose")
        self.robot.move_l(saved_pick_pose)

        self.open_gripper("release at saved pick pose")

        print("Retreating upward")
        self.robot.move_l(lift_pose)


def main():
    cam = None
    robot = None

    try:
        cam = ArucoCamera(cam_id=CAM_ID, marker_length=MARKER_LENGTH)
        tf = Transformations("handeye_result.npz")
        robot = URRobot(
            ROBOT_IP,
            tcp=GRIPPER_TCP,
            velocity=ROBOT_VELOCITY,
            acceleration=ROBOT_ACCELERATION,
            gripper_force=GRIPPER_FORCE,
            gripper_speed=GRIPPER_SPEED,
        )

        planner = LiftOnlyPlanner(cam, tf, robot)
        planner.run_marker_sequence(MARKER_IDS)

    finally:
        if cam is not None:
            cam.release()
        if robot is not None:
            robot.stop()


if __name__ == "__main__":
    main()
