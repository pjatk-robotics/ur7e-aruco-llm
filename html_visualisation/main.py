from audio_server import AudioServer
from camera_aruco import ArucoCamera
from planner import Planner
from transforms import Transformations
from ur_robot import URRobot
from voice_select import VoiceContainerSelector


ROBOT_IP = "10.20.3.29"
CAM_ID = 0
MARKER_LENGTH = 0.028

GRIPPER_TCP = [0.0, 0.0, 0.18, 0.0, 0.0, 0.0]
GRIPPER_FORCE = 50  # percent, same scale as 009_gripper_check.py
GRIPPER_SPEED = 100  # percent, same scale as 009_gripper_check.py


def main():
    cam = None
    robot = None
    voice_selector = None
    audio_server = None

    try:
        audio_server = AudioServer()
        audio_server.start()
        voice_selector = VoiceContainerSelector()
        cam = ArucoCamera(cam_id=CAM_ID, marker_length=MARKER_LENGTH)
        tf = Transformations("handeye_result.npz")
        robot = URRobot(
            ROBOT_IP,
            tcp=GRIPPER_TCP,
            gripper_force=GRIPPER_FORCE,
            gripper_speed=GRIPPER_SPEED,
        )
        planner = Planner(cam, tf, robot, voice_selector=voice_selector)
        planner.run_forever()
    finally:
        if audio_server is not None:
            audio_server.stop()
        if cam is not None:
            cam.release()
        if robot is not None:
            robot.stop()
        if voice_selector is not None:
            voice_selector.cleanup()


if __name__ == "__main__":
    main()
