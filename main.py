from camera_aruco import ArucoCamera
from audio_server import AudioServer
from planner import Planner
from transforms import Transformations
from ur_robot import URRobot
from voice_select import VoiceContainerSelector


ROBOT_IP = "10.20.3.136"
CAM_ID = 0
MARKER_LENGTH = 0.028

GRIPPER_TCP = [0.0, 0.0, 0.18, 0.0, 0.0, 0.0]
ROBOT_VELOCITY = 0.80
ROBOT_ACCELERATION = 0.60
GRIPPER_FORCE = 50  # percent, same scale as 009_gripper_check.py
GRIPPER_SPEED = 80  # percent, same scale as 009_gripper_check.py


def main():
    audio_server = None
    cam = None
    robot = None
    voice_selector = None

    try:
        audio_server = AudioServer()
        if not audio_server.start():
            audio_server = None

        if audio_server is not None:
            audio_server.broadcast_status("starting", "Starting camera")
        cam = ArucoCamera(cam_id=CAM_ID, marker_length=MARKER_LENGTH)
        if audio_server is not None:
            audio_server.set_camera_frame_provider(cam.get_latest_frame_jpeg)
        cam.start_preview()

        if audio_server is not None:
            audio_server.broadcast_status("loading_voice", "Loading speech recognition")
        voice_selector = VoiceContainerSelector(audio_visualizer=audio_server)
        tf = Transformations("handeye_result.npz")
        if audio_server is not None:
            audio_server.broadcast_status("connecting_robot", "Connecting to robot")
        robot = URRobot(
            ROBOT_IP,
            tcp=GRIPPER_TCP,
            velocity=ROBOT_VELOCITY,
            acceleration=ROBOT_ACCELERATION,
            gripper_force=GRIPPER_FORCE,
            gripper_speed=GRIPPER_SPEED,
        )
        if audio_server is not None:
            audio_server.broadcast_status("idle", "Ready for a voice command")
        planner = Planner(cam, tf, robot, voice_selector=voice_selector)
        planner.run_forever()
    finally:
        if cam is not None:
            cam.release()
        if robot is not None:
            robot.stop()
        if voice_selector is not None:
            voice_selector.cleanup()
        if audio_server is not None:
            audio_server.stop()


if __name__ == "__main__":
    main()
