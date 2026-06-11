"""Run the full voice-guided ArUco pick-and-place application."""

import socket

from camera_aruco import ArucoCamera
from audio_server import AudioServer
from planner import Planner
from transforms import Transformations
from voice_select import VoiceContainerSelector


ROBOT_IP = "10.20.3.136"
CAM_ID = 1
CAMERA_BACKEND = "dshow"
MARKER_LENGTH = 0.028
ROBOT_RTDE_PORT = 30004
ROBOT_CONNECT_TIMEOUT = 1.5

GRIPPER_TCP = [0.0, 0.0, 0.18, 0.0, 0.0, 0.0]
ROBOT_VELOCITY = 0.80
ROBOT_ACCELERATION = 0.60
GRIPPER_FORCE = 50  # percent, same scale as 009_gripper_check.py
GRIPPER_SPEED = 80  # percent, same scale as 009_gripper_check.py


def broadcast_status(audio_server, phase, message, **payload):
    if audio_server is not None:
        audio_server.broadcast_status(phase, message, **payload)


def robot_endpoint_reachable(ip, port=ROBOT_RTDE_PORT, timeout=ROBOT_CONNECT_TIMEOUT):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def connect_robot_or_none(audio_server):
    broadcast_status(
        audio_server,
        "connecting_robot",
        "Checking robot connection",
    )

    if not robot_endpoint_reachable(ROBOT_IP):
        print(
            f"WARNING: Robot {ROBOT_IP}:{ROBOT_RTDE_PORT} is not reachable. "
            "Running camera/voice mode without robot motion."
        )
        broadcast_status(
            audio_server,
            "idle",
            "Robot unavailable. Camera, Whisper and Ollama are still running.",
        )
        return None, None

    robot = None
    try:
        from ur_robot import URRobot

        robot = URRobot(
            ROBOT_IP,
            tcp=GRIPPER_TCP,
            velocity=ROBOT_VELOCITY,
            acceleration=ROBOT_ACCELERATION,
            gripper_force=GRIPPER_FORCE,
            gripper_speed=GRIPPER_SPEED,
        )
        tf = Transformations("handeye_result.npz")
        return robot, tf
    except Exception as exc:
        print(
            "WARNING: Robot initialization failed. "
            f"Running camera/voice mode without robot motion: {exc}"
        )
        if robot is not None:
            robot.stop()
        broadcast_status(
            audio_server,
            "idle",
            "Robot initialization failed. Camera, Whisper and Ollama are still running.",
        )
        return None, None


def main():
    """Initialize all runtime services and keep the planner running."""
    audio_server = None
    cam = None
    robot = None
    voice_selector = None

    try:
        # Start the optional browser dashboard before long-running services.
        audio_server = AudioServer()
        if not audio_server.start():
            audio_server = None

        if audio_server is not None:
            audio_server.broadcast_status("starting", "Starting camera")
        cam = ArucoCamera(
            cam_id=CAM_ID,
            marker_length=MARKER_LENGTH,
            backend=CAMERA_BACKEND,
        )
        if audio_server is not None:
            audio_server.set_camera_frame_provider(cam.get_latest_frame_jpeg)
        cam.start_preview()

        if audio_server is not None:
            audio_server.broadcast_status("loading_voice", "Loading speech recognition")
        voice_selector = VoiceContainerSelector(audio_visualizer=audio_server)
        robot, tf = connect_robot_or_none(audio_server)
        broadcast_status(audio_server, "idle", "Ready for a voice command")
        planner = Planner(cam, tf, robot, voice_selector=voice_selector)
        planner.run_forever()
    finally:
        # Release hardware and background threads in reverse startup order.
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
