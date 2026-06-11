"""Plan voice-selected ArUco pick, transfer, audio response, and return tasks."""

import json
import random
import time
import wave
from pathlib import Path

import numpy as np

from transforms import Transformations


AUDIO_RESPONSES_DIR = Path(__file__).with_name("audio_responses")
AUDIO_RESPONSE_VARIANTS = tuple(range(1, 6))
WAV_SAMPLE_WIDTH_DTYPES = {
    1: "uint8",
    2: "int16",
    3: "int24",
    4: "int32",
}


class Planner:
    """High-level coordinator for camera detection, robot motion, and gripper use."""

    @staticmethod
    def load_positions(json_path="robot_positions.json"):
        """Load named safe joint positions from JSON."""
        with open(json_path, "r") as f:
            return json.load(f)

    def __init__(self, cam, tf, robot, voice_selector=None):
        self.cam = cam
        self.tf = tf
        self.robot = robot
        self.voice_selector = voice_selector
        self.audio_visualizer = getattr(voice_selector, "audio_visualizer", None)
        self.robot_disabled_reason = None

        if self.robot is None:
            self.robot_disabled_reason = "robot is not connected"

        if self.robot is not None and self.tf is None:
            self.robot_disabled_reason = "hand-eye calibration is not available"
            self._stop_robot_safely()
            self.robot = None

        # Safe joint poses keep long moves predictable.
        self.home_joints = None
        self.transfer_joints = None
        if self.robot is not None:
            try:
                positions = self.load_positions()
                self.home_joints = positions["home"]["joints_rad"]
                self.transfer_joints = positions["transfer"]["joints_rad"]
            except Exception as exc:
                self.robot_disabled_reason = (
                    f"robot positions could not be loaded: {exc}"
                )
                self._stop_robot_safely()
                self.robot = None

        # Grip points are expressed in each marker's local coordinate frame.
        self.grip_points_marker = {
            0: np.array([0.035, 0.03, 0.005]),
            1: np.array([0.035, 0.03, 0.005]),
            2: np.array([0.035, 0.03, 0.005]),
            3: np.array([0.06, 0.03, -0.005]),
            4: np.array([0.06, 0.03, 0.00]),
            5: np.array([0.065, 0.03, 0.00]),
        }
        self.grasp_yaw_marker_rad = np.deg2rad(90.0)

        self.approach_height = 0.20
        self.transfer_wait_s = 3.0

        self.R_camera_down_base = (
            Transformations.tcp_down_base_rotation()
        )

    def run_forever(self):
        """Repeatedly ask for a marker ID and run the pick-place cycle."""
        self.cam.start_preview()

        if self.robot is None:
            self._announce_robot_disabled()
        else:
            try:
                self.move_home()
            except Exception as exc:
                self._disable_robot(f"initial HOME move failed: {exc}")

        while True:
            marker_id = self.ask_marker_id()

            if marker_id is None:
                print("Exiting")
                return

            if self.robot is None:
                self.skip_pick_place(marker_id)
                continue

            try:
                self.run_pick_place(marker_id)

            except Exception as exc:
                print(f"Task skipped: {exc}")
                self._disable_robot(f"robot task failed: {exc}")
                continue

            try:
                self.move_home()
            except Exception as exc:
                self._disable_robot(f"return HOME move failed: {exc}")

    def ask_marker_id(self):
        """Get the target marker from voice selection or terminal input."""
        detected = self.cam.get_detected_ids()

        if self.voice_selector is not None:
            marker_id = self.voice_selector.ask_container_id(detected_ids=detected)
            if marker_id is None:
                return None

            print(f"Planner selected marker/container ID: {marker_id}")
            return marker_id

        if detected:
            print(f"Detected marker IDs: {detected}")
        else:
            print("No marker detected yet")

        raw = input(
            "Enter ArUco marker ID to pick, or q to quit: "
        ).strip()

        if raw.lower() in {"q", "quit", "exit"}:
            return None

        try:
            return int(raw)

        except ValueError:
            print("Please enter a numeric marker ID")
        return self.ask_marker_id()

    def get_grip_point_marker(self, marker_id):
        """Return the configured grip point for one marker ID."""
        try:
            return self.grip_points_marker[marker_id]
        except KeyError as exc:
            raise RuntimeError(
                f"No grip point configured for marker {marker_id}"
            ) from exc

    def move_home(self):
        """Move to the taught HOME joint configuration."""
        if self.robot is None:
            return
        print("Moving to HOME joints")
        self.robot.move_j(self.home_joints)

    def move_transfer(self):
        """Move to the taught TRANSFER joint configuration."""
        if self.robot is None:
            return
        print("Moving to TRANSFER joints")
        self.robot.move_j(self.transfer_joints)

    def open_gripper(self, reason):
        """Open the gripper and print the current task reason."""
        if self.robot is None:
            return
        print(f"Opening gripper: {reason}")
        self.robot.open_gripper()

    def close_gripper(self, reason):
        """Close the gripper and print the current task reason."""
        if self.robot is None:
            return
        print(f"Closing gripper: {reason}")
        self.robot.close_gripper()

    def skip_pick_place(self, marker_id):
        """Keep voice/camera mode alive when robot motion is unavailable."""
        reason = self.robot_disabled_reason or "robot is not available"
        print(
            f"Robot unavailable ({reason}). "
            f"Selected marker/container ID={marker_id}; pick operation skipped."
        )
        self._publish_status(
            "idle",
            f"Robot unavailable. Selected ID={marker_id}; pick skipped.",
            selected_id=marker_id,
            detected_ids=self.cam.get_detected_ids(),
        )

    def _announce_robot_disabled(self):
        reason = self.robot_disabled_reason or "robot is not available"
        print(
            f"Robot unavailable ({reason}). "
            "Running camera/voice mode without robot motion."
        )

    def _disable_robot(self, reason):
        self._stop_robot_safely()
        self.robot_disabled_reason = reason
        self.robot = None
        print(
            f"WARNING: Robot disabled: {reason}. "
            "Camera/voice mode will continue."
        )
        self._publish_status(
            "idle",
            "Robot unavailable. Camera, Whisper and Ollama are still running.",
            detected_ids=self.cam.get_detected_ids(),
        )

    def _publish_status(self, phase, message, **payload):
        if self.audio_visualizer is not None:
            self.audio_visualizer.broadcast_status(phase, message, **payload)

    def _stop_robot_safely(self):
        if self.robot is None:
            return

        try:
            self.robot.stop()
        except Exception as exc:
            print(f"Robot stop warning: {exc}")

    def play_transition_audio_response(self, marker_id):
        """Play one random WAV response for the selected marker."""
        path = self.random_audio_response_path(marker_id)
        if path is None:
            print(
                f"WARNING: No audio response found for ID={marker_id} "
                f"in {AUDIO_RESPONSES_DIR}"
            )
            return

        print(f"Playing transition audio response: {path.name}")
        try:
            self.play_wav(path)
        except Exception as exc:
            print(f"WARNING: Could not play audio response {path.name}: {exc}")

    @staticmethod
    def random_audio_response_path(marker_id):
        """Return one existing response WAV for a marker, or None."""
        candidates = [
            AUDIO_RESPONSES_DIR / f"cat_{marker_id}_{variant}.wav"
            for variant in AUDIO_RESPONSE_VARIANTS
        ]
        existing = [path for path in candidates if path.exists()]
        if not existing:
            return None
        return random.choice(existing)

    @staticmethod
    def play_wav(path):
        """Play a WAV file through the default audio output device."""
        import sounddevice as sd

        with wave.open(str(path), "rb") as wav_file:
            sample_width = wav_file.getsampwidth()
            dtype = WAV_SAMPLE_WIDTH_DTYPES.get(sample_width)
            if dtype is None:
                raise ValueError(
                    f"unsupported WAV sample width: {sample_width} bytes"
                )

            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            data = wav_file.readframes(wav_file.getnframes())

        with sd.RawOutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
        ) as stream:
            stream.write(data)

    def run_pick_place(self, marker_id):
        """Pick the selected marker object, visit transfer, and put it back."""
        print(f"\nPreparing marker {marker_id}")

        # First estimate: enough to move the camera above the marker.
        pick_base, T_marker2base = (
            self.calculate_pick_point_base(marker_id)
        )

        # Center the camera for a cleaner second marker pose estimate.
        camera_center_pose, camera_base = (
            self.camera_center_pose_above_marker(
                T_marker2base
            )
        )

        print(
            f"Moving camera {self.approach_height:.2f} m above marker center"
        )
        print(f"Camera base position: {camera_base}")

        self.robot.move_l(camera_center_pose)

        # Second estimate: used for the actual pick pose.
        print("\nRecalculating marker pose near target")

        pick_base, T_marker2base = (
            self.calculate_pick_point_base(marker_id)
        )

        R_grasp_tcp2base = (
            Transformations.tcp_opposite_marker_rotation(
                T_marker2base,
                self.grasp_yaw_marker_rad,
            )
        )

        # Approach vertically above the configured grip point.
        approach_base = pick_base.copy()
        approach_base[2] += self.approach_height

        approach_pose = (
            Transformations.pose_from_position_rotation(
                approach_base,
                R_grasp_tcp2base,
            )
        )

        print(
            f"Moving {self.approach_height:.2f} m above marker grip point"
        )
        print(f"Approach base position: {approach_base}")

        self.robot.move_l(approach_pose)

        self.open_gripper(
            "before moving down to grip point"
        )

        pick_pose = (
            Transformations.pose_from_position_rotation(
                pick_base,
                R_grasp_tcp2base,
            )
        )

        print(f"Moving opened TCP to grip point")
        print(f"Pick base position: {pick_base}")

        self.robot.move_l(pick_pose)

        saved_pick_pose = self.robot.get_tcp_pose()

        print(
            f"Saved pick TCP pose in base:\n{saved_pick_pose}"
        )

        self.close_gripper("at grip point")

        # Lift straight up before any joint-space transfer move.
        lift_pose = list(saved_pick_pose)
        lift_pose[2] += self.approach_height

        print(
            f"Lifting object vertically by "
            f"{self.approach_height:.2f} m"
        )

        self.robot.move_l(lift_pose)

        # Use taught joints for the transfer pose instead of an ad hoc path.
        self.move_transfer()

        transfer_pose = self.robot.get_tcp_pose()

        print(
            f"Reached transfer TCP pose:\n{transfer_pose}"
        )

        self.play_transition_audio_response(marker_id)

        time.sleep(self.transfer_wait_s)

        # Return above the original pick point before descending.
        above_saved_pose = list(saved_pick_pose)
        above_saved_pose[2] += self.approach_height

        print("\nReturning above original pick position")
        self.robot.move_l(above_saved_pose)

        print("Moving down to saved pick pose")
        self.robot.move_l(saved_pick_pose)

        self.open_gripper(
            "release at saved pick pose"
        )

        print("Retreating upward")

        retreat_pose = list(saved_pick_pose)
        retreat_pose[2] += self.approach_height

        self.robot.move_l(retreat_pose)

    def calculate_pick_point_base(self, marker_id):
        """Calculate the marker grip point in robot base coordinates."""
        marker = self.cam.wait_for_marker(
            marker_id,
            timeout=10.0,
        )

        if marker is None:
            raise RuntimeError(
                f"Marker {marker_id} not visible"
            )

        T_marker2cam = marker["T_marker2cam"]
        grip_point_marker = self.get_grip_point_marker(marker_id)

        T_flange2base = (
            self.robot.get_flange2base()
        )

        pick_base, T_marker2base = (
            self.tf.point_marker2base(
                T_flange2base,
                T_marker2cam,
                grip_point_marker,
            )
        )

        print("\n=== marker -> camera -> flange -> base ===")

        print(f"marker id: {marker_id}")

        print(
            f"P_marker [m]: "
            f"{grip_point_marker}"
        )

        print(
            f"marker2cam translation [m]: "
            f"{T_marker2cam[:3, 3]}"
        )

        print(
            f"cam2flange translation [m]: "
            f"{self.tf.T_cam2flange[:3, 3]}"
        )

        print(
            f"flange2base translation [m]: "
            f"{T_flange2base[:3, 3]}"
        )

        print(
            f"marker2base translation [m]: "
            f"{T_marker2base[:3, 3]}"
        )

        print(
            f"pick point in base [m]: "
            f"{pick_base}"
        )

        return pick_base, T_marker2base

    def camera_center_pose_above_marker(self, T_marker2base):
        """Return a TCP pose that places the camera above the marker center."""
        marker_base = T_marker2base[:3, 3]

        camera_base = marker_base.copy()
        camera_base[2] += self.approach_height

        T_cam2base = Transformations.make_transform(
            self.R_camera_down_base,
            camera_base,
        )

        T_flange2base = (
            T_cam2base @ np.linalg.inv(self.tf.T_cam2flange)
        )
        T_tcp2base = (
            T_flange2base @ self.robot.get_tcp2flange()
        )

        return (
            Transformations.transform_to_pose(T_tcp2base),
            camera_base,
        )
