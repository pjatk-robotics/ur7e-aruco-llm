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
    @staticmethod
    def load_positions(json_path="robot_positions.json"):
        with open(json_path, "r") as f:
            return json.load(f)

    def __init__(self, cam, tf, robot, voice_selector=None):
        self.cam = cam
        self.tf = tf
        self.robot = robot
        self.voice_selector = voice_selector

        # ==========================================
        # LOAD SAFE JOINT POSITIONS
        # ==========================================
        positions = self.load_positions()

        self.home_joints = positions["home"]["joints_rad"]
        self.transfer_joints = positions["transfer"]["joints_rad"]

        # ==========================================
        # TASK CONFIGURATION
        # ==========================================
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

    # =========================================================
    # MAIN LOOP
    # =========================================================
    def run_forever(self):
        self.cam.start_preview()

        self.move_home()

        while True:
            marker_id = self.ask_marker_id()

            if marker_id is None:
                print("Exiting")
                return

            try:
                self.run_pick_place(marker_id)

            except RuntimeError as exc:
                print(f"Task skipped: {exc}")

            self.move_home()

    # =========================================================
    # USER INPUT
    # =========================================================
    def ask_marker_id(self):
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
        try:
            return self.grip_points_marker[marker_id]
        except KeyError as exc:
            raise RuntimeError(
                f"No grip point configured for marker {marker_id}"
            ) from exc

    # =========================================================
    # SAFE JOINT MOVES
    # =========================================================
    def move_home(self):
        print("Moving to HOME joints")
        self.robot.move_j(self.home_joints)

    def move_transfer(self):
        print("Moving to TRANSFER joints")
        self.robot.move_j(self.transfer_joints)

    # =========================================================
    # GRIPPER
    # =========================================================
    def open_gripper(self, reason):
        print(f"Opening gripper: {reason}")
        self.robot.open_gripper()

    def close_gripper(self, reason):
        print(f"Closing gripper: {reason}")
        self.robot.close_gripper()

    # =========================================================
    # AUDIO RESPONSES
    # =========================================================
    def play_transition_audio_response(self, marker_id):
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

    # =========================================================
    # PICK & PLACE
    # =========================================================
    def run_pick_place(self, marker_id):
        print(f"\nPreparing marker {marker_id}")

        # =====================================================
        # CALCULATE PICK POSITION
        # =====================================================
        pick_base, T_marker2base = (
            self.calculate_pick_point_base(marker_id)
        )

        # =====================================================
        # CENTER CAMERA FOR PRECISE RECALCULATION
        # =====================================================
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

        # =====================================================
        # RECALCULATE PRECISE PICK
        # =====================================================
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

        # =====================================================
        # APPROACH POSITION
        # =====================================================
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

        # =====================================================
        # OPEN GRIPPER
        # =====================================================
        self.open_gripper(
            "before moving down to grip point"
        )

        pick_pose = (
            Transformations.pose_from_position_rotation(
                pick_base,
                R_grasp_tcp2base,
            )
        )

        # =====================================================
        # MOVE TO PICK
        # =====================================================
        print(f"Moving opened TCP to grip point")
        print(f"Pick base position: {pick_base}")

        self.robot.move_l(pick_pose)

        saved_pick_pose = self.robot.get_tcp_pose()

        print(
            f"Saved pick TCP pose in base:\n{saved_pick_pose}"
        )

        # =====================================================
        # CLOSE GRIPPER
        # =====================================================
        self.close_gripper("at grip point")

        # =====================================================
        # LIFT OBJECT VERTICALLY
        # =====================================================
        lift_pose = list(saved_pick_pose)
        lift_pose[2] += self.approach_height

        print(
            f"Lifting object vertically by "
            f"{self.approach_height:.2f} m"
        )

        self.robot.move_l(lift_pose)

        # =====================================================
        # SAFE TRANSFER MOVE
        # =====================================================
        self.move_transfer()

        transfer_pose = self.robot.get_tcp_pose()

        print(
            f"Reached transfer TCP pose:\n{transfer_pose}"
        )

        self.play_transition_audio_response(marker_id)

        time.sleep(self.transfer_wait_s)

        # =====================================================
        # RETURN ABOVE PICK POSITION
        # =====================================================
        above_saved_pose = list(saved_pick_pose)
        above_saved_pose[2] += self.approach_height

        print("\nReturning above original pick position")
        self.robot.move_l(above_saved_pose)

        # =====================================================
        # MOVE DOWN
        # =====================================================
        print("Moving down to saved pick pose")
        self.robot.move_l(saved_pick_pose)

        # =====================================================
        # RELEASE
        # =====================================================
        self.open_gripper(
            "release at saved pick pose"
        )

        # =====================================================
        # RETREAT UP
        # =====================================================
        print("Retreating upward")

        retreat_pose = list(saved_pick_pose)
        retreat_pose[2] += self.approach_height

        self.robot.move_l(retreat_pose)

    # =========================================================
    # TRANSFORM COMPUTATION
    # =========================================================
    def calculate_pick_point_base(self, marker_id):
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
