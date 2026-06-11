"""Detect ArUco markers from the calibrated camera and expose latest poses."""

import threading
import time

import cv2
import numpy as np

from transforms import Transformations


CAPTURE_BACKENDS = {
    "msmf": ("MSMF", cv2.CAP_MSMF),
    "dshow": ("DSHOW", cv2.CAP_DSHOW),
    "any": ("ANY", cv2.CAP_ANY),
}


class ArucoCamera:
    """Continuously captures camera frames and estimates marker-to-camera poses."""

    def __init__(
        self,
        cam_id=2,
        marker_length=0.032,
        max_distance=1.5,
        show_window=False,
        backend="dshow",
    ):
        self.backend_name, self.backend_id = self._resolve_backend(backend)

        # Match the capture settings used during camera calibration.
        self.cap = cv2.VideoCapture(cam_id, self.backend_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {cam_id} with backend {self.backend_name}"
            )

        self.marker_length = marker_length
        self.max_distance = max_distance
        self.show_window = show_window
        self.window_name = "ArUco camera"

        # Load intrinsic calibration produced by the ChArUco calibration script.
        self.mtx = np.load("camera_matrix.npy")
        self.dist = np.load("dist_coeffs.npy")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self.params = cv2.aruco.DetectorParameters_create()
        else:
            self.params = cv2.aruco.DetectorParameters()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(
                self.aruco_dict,
                self.params,
            )
        elif hasattr(cv2.aruco, "detectMarkers"):
            self.detector = None
        else:
            raise RuntimeError(
                "Installed OpenCV does not provide ArUco marker detection. "
                "Install an OpenCV build with the aruco module."
            )

        half_marker = self.marker_length / 2.0
        self._marker_object_points = np.array(
            [
                [-half_marker, half_marker, 0.0],
                [half_marker, half_marker, 0.0],
                [half_marker, -half_marker, 0.0],
                [-half_marker, -half_marker, 0.0],
            ],
            dtype=np.float32,
        )

        self._latest_markers = {}
        self._latest_frame_jpeg = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    @staticmethod
    def _resolve_backend(backend):
        if isinstance(backend, str):
            key = backend.lower()
            try:
                return CAPTURE_BACKENDS[key]
            except KeyError as exc:
                choices = ", ".join(sorted(CAPTURE_BACKENDS))
                raise ValueError(
                    f"Unknown camera backend {backend!r}. Use one of: {choices}"
                ) from exc

        return str(backend), int(backend)

    def start_preview(self):
        """Start the background capture and marker detection loop."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._thread.start()

    def stop_preview(self):
        """Stop background capture and close the optional OpenCV preview window."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self.show_window:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass

    def release(self):
        """Stop the preview thread and release the camera handle."""
        self.stop_preview()
        self.cap.release()

    def get_latest_markers(self):
        """Return copies of all currently visible marker pose records."""
        with self._lock:
            return {
                marker_id: self._copy_marker(marker)
                for marker_id, marker in self._latest_markers.items()
            }

    def get_latest_marker(self, marker_id):
        with self._lock:
            marker = self._latest_markers.get(marker_id)
            if marker is None:
                return None
            return self._copy_marker(marker)

    def get_detected_ids(self):
        return sorted(self.get_latest_markers().keys())

    def wait_for_marker(self, marker_id, timeout=None):
        """Wait until a marker appears, or return None after timeout."""
        start = time.time()
        while True:
            marker = self.get_latest_marker(marker_id)
            if marker is not None:
                return marker

            if timeout is not None and time.time() - start > timeout:
                return None

            time.sleep(0.05)

    def get_object_poses_cam(self):
        """Return marker pose vectors in camera coordinates for all detections."""
        markers = self.get_latest_markers()
        if not markers:
            return None
        return [
            (marker_id, marker["rvec_marker2cam"], marker["t_marker2cam"])
            for marker_id, marker in markers.items()
        ]

    def get_latest_frame_jpeg(self):
        """Return the latest annotated frame as JPEG bytes for streaming."""
        with self._lock:
            return self._latest_frame_jpeg

    def _preview_loop(self):
        """Capture frames, update marker state, and draw the debug overlay."""
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            markers, corners, ids = self._detect_markers(frame)
            with self._lock:
                self._latest_markers = markers

            self._draw_overlay(frame, markers, corners, ids)
            self._store_latest_frame_jpeg(frame)

            if self.show_window:
                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._running = False

    def _store_latest_frame_jpeg(self, frame):
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 82],
        )
        if not ok:
            return

        with self._lock:
            self._latest_frame_jpeg = encoded.tobytes()

    def _detect_markers(self, frame):
        """Detect markers and convert valid detections into transform records."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect_marker_corners(gray)

        if ids is None:
            return {}, corners, ids

        poses = self._estimate_marker_poses(corners)

        markers = {}
        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            rvec_marker2cam, t_marker2cam = poses[i]
            if rvec_marker2cam is None or t_marker2cam is None:
                continue

            if (
                not np.isfinite(t_marker2cam).all()
                or np.linalg.norm(t_marker2cam) > self.max_distance
            ):
                continue

            R_marker2cam, _ = cv2.Rodrigues(rvec_marker2cam)
            T_marker2cam = Transformations.make_transform(
                R_marker2cam,
                t_marker2cam,
            )

            markers[marker_id] = {
                "id": marker_id,
                "rvec_marker2cam": rvec_marker2cam.copy(),
                "t_marker2cam": t_marker2cam.copy(),
                "R_marker2cam": R_marker2cam.copy(),
                "T_marker2cam": T_marker2cam.copy(),
            }

        return markers, corners, ids

    def _detect_marker_corners(self, gray):
        if self.detector is not None:
            return self.detector.detectMarkers(gray)

        return cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.params
        )

    def _estimate_marker_poses(self, corners):
        if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.mtx, self.dist
            )
            return [
                (rvecs[i][0].copy(), tvecs[i][0].copy())
                for i in range(len(corners))
            ]

        return [self._estimate_marker_pose(corner) for corner in corners]

    def _estimate_marker_pose(self, corner):
        """Estimate one marker pose with solvePnP when OpenCV lacks the helper."""
        image_points = corner.reshape(-1, 2).astype(np.float32)
        flags = (
            cv2.SOLVEPNP_IPPE_SQUARE
            if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
            else cv2.SOLVEPNP_ITERATIVE
        )
        success, rvec, tvec = cv2.solvePnP(
            self._marker_object_points,
            image_points,
            self.mtx,
            self.dist,
            flags=flags,
        )

        if not success and flags != cv2.SOLVEPNP_ITERATIVE:
            success, rvec, tvec = cv2.solvePnP(
                self._marker_object_points,
                image_points,
                self.mtx,
                self.dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if not success:
            return None, None

        return rvec.reshape(3), tvec.reshape(3)

    def _draw_overlay(self, frame, markers, corners, ids):
        if ids is None or not markers:
            cv2.putText(
                frame,
                "No ArUco markers",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        row = 40
        for marker_id, marker in sorted(markers.items()):
            rvec = marker["rvec_marker2cam"]
            tvec = marker["t_marker2cam"]
            cv2.drawFrameAxes(frame, self.mtx, self.dist, rvec, tvec, 0.03)

            text = (
                f"ID {marker_id} marker2cam: "
                f"x={tvec[0]:.3f} y={tvec[1]:.3f} z={tvec[2]:.3f} m"
            )
            cv2.putText(
                frame,
                text,
                (20, row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            row += 26

            rtext = (
                f"rvec: [{rvec[0]:.2f}, {rvec[1]:.2f}, {rvec[2]:.2f}]"
            )
            cv2.putText(
                frame,
                rtext,
                (20, row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            row += 24

    def _copy_marker(self, marker):
        return {
            "id": marker["id"],
            "rvec_marker2cam": marker["rvec_marker2cam"].copy(),
            "t_marker2cam": marker["t_marker2cam"].copy(),
            "R_marker2cam": marker["R_marker2cam"].copy(),
            "T_marker2cam": marker["T_marker2cam"].copy(),
        }
