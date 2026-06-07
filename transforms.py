import cv2
import numpy as np


class Transformations:
    """
    Homogeneous transforms using the requested notation:

    - marker2cam maps marker coordinates into camera coordinates.
    - cam2flange maps camera coordinates into flange coordinates.
    - flange2base maps flange coordinates into base coordinates.
    - marker2base maps marker coordinates into base coordinates.
    """

    def __init__(self, path="handeye_result.npz"):
        data = np.load(path)
        self.T_cam2flange = np.asarray(data["T_cam2flange"], dtype=float)
        self.T_flange2cam = np.asarray(data["T_flange2cam"], dtype=float)

    @staticmethod
    def make_transform(R, t):
        T = np.eye(4)
        T[:3, :3] = np.asarray(R, dtype=float)
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
        return T

    @staticmethod
    def pose_to_transform(pose):
        R, _ = cv2.Rodrigues(np.asarray(pose[3:6], dtype=float))
        return Transformations.make_transform(R, pose[:3])

    @staticmethod
    def transform_to_pose(T):
        rvec, _ = cv2.Rodrigues(np.asarray(T[:3, :3], dtype=float))
        return [
            float(T[0, 3]),
            float(T[1, 3]),
            float(T[2, 3]),
            float(rvec[0, 0]),
            float(rvec[1, 0]),
            float(rvec[2, 0]),
        ]

    @staticmethod
    def transform_point(T, point):
        P = np.append(np.asarray(point, dtype=float).reshape(3), 1.0)
        return (T @ P)[:3]

    @staticmethod
    def pose_from_position_rotation(position, R_tcp2base):
        rvec, _ = cv2.Rodrigues(np.asarray(R_tcp2base, dtype=float))
        return [
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(rvec[0, 0]),
            float(rvec[1, 0]),
            float(rvec[2, 0]),
        ]

    def marker2base(self, T_flange2base, T_marker2cam):
        return T_flange2base @ self.T_cam2flange @ T_marker2cam

    def point_marker2base(self, T_flange2base, T_marker2cam, P_marker):
        T_marker2base = self.marker2base(T_flange2base, T_marker2cam)
        return self.transform_point(T_marker2base, P_marker), T_marker2base

    @staticmethod
    def tcp_down_base_rotation():
        # X_tcp = X_base, Y_tcp = -Y_base, Z_tcp = -Z_base.
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )

    @staticmethod
    def rotation_z(angle_rad):
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        return np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def tcp_opposite_marker_rotation(T_marker2base, yaw_marker_rad=0.0):
        # All three axes cannot be opposite in a valid right-handed rotation.
        # The yaw rotates the gripper in the marker plane while keeping the
        # approach axis opposite to marker Z.
        R_tcp2marker = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        R_tcp2marker = (
            Transformations.rotation_z(yaw_marker_rad)
            @ R_tcp2marker
        )
        R_marker2base = T_marker2base[:3, :3]
        return R_marker2base @ R_tcp2marker
