import os

os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")

import argparse
import subprocess
import time

import cv2


WINDOW_NAME = "Orbec camera ID finder"

BACKENDS = {
    "msmf": ("MSMF", cv2.CAP_MSMF),
    "dshow": ("DSHOW", cv2.CAP_DSHOW),
    "any": ("ANY", cv2.CAP_ANY),
}


def fourcc_to_text(value):
    try:
        code = int(value)
    except (TypeError, ValueError):
        return "unknown"

    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    if all(32 <= ord(char) <= 126 for char in chars):
        return "".join(chars)
    return "unknown"


def print_windows_camera_devices():
    if os.name != "nt":
        return

    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            "$devices = Get-PnpDevice -PresentOnly | "
            "Where-Object { "
            "($_.Class -in @('Camera', 'Image')) -or "
            "($_.FriendlyName -match 'Orbbec|Orbec|Astra|Gemini|Femto') "
            "}; "
            "$devices | Select-Object FriendlyName,Class,InstanceId | "
            "Format-Table -AutoSize | Out-String -Width 240"
        ),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return

    output = result.stdout.strip()
    if output:
        print("\nWindows USB/camera devices:")
        print(output)
        print(
            "Note: Windows device names do not always map directly to OpenCV IDs."
        )


def open_capture(camera_id, backend_id, width, height):
    cap = cv2.VideoCapture(camera_id, backend_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_first_frame(cap, attempts=15):
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return frame
        time.sleep(0.05)
    return None


def probe_camera(camera_id, backend_key, width, height):
    backend_name, backend_id = BACKENDS[backend_key]
    cap = open_capture(camera_id, backend_id, width, height)

    if not cap.isOpened():
        cap.release()
        return None

    frame = read_first_frame(cap)
    if frame is None:
        cap.release()
        return None

    frame_height, frame_width = frame.shape[:2]
    info = {
        "camera_id": camera_id,
        "backend_key": backend_key,
        "backend_name": backend_name,
        "backend_id": backend_id,
        "width": frame_width,
        "height": frame_height,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    cap.release()
    return info


def scan_cameras(max_id, backend_key, width, height):
    backend_name = BACKENDS[backend_key][0]
    print(f"Scanning OpenCV camera IDs 0..{max_id} with backend {backend_name}")

    found = []
    for camera_id in range(max_id + 1):
        info = probe_camera(camera_id, backend_key, width, height)
        if info is None:
            print(f"ID {camera_id}: not available")
            continue

        found.append(info)
        print(
            f"ID {camera_id}: OK | "
            f"{info['width']}x{info['height']} | "
            f"fps={info['fps']:.1f} | fourcc={info['fourcc']}"
        )

    return found


def draw_label(frame, lines):
    x, y = 18, 30
    line_height = 28
    box_height = line_height * len(lines) + 14
    cv2.rectangle(frame, (8, 8), (760, box_height), (0, 0, 0), -1)

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        y += line_height


def preview_candidates(candidates, width, height):
    print("\nPreview controls:")
    print("  s = select this camera ID")
    print("  n / ENTER / SPACE = next camera")
    print("  q / ESC = quit preview")

    selected = None
    for info in candidates:
        camera_id = info["camera_id"]
        backend_id = info["backend_id"]
        backend_name = info["backend_name"]

        cap = open_capture(camera_id, backend_id, width, height)
        if not cap.isOpened():
            cap.release()
            continue

        print(f"\nShowing camera ID {camera_id} using {backend_name}")
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            draw_label(
                frame,
                [
                    f"OpenCV camera ID: {camera_id}  Backend: {backend_name}",
                    "If this is the Orbec/Orbbec image, press s.",
                    "Press n for next, q to quit.",
                ],
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                selected = info
                break
            if key in (ord("n"), 13, 32):
                break
            if key in (ord("q"), 27):
                cap.release()
                cv2.destroyAllWindows()
                return selected

        cap.release()

        if selected is not None:
            break

    cv2.destroyAllWindows()
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Find the OpenCV camera ID for a USB Orbec/Orbbec camera."
    )
    parser.add_argument("--max-id", type=int, default=4q)
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS.keys()),
        default="msmf",
        help="Use msmf for your current scripts, or try dshow if MSMF fails.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Only print available camera IDs; do not open preview windows.",
    )
    args = parser.parse_args()

    print_windows_camera_devices()
    candidates = scan_cameras(args.max_id, args.backend, args.width, args.height)

    if not candidates:
        print("\nNo cameras opened. Try:")
        print("  python 007_find_orbec_camera_id.py --backend dshow")
        print("  python 007_find_orbec_camera_id.py --max-id 20")
        return

    if args.no_preview:
        print("\nUse the ID that belongs to the Orbec/Orbbec camera image.")
        return

    selected = preview_candidates(candidates, args.width, args.height)
    if selected is None:
        print("\nNo camera selected.")
        return

    print("\nSelected camera:")
    print(
        f"CAM_ID = {selected['camera_id']} "
        f"with backend {selected['backend_name']}"
    )
    print("Use this value in your scripts, for example:")
    print(f"CAM_ID = {selected['camera_id']}")


if __name__ == "__main__":
    main()
