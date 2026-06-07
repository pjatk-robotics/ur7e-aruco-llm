import asyncio
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


class AudioServer:
    """Local server for live audio/status events and camera visualization."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        http_port: int = 8766,
    ) -> None:
        self.host = host
        self.port = port
        self.http_port = http_port
        self._frame_provider: Callable[[], bytes | None] | None = None
        self._placeholder_frames: dict[str, bytes] = {}
        self._last_status: dict[str, Any] = {
            "phase": "idle",
            "message": "Ready",
        }
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._clients: set[Any] = set()
        self._running = False
        self._thread: threading.Thread | None = None
        self._http_thread: threading.Thread | None = None
        self._http_server: ThreadingHTTPServer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._startup_error: Exception | None = None

    @property
    def url(self) -> str:
        return f"ws://localhost:{self.port}"

    @property
    def html_path(self) -> Path:
        return Path(__file__).with_name("index.html").resolve()

    @property
    def page_url(self) -> str:
        return f"http://localhost:{self.http_port}/"

    def set_camera_frame_provider(
        self,
        frame_provider: Callable[[], bytes | None],
    ) -> None:
        self._frame_provider = frame_provider

    def start(self) -> bool:
        """Start the WebSocket and HTTP servers in background threads."""
        if self._running:
            return True

        self._running = True
        self._ready.clear()
        self._startup_error = None

        try:
            self._start_http_server()
        except OSError as exc:
            self._startup_error = exc
            self._running = False
            print(f"[AudioServer] Could not start HTTP server: {exc}")
            return False

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        ready = self._ready.wait(timeout=3.0)

        if self._startup_error is not None or not ready:
            print(f"[AudioServer] Could not start: {self._startup_error}")
            self.stop()
            return False

        return self._running

    def broadcast_status(
        self,
        phase: str,
        message: str,
        **payload: Any,
    ) -> None:
        """Send one workflow status update to all connected visualizations."""
        status = {
            "phase": phase,
            "message": message,
            **payload,
        }
        self._last_status = status
        self._send_json({"status": status})

    def broadcast_audio_level(self, level: float) -> None:
        """Send one audio level sample to all connected visualizations."""
        level = max(0.0, min(255.0, float(level)))
        self._send_json({"audio": level})

    def wait_for_command(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Return the next browser command, or None when the timeout expires."""
        try:
            return self._commands.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_commands(self) -> None:
        """Drop stale browser commands before entering an idle wait state."""
        while True:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                return

    def _send_json(self, payload: dict[str, Any]) -> None:
        if not self._running or self._loop is None:
            return

        message = json.dumps(payload)
        self._loop.call_soon_threadsafe(self._schedule_broadcast, message)

    def stop(self) -> None:
        """Stop the server and release resources."""
        self.broadcast_audio_level(0.0)
        self._running = False

        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None

        if self._http_thread is not None:
            self._http_thread.join(timeout=2.0)
            self._http_thread = None

    def _start_http_server(self) -> None:
        handler = self._make_http_handler()
        self._http_server = ThreadingHTTPServer((self.host, self.http_port), handler)
        self._http_server.daemon_threads = True
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            daemon=True,
        )
        self._http_thread.start()

    def _run_loop(self) -> None:
        try:
            import websockets

            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._stop_event = asyncio.Event()
            self._loop.run_until_complete(self._serve(websockets))
        except Exception as exc:
            self._startup_error = exc
            self._running = False
            self._ready.set()
        finally:
            if self._loop is not None:
                self._loop.close()
                self._loop = None
            self._stop_event = None

    async def _serve(self, websockets_module: Any) -> None:
        async with websockets_module.serve(
            self._handler,
            self.host,
            self.port,
        ):
            print(f"[AudioServer] Visualization websocket: {self.url}")
            print(f"[AudioServer] Open: {self.page_url}")
            self._ready.set()
            if self._stop_event is not None:
                await self._stop_event.wait()

    async def _handler(self, websocket: Any, *args: Any) -> None:
        self._clients.add(websocket)
        print(f"[AudioServer] Client connected. Total clients: {len(self._clients)}")
        try:
            await websocket.send(json.dumps({"status": self._last_status}))
            async for message in websocket:
                await self._handle_client_message(websocket, message)
        finally:
            self._clients.discard(websocket)
            print(
                f"[AudioServer] Client disconnected. "
                f"Total clients: {len(self._clients)}"
            )

    async def _handle_client_message(self, websocket: Any, message: Any) -> None:
        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return

        command = payload.get("command")
        if not isinstance(command, str):
            return

        if command not in {"start_listening", "quit", "select_id"}:
            return

        self._commands.put(payload)
        try:
            await websocket.send(json.dumps({"ack": command}))
        except Exception:
            self._clients.discard(websocket)

    def _schedule_broadcast(self, message: str) -> None:
        for websocket in list(self._clients):
            asyncio.create_task(self._safe_send(websocket, message))

    async def _safe_send(self, websocket: Any, message: str) -> None:
        try:
            await websocket.send(message)
        except Exception:
            self._clients.discard(websocket)

    def _placeholder_frame(self, message: str) -> bytes:
        cached = self._placeholder_frames.get(message)
        if cached is not None:
            return cached

        try:
            import cv2
            import numpy as np

            frame = np.full((720, 1280, 3), (32, 24, 17), dtype=np.uint8)
            cv2.putText(
                frame,
                message,
                (90, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (230, 236, 242),
                3,
                cv2.LINE_AA,
            )
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 82],
            )
            if ok:
                cached = encoded.tobytes()
                self._placeholder_frames[message] = cached
                return cached
        except Exception:
            pass

        # 1x1 black JPEG fallback.
        cached = (
            b"\xff\xd8\xff\xdb\x00C\x00"
            b"\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\x09\x09\x08"
            b"\x0a\x0c\x14\x0d\x0c\x0b\x0b\x0c\x19\x12\x13\x0f"
            b"\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c"
            b"\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b"
            b"\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14"
            b"\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?"
            b"\x00\xd2\xcf \xff\xd9"
        )
        self._placeholder_frames[message] = cached
        return cached

    def _make_http_handler(self):
        manager = self

        class VisualizationRequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in {"/", "/index.html"}:
                    self._serve_index()
                    return

                if path == "/camera.mjpg":
                    self._serve_camera_stream()
                    return

                self.send_error(404)

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _serve_index(self) -> None:
                try:
                    content = manager.html_path.read_bytes()
                except OSError:
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def _serve_camera_stream(self) -> None:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()

                while manager._running:
                    if manager._frame_provider is None:
                        frame = manager._placeholder_frame(
                            "Camera provider not connected"
                        )
                    else:
                        frame = manager._frame_provider()

                    if frame is None:
                        frame = manager._placeholder_frame(
                            "Waiting for camera frame"
                        )

                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except OSError:
                        break

                    time.sleep(1 / 20)

        return VisualizationRequestHandler


if __name__ == "__main__":
    server = AudioServer()
    server.start()

    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
