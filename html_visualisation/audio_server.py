import asyncio
import json
import threading
import numpy as np
import sounddevice as sd
import websockets

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512
FFT_SIZE = 256


class AudioServer:
    """WebSocket server that streams microphone amplitude to connected clients."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: set[websockets.WebSocketServerProtocol] = set()
        self._running = False
        self._task: asyncio.Task | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._task = self._loop.create_task(self._serve())
        self._loop.run_until_complete(self._task)

    async def _serve(self) -> None:
        server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
        )
        print(f"[AudioServer] WebSocket server listening on ws://{self.host}:{self.port}")
        asyncio.create_task(self._capture_loop())
        await server.wait_closed()

    async def _handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        self._clients.add(websocket)
        print(f"[AudioServer] Client connected. Total clients: {len(self._clients)}")
        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            print(f"[AudioServer] Client disconnected. Total clients: {len(self._clients)}")

    async def _capture_loop(self) -> None:
        """Capture microphone audio and broadcast average amplitude to all clients."""
        print("[AudioServer] Starting microphone capture...")
        loop = asyncio.get_running_loop()

        def audio_callback(indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags) -> None:
            if not self._clients:
                return
            # indata shape: (frames, channels)
            avg = float(np.mean(np.abs(indata))) * 128.0
            avg = min(avg, 255.0)
            message = json.dumps({"audio": avg})
            for ws in list(self._clients):
                loop.call_soon_threadsafe(lambda w=ws, m=message: asyncio.create_task(self._safe_send(w, m)))

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=CHUNK_SIZE,
                dtype="float32",
                callback=audio_callback,
            ):
                while self._running:
                    await asyncio.sleep(0.1)
        except Exception as exc:
            print(f"[AudioServer] Microphone error: {exc}")

    async def _safe_send(self, ws: websockets.WebSocketServerProtocol, message: str) -> None:
        try:
            await ws.send(message)
        except websockets.exceptions.ConnectionClosed:
            self._clients.discard(ws)
        except Exception:
            pass

    def stop(self) -> None:
        """Stop the server and release resources."""
        self._running = False
        if self._loop is not None and self._task is not None:
            self._loop.call_soon_threadsafe(self._task.cancel)
        if self._thread is not None:
            self._thread.join(timeout=2)
        print("[AudioServer] Stopped.")


if __name__ == "__main__":
    server = AudioServer()
    server.start()
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
