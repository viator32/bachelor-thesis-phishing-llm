from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx


class MlxServerStartupError(RuntimeError):
    pass


class MlxServerProcess:
    """Owns the lifecycle of a locally spawned `mlx_lm.server` process.

    If a server is already reachable at `base_url`, it is adopted instead of spawning a new
    one — in that case this instance never owns a process and `load_duration_ms`/memory sampling
    are unavailable because this instance does not own the external oMLX process.
    """

    # Configure an optional owned local server and its log directory.
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080",
        startup_timeout: float = 180,
        log_dir: str | Path = "results/logs",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.startup_timeout = startup_timeout
        self.log_dir = Path(log_dir)
        self.process: subprocess.Popen | None = None
        self.load_duration_ms: float | None = None
        self._log_path: Path | None = None
        self._started = False

    # Read the owned server process ID.
    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process else None

    # Probe the model endpoint.
    def _reachable(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/v1/models", timeout=2)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    # Adopt a reachable server or spawn and warm up an owned server.
    def ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        if self._reachable():
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        port = self.base_url.rsplit(":", 1)[-1]
        self._log_path = self.log_dir / f"mlx-server-{port}.log"
        log_handle = self._log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "mlx_lm.server", "--model", self.model, "--port", port],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise MlxServerStartupError(
                    f"mlx_lm.server exited early (code {self.process.returncode}). "
                    f"See {self._log_path} for details."
                )
            if self._reachable():
                break
            time.sleep(1)
        else:
            self.terminate()
            raise MlxServerStartupError(
                f"mlx_lm.server did not become reachable at {self.base_url} within "
                f"{self.startup_timeout}s. See {self._log_path} for details."
            )

        # A 1-token warmup call forces model weights to actually load before any real request is
        # timed, so load_duration_ms below reflects load time rather than the first inference.
        started = time.perf_counter()
        try:
            httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
                timeout=self.startup_timeout,
            )
        except httpx.HTTPError as exc:
            raise MlxServerStartupError(
                f"mlx_lm.server warmup request failed: {exc}. See {self._log_path} for details."
            ) from exc
        self.load_duration_ms = (time.perf_counter() - started) * 1000

    # Terminate an owned process, killing it if shutdown times out.
    def terminate(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
