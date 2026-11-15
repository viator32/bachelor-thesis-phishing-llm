from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from .mlx_server import MlxServerProcess


# Read MLX_API_KEY or the configured oMLX settings file.
def mlx_api_key() -> str | None:
    direct = os.getenv("MLX_API_KEY")
    if direct:
        return direct
    settings_path = (
        os.getenv("OMLX_SETTINGS_PATH")
        or os.getenv("OMLX_SETTINGS_FILE")
        or "~/.omlx/settings.json"
    )
    try:
        settings = json.loads(Path(settings_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    key = (settings.get("auth") or {}).get("api_key")
    return str(key) if key else None


class MlxAdapter:
    """Talks to a local OpenAI-compatible MLX server (Apple Silicon only).

    The oMLX app manages model downloads and its OpenAI-compatible server. Its configured URL,
    API key and exact model ID must be passed to this adapter. ``auto_start`` is only for a raw
    ``mlx_lm.server`` process managed by the prototype.
    """

    # Configure MLX requests and optionally create an owned server manager.
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        launch_model: str | None = None,
        timeout_seconds: int = 180,
        temperature: float = 0.0,
        seed: int = 42,
        max_output_tokens: int = 1024,
        auto_start: bool = False,
        startup_timeout: float = 180,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self._server: MlxServerProcess | None = None
        if auto_start:
            self._server = MlxServerProcess(
                model=launch_model or model,
                base_url=self.base_url,
                startup_timeout=startup_timeout,
            )

    # Build bearer authentication when an API key is configured.
    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    # Start the owned server when one is configured.
    def _ensure_ready(self) -> None:
        if self._server is not None:
            self._server.ensure_started()

    # Report adapter availability and model information.
    def health(self) -> dict[str, Any]:
        self._ensure_ready()
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{self.base_url}/v1/models", headers=self._headers)
            response.raise_for_status()
            body = response.json()
        names = [item.get("id") for item in body.get("data", [])]
        return {"backend": "mlx", "reachable": True, "models": names, "selected": self.model}

    # Release owned resources; the base adapter does nothing.
    def close(self) -> None:
        if self._server is not None:
            self._server.terminate()
