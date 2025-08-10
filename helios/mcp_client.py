from __future__ import annotations

import anyio
import httpx
from dataclasses import dataclass
from typing import Any, Optional
from loguru import logger


@dataclass
class MCPClient:
    base_url: str
    auth_token: Optional[str] = None
    timeout_s: float = 30.0

    async def call(self, tool: str, payload: dict[str, Any], timeout_s: Optional[float] = None) -> dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("MCP base_url not configured")
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        url = f"{self.base_url.rstrip('/')}/tools/{tool}"
        to = timeout_s or self.timeout_s
        logger.debug(f"MCP call {tool} -> {url} (timeout={to}s)")
        async with httpx.AsyncClient(timeout=to) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def try_call(self, tool: str, payload: dict[str, Any], timeout_s: Optional[float] = None) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        try:
            data = await self.call(tool, payload, timeout_s)
            return data, None
        except Exception as e:
            logger.warning(f"MCP call failed: {tool}: {e}")
            return None, str(e)

    @staticmethod
    def from_env(base_url: Optional[str], auth_token: Optional[str]) -> Optional["MCPClient"]:
        if not base_url:
            return None
        return MCPClient(base_url=base_url, auth_token=auth_token)
