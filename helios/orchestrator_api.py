from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import FastAPI, BackgroundTasks

from .config import load_config
from .services.helios_orchestrator import create_helios_orchestrator, HeliosOrchestrator


app = FastAPI(title="Helios Orchestrator", version="1.0.0")


# Global orchestrator instance
orchestrator: HeliosOrchestrator | None = None


@app.on_event("startup")
async def on_startup() -> None:
    global orchestrator
    cfg = load_config()
    orchestrator = await create_helios_orchestrator(cfg)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    if orchestrator is not None:
        await orchestrator.cleanup()


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> Dict[str, Any]:
    return {"status": "ready"}


@app.post("/run")
async def run_once() -> Dict[str, Any]:
    if orchestrator is None:
        return {"status": "error", "message": "Orchestrator not initialized"}
    result = await orchestrator.run_complete_cycle()  # Call the correct method
    return result


@app.post("/run-async")
async def run_async(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    if orchestrator is None:
        return {"status": "error", "message": "Orchestrator not initialized"}

    async def _task() -> None:
        await orchestrator.run_complete_cycle()

    # Detach the long-running work from the request
    background_tasks.add_task(asyncio.create_task, _task())
    return {"status": "accepted"}


