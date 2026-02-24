"""FastAPI application for the AFE Viewer backend."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

# Resolve paths relative to this file:
#   app/main.py -> server/ -> viewer/ -> viewer/public/presets/
SERVER_DIR = Path(__file__).resolve().parent.parent
VIEWER_DIR = SERVER_DIR.parent
PRESETS_INDEX = VIEWER_DIR / "public" / "presets" / "index.json"

# Shared state populated at startup
preset_registry: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load preset index on startup."""
    if PRESETS_INDEX.exists():
        with open(PRESETS_INDEX) as f:
            preset_registry.extend(json.load(f))
    yield


app = FastAPI(
    title="AFE Viewer Server",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS -- allow the Vite dev server and common local ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


def run() -> None:
    """Entry point for ``serve`` console script."""
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
