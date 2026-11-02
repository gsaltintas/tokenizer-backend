import asyncio
import gc
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers import (
    comparison,
    intrinsic_eval,
    language,
    merge_forest,
    merge_tree,
    morphemes,
    multiplicity,
    pretokenize,
    sanity_check,
    tokenize,
    tokenizers,
    undertrained,
    visualize,
    vocabulary,
)
from app.services.registry import registry


logger = logging.getLogger(__name__)

# Tokenizers no request has used for this many minutes are unloaded (0 disables).
# Unloading is safe: the next request that names one loads it again.
IDLE_TTL_MIN = float(os.environ.get("TOKENIZER_IDLE_TTL_MIN", "30"))
IDLE_CHECK_S = 300


async def _evict_idle_tokenizers():
    while True:
        await asyncio.sleep(IDLE_CHECK_S)
        evicted = registry.evict_idle(IDLE_TTL_MIN * 60)
        if evicted:
            gc.collect()
            logger.warning("Unloaded idle tokenizers: %s", ", ".join(evicted))


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_evict_idle_tokenizers()) if IDLE_TTL_MIN > 0 else None
    yield
    if task:
        task.cancel()


app = FastAPI(
    title="Tokenizer Explorer",
    version="0.1.0",
    lifespan=lifespan,
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# MUST be BEFORE including routers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tokenizers.gsaltintas.com",
        "https://tokenizer-exploration.gsaltintas78.workers.dev",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tokenizers.router)
app.include_router(tokenize.router)
app.include_router(vocabulary.router)
app.include_router(multiplicity.router)
app.include_router(language.router)
app.include_router(morphemes.router)
app.include_router(undertrained.router)
app.include_router(comparison.router)
app.include_router(merge_tree.router)
app.include_router(merge_forest.router)
app.include_router(pretokenize.router)
app.include_router(intrinsic_eval.router)
app.include_router(sanity_check.router)
app.include_router(visualize.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve the React SPA — must come after all API routes
_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.exists():
    app.mount("/assets", StaticFiles(directory=_static_dir / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        return FileResponse(_static_dir / "index.html")
