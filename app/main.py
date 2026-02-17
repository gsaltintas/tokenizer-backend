from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.routers import (
    comparison,
    language,
    merge_tree,
    morphemes,
    multiplicity,
    tokenize,
    tokenizers,
    undertrained,
    vocabulary,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


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
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Explicit OPTIONS handler for CORS preflight
@app.options("/{full_path:path}")
async def options_handler(request: Request, full_path: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
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


@app.get("/api/health")
async def health():
    return {"status": "ok"}
