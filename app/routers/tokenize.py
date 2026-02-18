import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.models.schemas import TokenInfo, TokenizeRequest, TokenizeResponse
from app.services.registry import registry

router = APIRouter(prefix="/api/tokenize", tags=["tokenize"])


def _build_tokens(adapter, text: str) -> list[TokenInfo]:
    token_ids = adapter.encode(text)
    tokens: list[TokenInfo] = []
    offset = 0
    prev_decoded = ""
    for i, tid in enumerate(token_ids):
        # Use incremental decoding to preserve context (e.g. SentencePiece ▁ → space).
        # Decode prefix token_ids[:i+1] and diff against previous prefix.
        curr_decoded = adapter.decode(token_ids[: i + 1])
        token_str = curr_decoded[len(prev_decoded):]
        prev_decoded = curr_decoded

        token_bytes = token_str.encode("utf-8", errors="replace")
        start = text.find(token_str, offset)
        if start == -1:
            start = offset
        end = start + len(token_str)
        offset = end
        tokens.append(
            TokenInfo(
                id=tid,
                token_str=token_str,
                token_bytes_hex=token_bytes.hex(),
                byte_length=len(token_bytes),
                start=start,
                end=end,
            )
        )
    return tokens


@router.post("", response_model=TokenizeResponse)
async def tokenize_text(req: TokenizeRequest):
    """Encode text and return tokens with positions."""
    adapter = registry.get(req.tokenizer_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{req.tokenizer_id}' not loaded")

    tokens = _build_tokens(adapter, req.text)
    return TokenizeResponse(
        tokens=tokens,
        token_count=len(tokens),
        char_count=len(req.text),
    )


@router.websocket("/ws")
async def tokenize_ws(websocket: WebSocket):
    """Real-time tokenization via WebSocket."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                tokenizer_id = msg.get("tokenizer_id", "")
                text = msg.get("text", "")

                adapter = registry.get(tokenizer_id)
                if adapter is None:
                    await websocket.send_json(
                        {"error": f"Tokenizer '{tokenizer_id}' not loaded"}
                    )
                    continue

                tokens = _build_tokens(adapter, text)
                response = TokenizeResponse(
                    tokens=tokens,
                    token_count=len(tokens),
                    char_count=len(text),
                )
                await websocket.send_json(response.model_dump())
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
            except Exception as e:
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        pass
