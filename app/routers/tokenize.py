import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.models.schemas import TokenInfo, TokenizeRequest, TokenizeResponse
from app.services.registry import registry
from app.services.tokens import build_token_infos

router = APIRouter(prefix="/api/tokenize", tags=["tokenize"])


@router.post("", response_model=TokenizeResponse)
async def tokenize_text(req: TokenizeRequest):
    """Encode text and return tokens with positions."""
    adapter = registry.get(req.tokenizer_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{req.tokenizer_id}' not loaded")

    tokens = build_token_infos(adapter, req.text)
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

                tokens = build_token_infos(adapter, text)
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
