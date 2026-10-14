from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import pretokenize as svc
from app.services.registry import registry

router = APIRouter(prefix="/api/pretokenize", tags=["pretokenize"])


class PretokenizeRequest(BaseModel):
    text: str = Field(..., max_length=2000)
    tokenizer_id: str


class NormalizationInfoOut(BaseModel):
    type: str
    normalized_text: str
    changed: bool


class EncodedUnitOut(BaseModel):
    text: str
    atomic_ids: list[int]
    atomic_labels: list[str]


class EncodingInfoOut(BaseModel):
    type: str
    description: str
    units: list[EncodedUnitOut]


class PretokenizeResponse(BaseModel):
    normalization: NormalizationInfoOut
    chunks: list[str]
    chunk_spans: list[tuple[int, int]]
    chunk_count: int
    pretokenizer_type: str
    pretokenizer_description: str
    regex_pattern: str | None
    encoding: EncodingInfoOut | None = None


@router.post("", response_model=PretokenizeResponse)
async def pretokenize_text(req: PretokenizeRequest):
    adapter = registry.get(req.tokenizer_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{req.tokenizer_id}' not loaded")
    result = svc.analyze(adapter, req.text)
    return PretokenizeResponse(
        normalization=NormalizationInfoOut(
            type=result.normalization.type,
            normalized_text=result.normalization.normalized_text,
            changed=result.normalization.changed,
        ),
        chunks=result.chunks,
        chunk_spans=result.chunk_spans,
        chunk_count=result.chunk_count,
        pretokenizer_type=result.pretokenizer_type,
        pretokenizer_description=result.pretokenizer_description,
        regex_pattern=result.regex_pattern,
        encoding=EncodingInfoOut.model_validate(result.encoding, from_attributes=True)
        if result.encoding
        else None,
    )
