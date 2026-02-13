from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ComparisonOverlapRequest,
    ComparisonTokenizeRequest,
    ComparisonTokenizeResponse,
    EfficiencyMetric,
    EfficiencyRequest,
    EfficiencyResponse,
    OverlapResult,
    TokenizerTokenization,
    TokenInfo,
)
from app.services.comparison import compare_tokenization, compute_efficiency, compute_overlap
from app.services.registry import registry

router = APIRouter(prefix="/api/comparison", tags=["comparison"])


def _get_adapters(tokenizer_ids: list[str]):
    adapters = {}
    for tok_id in tokenizer_ids:
        adapter = registry.get(tok_id)
        if adapter is None:
            raise HTTPException(
                status_code=404, detail=f"Tokenizer '{tok_id}' not loaded"
            )
        adapters[tok_id] = adapter
    return adapters


@router.post("/overlap", response_model=OverlapResult)
async def get_overlap(req: ComparisonOverlapRequest):
    adapters = _get_adapters(req.tokenizer_ids)
    result = compute_overlap(adapters)
    return OverlapResult(**result)


@router.post("/tokenize", response_model=ComparisonTokenizeResponse)
async def compare_tokenize(req: ComparisonTokenizeRequest):
    adapters = _get_adapters(req.tokenizer_ids)
    results = compare_tokenization(adapters, req.text)
    return ComparisonTokenizeResponse(
        results=[
            TokenizerTokenization(
                tokenizer_id=r["tokenizer_id"],
                tokens=[TokenInfo(**t) for t in r["tokens"]],
                token_count=r["token_count"],
            )
            for r in results
        ],
        text=req.text,
    )


@router.post("/efficiency", response_model=EfficiencyResponse)
async def compare_efficiency(req: EfficiencyRequest):
    adapters = _get_adapters(req.tokenizer_ids)
    results = compute_efficiency(adapters, req.sample_texts)
    return EfficiencyResponse(
        metrics=[EfficiencyMetric(**r) for r in results]
    )
