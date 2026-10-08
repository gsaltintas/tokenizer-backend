from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ComparisonFloresRequest,
    ComparisonOverlapRequest,
    ComparisonPretokenizeResponse,
    ComparisonTextRequest,
    ComparisonTokenizeRequest,
    ComparisonTokenizeResponse,
    EfficiencyMetric,
    EfficiencyRequest,
    EfficiencyResponse,
    OverlapResult,
    TokenizerTokenization,
    TokenInfo,
)
from app.services.comparison import (
    compare_pretokenization,
    compare_tokenization,
    compute_efficiency,
    compute_overlap,
)
from app.services.intrinsic_eval import flores_gini_metrics, per_text_metrics
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


# The endpoints below are plain `def`: FastAPI runs them in a worker thread, so a slow
# comparison doesn't block other requests.


@router.post("/pretokenize", response_model=ComparisonPretokenizeResponse)
def compare_pretokenize(req: ComparisonTextRequest):
    """Normalization, pre-token chunks and tokens per tokenizer, with pairwise boundary agreement."""
    adapters = _get_adapters(req.tokenizer_ids)
    return compare_pretokenization(adapters, req.text)


@router.post("/intrinsic")
def compare_intrinsic(req: ComparisonTextRequest):
    """TokEval per-text metrics for each tokenizer on the same text."""
    adapters = _get_adapters(req.tokenizer_ids)
    return {
        "results": [
            {"tokenizer_id": tok_id, "metrics": per_text_metrics(adapter, req.text)}
            for tok_id, adapter in adapters.items()
        ]
    }


@router.post("/flores")
def compare_flores(req: ComparisonFloresRequest):
    """FLORES+ per-language compression and Gini for each tokenizer, on the same sentences."""
    adapters = _get_adapters(req.tokenizer_ids)
    return {
        "results": [
            {"tokenizer_id": tok_id, **flores_gini_metrics(adapter, req.language_codes, req.n_samples)}
            for tok_id, adapter in adapters.items()
        ]
    }
