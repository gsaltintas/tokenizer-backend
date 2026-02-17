from pydantic import BaseModel, Field


class TokenizerInfo(BaseModel):
    id: str
    name: str
    tokenizer_type: str
    vocab_size: int
    source: str  # "tiktoken", "huggingface", "sentencepiece"


class TokenizerListResponse(BaseModel):
    tokenizers: list[TokenizerInfo]


class LoadTokenizerRequest(BaseModel):
    name: str = Field(..., description="Tokenizer name, HuggingFace model ID, or file path")


class LoadTokenizerResponse(BaseModel):
    tokenizer: TokenizerInfo


class TokenInfo(BaseModel):
    id: int
    token_str: str
    token_bytes_hex: str
    byte_length: int
    start: int | None = None
    end: int | None = None


class TokenizeRequest(BaseModel):
    text: str
    tokenizer_id: str


class TokenizeResponse(BaseModel):
    tokens: list[TokenInfo]
    token_count: int
    char_count: int


class VocabEntry(BaseModel):
    id: int
    token_str: str
    token_bytes_hex: str
    byte_length: int
    script: str = ""
    morpheme_type: str = ""


class VocabResponse(BaseModel):
    entries: list[VocabEntry]
    total: int
    page: int
    page_size: int


class VocabStatsResponse(BaseModel):
    vocab_size: int
    avg_token_length: float
    max_token_length: int
    length_distribution: dict[int, int]
    script_distribution: dict[str, int]


class VariantInfo(BaseModel):
    token_id: int
    token_str: str
    has_space_prefix: bool
    casing: str  # "lower", "upper", "title", "mixed"
    has_punctuation: bool


class MultiplicityGroup(BaseModel):
    base_form: str
    variants: list[VariantInfo]
    count: int


class MultiplicityResponse(BaseModel):
    groups: list[MultiplicityGroup]
    total_groups: int
    page: int
    page_size: int


class ScriptCategory(BaseModel):
    script: str
    token_count: int
    percentage: float
    example_tokens: list[str]


class LanguageCompositionResponse(BaseModel):
    categories: list[ScriptCategory]
    total_tokens: int
    mixed_script_count: int


class MorphemeBreakdown(BaseModel):
    token_str: str
    token_id: int
    morpheme_type: str  # "morpheme", "morpheme_composite", "subword", "arbitrary"
    morphemes: list[str]


class MorphemeAnalysisResponse(BaseModel):
    breakdowns: list[MorphemeBreakdown]
    total: int
    page: int
    page_size: int
    type_distribution: dict[str, int]


class UndertrainedToken(BaseModel):
    token_id: int
    token_str: str
    token_bytes_hex: str
    reason: str
    confidence: float
    expected_merge_path: list[str]
    actual_merge_result: list[str]


class UndertrainedResponse(BaseModel):
    tokens: list[UndertrainedToken]
    total: int
    page: int
    page_size: int
    bpe_available: bool


class ComparisonOverlapRequest(BaseModel):
    tokenizer_ids: list[str] = Field(..., min_length=2)


class OverlapResult(BaseModel):
    shared_tokens: int
    unique_per_tokenizer: dict[str, int]
    total_union: int
    overlap_percentage: float
    shared_sample: list[str]
    unique_samples: dict[str, list[str]]


class ComparisonTokenizeRequest(BaseModel):
    text: str
    tokenizer_ids: list[str] = Field(..., min_length=2)


class TokenizerTokenization(BaseModel):
    tokenizer_id: str
    tokens: list[TokenInfo]
    token_count: int


class ComparisonTokenizeResponse(BaseModel):
    results: list[TokenizerTokenization]
    text: str


class EfficiencyRequest(BaseModel):
    tokenizer_ids: list[str] = Field(..., min_length=2)
    sample_texts: list[str] | None = None


class EfficiencyMetric(BaseModel):
    tokenizer_id: str
    avg_tokens_per_word: float
    avg_token_length_chars: float
    total_tokens: int
    total_chars: int


class EfficiencyResponse(BaseModel):
    metrics: list[EfficiencyMetric]


# ── Merge Tree ────────────────────────────────────────────────────────────────


class MergeTreeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200)
    tokenizer_ids: list[str] = Field(..., min_length=2, max_length=2)


class MergeTreeNode(BaseModel):
    token: str
    rank: int
    is_leaf: bool
    left: "MergeTreeNode | None" = None
    right: "MergeTreeNode | None" = None


class MergeStepInfo(BaseModel):
    step: int
    merged_token: str
    rank: int
    tokens_after: list[str]


class MergeTreeTokenizerResult(BaseModel):
    name: str
    trees: list[MergeTreeNode]
    steps: list[MergeStepInfo]
    final_tokens: list[str]


class ConflictAnalysis(BaseModel):
    shared_intermediates: list[str]
    only_a: list[str]
    only_b: list[str]
    is_compatible: bool
    conflict_count: int


class MergeTreeComparisonResponse(BaseModel):
    text: str
    initial_bytes: list[str]
    tokenizer_a: MergeTreeTokenizerResult
    tokenizer_b: MergeTreeTokenizerResult
    conflict_analysis: ConflictAnalysis
