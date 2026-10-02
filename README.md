# Tokenizer Explorer — Backend

FastAPI service behind the [tokenizer visualization](https://tokenizers.gsaltintas.com) frontend. It loads tokenizers from several sources and exposes analyses of them over a JSON/WebSocket API: tokenization, vocabulary statistics, BPE merge trees, cross-tokenizer comparison, and intrinsic evaluation on FLORES+.

## Supported tokenizers

Tokenizers are loaded by name and kept in an in-memory LRU cache (10 entries). A name is resolved in this order (`app/services/registry.py`):

1. **tiktoken**: `gpt-4o`, `gpt-4`, `gpt-3.5-turbo`, `cl100k_base`, `o200k_base`, `p50k_base`, `r50k_base`, `gpt2`
2. **SentencePiece**: a path to an existing `.model` file
3. **TokenMonster**: a `.vocab` file or a preset such as `english-32000-consistent-v1` (requires `pip install tokenmonster`, which is not in the default dependencies)
4. **Hugging Face**: any Hub model ID, e.g. `meta-llama/Llama-3.2-1B`, with an optional `subfolder`

Gated Hub models (Llama, Gemma, …) need a Hugging Face token in the environment (`HF_TOKEN`, or `huggingface-cli login`).

## Setup

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

Interactive API docs are then at <http://localhost:8000/docs>.

`uv sync` installs CPU-only PyTorch and pulls [`tokenizer-intrinsic-evals`](https://github.com/cimeister/tokenizer-intrinsic-evals) from GitHub. `requirements.txt` is a lighter, older dependency list without the intrinsic-eval stack; `pyproject.toml` is authoritative.

### Running on a cluster

`serve.sh` starts the server on `127.0.0.1:8000` inside a Slurm allocation:

```bash
srun --jobid=<id> --overlap ./serve.sh
```

It puts the Hugging Face cache on node-local disk (`/localscratch` if present, otherwise `/tmp`). If `~/.cloudflared-token` exists, it also starts a Cloudflare tunnel, passing the token through the environment rather than argv. It restarts the backend if it exits or fails `/api/health`, and only connects the tunnel once the backend is healthy.

For continuous uptime, run `./ensure_serving.sh`. It submits `serve.sbatch`, a 5-day job that queues its successor on another node to start 12 h before its own time limit. Once the successor is healthy, it cancels the old job, and the tunnel fails over without a gap. `ensure_serving.sh` is idempotent: it submits a job if none is queued and releases a held successor early if nothing is running. Run it from cron as a watchdog (see the header of the script). To stop: `touch STOP_SERVING && scancel -n tokenizer-serve`.

### Serving the frontend

If a `static/` directory exists next to `app/` (the frontend's `dist/` build output), the app serves it at `/` with an SPA fallback, so one process can host both.

### CORS

Allowed origins are hard-coded in `app/main.py`: the production domains plus `localhost:5173` / `127.0.0.1:5173` for the Vite dev server. Add any new frontend origin there.

## Workflow

Most endpoints act on an already-loaded tokenizer. Load it first:

```bash
curl -X POST localhost:8000/api/tokenizers/load \
  -H 'Content-Type: application/json' \
  -d '{"name": "gpt-4o"}'

curl -X POST localhost:8000/api/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"tokenizer_id": "gpt-4o", "text": "Hello, world!"}'
```

Endpoints for a tokenizer that isn't loaded return `404`.

## API overview

Tokenizer IDs are path parameters and may contain slashes (`/api/vocab/meta-llama/Llama-3.2-1B`).

| Area | Endpoint | Description |
| --- | --- | --- |
| Tokenizers | `GET /api/tokenizers` | Loaded tokenizers plus presets |
| | `POST /api/tokenizers/load` | Load by `name` (+ optional `subfolder`) |
| | `POST /api/tokenizers/{id}/reload` | Evict from the cache and reload |
| Tokenize | `POST /api/tokenize` | Tokens with IDs, bytes, and character offsets |
| | `WS /api/tokenize/ws` | Live tokenization: send `{"tokenizer_id", "text"}` JSON, get the same response as the POST |
| Pre-tokenize | `POST /api/pretokenize` | Pre-tokenizer splits |
| Vocabulary | `GET /api/vocab/{id}` | Paginated vocabulary listing |
| | `GET /api/vocab/stats/{id}` | Vocabulary statistics |
| Multiplicity | `GET /api/multiplicity/{id}` | Tokens that are variants of the same string (case, leading space, …) |
| | `GET /api/multiplicity/search/{id}` | Search multiplicity groups |
| Language | `GET /api/language/{id}` | Script/language composition of the vocabulary |
| Morphemes | `GET /api/morphemes/{id}` | Morpheme alignment analysis |
| Undertrained | `GET /api/undertrained/{id}` | Likely undertrained or unreachable tokens |
| Comparison | `POST /api/comparison/overlap` | Vocabulary overlap between tokenizers |
| | `POST /api/comparison/tokenize` | Same text through several tokenizers |
| | `POST /api/comparison/efficiency` | Compression/efficiency comparison |
| Merge tree | `POST /api/merge-tree/compare` | Compare BPE merge derivations of a string |
| Merge forest | `GET /api/merge-forest/{id}` | BPE merges as a flat forest |
| | `GET /api/merge-forest/trees/{id}` | Root trees of the forest |
| | `GET /api/merge-forest/subtree/{rank}` | Subtree under a merge rank |
| Intrinsic eval | `GET /api/intrinsic-eval/languages` | Supported FLORES+ languages |
| | `POST /api/intrinsic-eval/{id}/per-text` | Metrics for one text |
| | `POST /api/intrinsic-eval/{id}/flores` | Metrics across FLORES+ languages |
| Visualize | `GET /api/visualize/samples` | Built-in code / math / multilingual samples from `tokenizer-visualize` |
| | `POST /api/visualize` | Token boundaries on source text for up to 8 tokenizers: runs per owning token, sub-character splits, whitespace and indentation stats |
| Sanity check | `POST /api/sanity-check/{id}` | The 16 `tokenizer-sanity-check` checks on the built-in probes, with overall severity and the CLI's exit code |
| Health | `GET /api/health` | Liveness check |

Request and response schemas are in `app/models/schemas.py` and at `/docs`.

## Layout

```
app/
  main.py        FastAPI app, CORS, router registration, SPA static serving
  routers/       One module per API area (thin HTTP layer)
  services/      Analysis logic
    adapter.py   Common interface over tiktoken / HF / SentencePiece / TokenMonster
    registry.py  Name resolution and LRU cache of loaded tokenizers
    tokeval_wrapper.py  Adapter -> tokenizer-intrinsic-evals TokenizerWrapper
                        (tiktoken is converted to an equivalent tokenizers.Tokenizer)
  models/
    schemas.py   Pydantic request/response models
serve.sh         Cluster launch script (HF cache + optional Cloudflare tunnel, supervised)
serve.sbatch     Self-renewing Slurm job around serve.sh
ensure_serving.sh  Idempotent start/watchdog for the serve.sbatch chain
```

To support a new tokenizer library, subclass `TokenizerAdapter` in `adapter.py` and add a resolution rule in `TokenizerRegistry._create_adapter`.
