# 🤖 Local RAG Chatbot Agent

A fully offline RAG agent for querying local document directories.

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain 0.3+ |
| Graph & Memory | LangGraph + SQLite |
| Vector Store | ChromaDB |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace, local) |
| Reranker | `BAAI/bge-reranker-base` (CrossEncoder) |
| LLM | `Qwen2.5-7B-Instruct-Q4_K_M` (GGUF via llama-cpp-python) |
| GPU Acceleration | Apple Metal (M-series chips) |
| Interface | CLI or Gradio Web UI |

---

## Architecture

```
[Documents] → [Loader] → [Splitter] → [Embeddings] → [ChromaDB]
                                                           │
[User Query] → [Retriever MMR TOP-9] ─────────────────────┘
                        │
                   [Reranker]  ← CrossEncoder scores each chunk
                        │
                  [TOP-3 Chunks]
                        │
                   [LLM Qwen2.5]  ← generates answer using context
                        │
               [LangGraph + SQLite]  ← persists conversation state
                        │
                [Answer + Sources]
```

---

## Requirements

- Python 3.11
- Apple Silicon Mac (M1/M2/M3) — or any machine with sufficient RAM
- ~6 GB free RAM for the 7B model
- ~1 GB disk space for models and index

---

## Installation

### 1. Clone the project and create a virtual environment
```bash
git clone <your-repo>
cd local-rag-agent-DOC

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install llama-cpp-python with Metal support (Apple Silicon)
```bash
# ⚠️ Run this BEFORE pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir
```

> On non-Apple machines, use `CMAKE_ARGS="-DLLAMA_CUDA=on"` for NVIDIA GPU,
> or omit `CMAKE_ARGS` entirely for CPU-only.

### 3. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the GGUF model
```bash
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
  --include "Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
  --local-dir ./models
```

> The embedding model (`BAAI/bge-small-en-v1.5`) and reranker (`BAAI/bge-reranker-base`)
> are downloaded automatically from HuggingFace on first run (~400 MB total),
> then cached locally for fully offline use.

---

## Usage

### Place your documents in the `docs/` folder
```
docs/
├── product_manual.pdf
├── technical_specs.md
├── release_notes.txt
└── guides/
    ├── installation.pdf
    └── configuration.docx
```

Supported formats: PDF, Markdown, TXT, DOCX, HTML

### Launch in CLI mode
```bash
python rag_agent.py --docs ./docs
```

### Launch with Gradio web UI
```bash
python rag_agent.py --docs ./docs --ui
# Open http://localhost:7860
```

### Force index rebuild (after adding new documents)
```bash
python rag_agent.py --docs ./docs --rebuild
```

### Use a named conversation thread
```bash
# Separate conversation histories
python rag_agent.py --docs ./docs --thread work
python rag_agent.py --docs ./docs --thread personal
```

---

## Configurable Parameters (`rag_agent.py`)

```python
EMBED_MODEL      = "BAAI/bge-small-en-v1.5"              # embedding model
RERANKER_MODEL   = "BAAI/bge-reranker-base"               # reranker model
MODEL_PATH       = "./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"  # LLM path
CHUNK_SIZE       = 600     # text chunk size
CHUNK_OVERLAP    = 150     # overlap between chunks
TOP_K            = 9       # chunks retrieved before reranking
TOP_K_RERANKED   = 3       # chunks passed to LLM after reranking
```

### Alternative LLM models

| Model | RAM Required | Download |
|---|---|---|
| `Qwen2.5-7B-Instruct-Q4_K_M` | ~6 GB | `bartowski/Qwen2.5-7B-Instruct-GGUF` |
| `Qwen2.5-3B-Instruct-Q4_K_M` | ~3 GB | `bartowski/Qwen2.5-3B-Instruct-GGUF` |
| `Llama-3.2-3B-Instruct-Q4_K_M` | ~3 GB | `bartowski/Llama-3.2-3B-Instruct-GGUF` |
| `Mistral-7B-Instruct-Q4_K_M` | ~6 GB | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` |

---

## Project Structure

```
local-rag-agent-DOC/
├── rag_agent.py        ← main agent (LangGraph pipeline)
├── gradio_ui.py        ← web interface
├── requirements.txt    ← Python dependencies
├── docs/               ← your documents go here
├── models/             ← GGUF model files
├── chroma_db/          ← vector index (auto-generated)
└── memory.db           ← conversation history (auto-generated)
```

---

## Troubleshooting

**`ModuleNotFoundError`** — make sure the virtual environment is active:
```bash
source .venv/bin/activate
```

**`Context window exceeded`** — reduce `TOP_K` or `CHUNK_SIZE` in `rag_agent.py`

**Model not found** — check that the `.gguf` file is in `./models/` and that `MODEL_PATH` matches the filename exactly

**Documents not indexed** — run with `--rebuild` to regenerate the index after adding new files

**Out of memory** — use a smaller model (3B instead of 7B) or reduce `n_ctx` in the `LlamaCpp` config

---

## Roadmap

- [x] First working version
- [x] LangGraph memory with SQLite persistence
- [x] Reranking with CrossEncoder
- [x] Apple Metal GPU acceleration
- [ ] Solve problems with UI
- [ ] Test the re-ranking with different top-k