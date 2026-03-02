# 🤖 Local RAG Chatbot Agent

Agente RAG completamente offline per interrogare directory di documenti locali.

## Stack Tecnico

| Componente | Tecnologia |
|---|---|
| Framework | LangChain |
| Vector Store | ChromaDB |
| Embeddings | `nomic-embed-text` via Ollama |
| LLM | `qwen2.5:7b-instruct-q4_K_M` via Ollama |
| Interfaccia | CLI o Gradio Web UI |

---

## Installazione Rapida

### 1. Installa Ollama
```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows: scarica da https://ollama.com/download
```

### 2. Scarica i modelli
```bash
# Modello LLM (4.5 GB circa)
ollama pull qwen2.5:7b-instruct-q4_K_M

# Modello embeddings (274 MB)
ollama pull nomic-embed-text
```

### 3. Installa le dipendenze Python
```bash
pip install -r requirements.txt
```

---

## Utilizzo

### Metti i tuoi documenti nella cartella `docs/`
```
docs/
├── manuale_prodotto.pdf
├── specifiche_tecniche.md
├── note_rilascio.txt
└── guide/
    ├── installazione.pdf
    └── configurazione.docx
```

### Avvia in modalità CLI
```bash
python rag_agent.py --docs ./docs
```

### Avvia con interfaccia web (Gradio)
```bash
python rag_agent.py --docs ./docs --ui
# Apri http://localhost:7860
```

### Forza ricostruzione indice (dopo aggiunta nuovi documenti)
```bash
python rag_agent.py --docs ./docs --rebuild
```

---

## Parametri Configurabili (`rag_agent.py`)

```python
EMBED_MODEL   = "nomic-embed-text"              # modello embeddings
LLM_MODEL     = "qwen2.5:7b-instruct-q4_K_M"   # modello LLM
CHUNK_SIZE    = 1000    # dimensione chunk testo
CHUNK_OVERLAP = 150     # overlap tra chunk
TOP_K         = 5       # chunk da recuperare per query
```

### Modelli LLM alternativi consigliati

| Modello | RAM Richiesta | Comando |
|---|---|---|
| `qwen2.5:7b-instruct-q4_K_M` | 6 GB | `ollama pull qwen2.5:7b-instruct-q4_K_M` |
| `llama3.2:3b` | 3 GB | `ollama pull llama3.2:3b` |
| `mistral:7b-instruct-q4_K_M` | 6 GB | `ollama pull mistral:7b-instruct-q4_K_M` |
| `phi3.5:mini` | 2.5 GB | `ollama pull phi3.5:mini` |

---

## Alternativa GGUF Diretta (senza Ollama)

Se preferisci caricare un file `.gguf` direttamente:

```bash
pip install llama-cpp-python
```

In `rag_agent.py`, commenta la sezione Ollama e decommenta:
```python
from langchain_community.llms import LlamaCpp

self.llm = LlamaCpp(
    model_path="./models/qwen2.5-7b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=35,   # 0 se solo CPU
    temperature=0.1,
    verbose=False,
)
```

---

## Formati Documenti Supportati

- ✅ PDF
- ✅ Markdown (`.md`)
- ✅ Testo (`.txt`)
- ✅ Word (`.docx`)
- ✅ HTML

---

## Architettura

```
[Documenti] → [Loader] → [Splitter] → [Embeddings] → [ChromaDB]
                                                           │
[Query Utente] → [Retriever MMR] → [TOP-K Chunks] ────────┘
                                        │
                                  [LLM Locale] → [Risposta + Sorgenti]
```

---

## Troubleshooting

**Errore connessione Ollama**: Assicurati che Ollama sia in esecuzione con `ollama serve`

**Out of memory**: Usa un modello più piccolo (es. `phi3.5:mini`) o riduci `n_ctx`

**Documenti non indicizzati**: Esegui con `--rebuild` per rigenerare l'indice
