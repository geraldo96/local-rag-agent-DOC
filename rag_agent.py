"""
Local RAG Chatbot Agent — LangGraph + SqliteSaver
==================================================
Stack: LangChain 0.3+ + LangGraph + ChromaDB + LlamaCpp (GGUF Metal) + HuggingFace Embeddings
Memoria persistente su SQLite — la conversazione sopravvive al riavvio
100% offline, nessun Ollama richiesto
"""

import argparse
from pathlib import Path
from typing import List, Annotated
from typing_extensions import TypedDict

# ── Document loaders ──────────────────────────────────────────────────────────
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)

# ── Text splitter ─────────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Vector store + Embeddings ─────────────────────────────────────────────────
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── LLM locale GGUF con Metal ─────────────────────────────────────────────────
from langchain_community.llms import LlamaCpp

# ── Prompt ────────────────────────────────────────────────────────────────────
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# ── Reranker ──────────────────────────────────────────────────────────────────
from sentence_transformers import CrossEncoder

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CHROMA_DB_DIR = "./chroma_db"
SQLITE_PATH   = "./memory.db"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
MODEL_PATH    = "./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

CHUNK_SIZE       = 600
CHUNK_OVERLAP    = 150
TOP_K            = 9     # chunk iniziali da recuperare (poi reranker ne sceglie 3)
TOP_K_RERANKED   = 3     # chunk finali dopo reranking
RERANKER_MODEL   = "BAAI/bge-reranker-base"

# Prompt per riformulare la domanda tenendo conto dello storico
QA_PROMPT = """<|im_start|>system
Sei un assistente esperto che risponde SOLO usando le informazioni fornite nel contesto.
Se la risposta non è nel contesto, dillo chiaramente senza inventare nulla.
Rispondi sempre nella lingua dell'utente.
<|im_end|>
<|im_start|>user
Contesto:
{context}

Domanda: {question}
<|im_end|>
<|im_start|>assistant
"""


# ─────────────────────────────────────────────────────────────────────────────
# STATE — cosa ricorda il grafo ad ogni step
# ─────────────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # storico conversazione
    context:  List[str]                                   # chunk recuperati
    sources:  List[str]                                   # sorgenti documenti


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(docs_dir: str):
    loaders = {
        "**/*.pdf":  (DirectoryLoader, {"loader_cls": PyMuPDFLoader}),
        "**/*.md":   (DirectoryLoader, {"loader_cls": UnstructuredMarkdownLoader}),
        "**/*.txt":  (DirectoryLoader, {"loader_cls": TextLoader, "loader_kwargs": {"encoding": "utf-8"}}),
        "**/*.docx": (DirectoryLoader, {"loader_cls": UnstructuredWordDocumentLoader}),
        "**/*.html": (DirectoryLoader, {"loader_cls": TextLoader, "loader_kwargs": {"encoding": "utf-8"}}),
    }

    all_docs = []
    for glob_pattern, (loader_cls, kwargs) in loaders.items():
        try:
            loader = loader_cls(docs_dir, glob=glob_pattern, **kwargs, show_progress=True, use_multithreading=True)
            docs = loader.load()
            print(f"  ✓ {glob_pattern}: {len(docs)} documento/i")
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ⚠ {glob_pattern}: {e}")

    if not all_docs:
        raise ValueError(f"Nessun documento trovato in: {docs_dir}")

    print(f"\n📄 Totale documenti caricati: {len(all_docs)}")
    return all_docs


def build_vectorstore(docs_dir: str, force_rebuild: bool = False) -> Chroma:
    print(f"🔄 Carico embeddings: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if Path(CHROMA_DB_DIR).exists() and not force_rebuild:
        print(f"🗄  Carico vectorstore esistente da {CHROMA_DB_DIR}")
        return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

    print(f"🔨 Costruisco nuovo vectorstore da: {docs_dir}")
    docs = load_documents(docs_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Chunks creati: {len(chunks)}")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    print(f"✅ Vectorstore salvato in {CHROMA_DB_DIR}")
    return db


# ─────────────────────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────────────────────

class LocalRAGAgent:
    def __init__(self, docs_dir: str, force_rebuild: bool = False):
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"\n❌ Modello GGUF non trovato in: {MODEL_PATH}\n"
                f"Scaricalo con:\n"
                f"  huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF "
                f"--include 'Qwen2.5-7B-Instruct-Q4_K_M.gguf' --local-dir ./models\n"
            )

        # ── Vectorstore ────────────────────────────────────────────────────
        self.db = build_vectorstore(docs_dir, force_rebuild)
        self.retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3},
        )


        # ── LLM ────────────────────────────────────────────────────────────
        print(f"🧠 Carico LLM: {MODEL_PATH}")
        self.llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_gpu_layers=-1,
            temperature=0.1,
            max_tokens=1024,
            repeat_penalty=1.3,
            top_k=40,
            top_p=0.95,
            stop=["<|im_end|>", "<|im_start|>"],
            verbose=False,
        )

        # ── Reranker ───────────────────────────────────────────────────────
        print(f"🔀 Carico reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(
            RERANKER_MODEL,
            device="mps",    # Metal su Apple Silicon
        )

        # ── Prompt ─────────────────────────────────────────────────────────
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=QA_PROMPT,
        )

        # ── Costruisce il grafo LangGraph ───────────────────────────────────
        self.app = self._build_graph()

    def _build_graph(self):
        """Costruisce il grafo LangGraph con memoria persistente su SQLite."""

        llm        = self.llm
        retriever  = self.retriever
        reranker   = self.reranker
        qa_prompt  = self.qa_prompt

        # ── Nodo 1: passa la domanda così com'è ────────────────────────────
        def contextualize(state: RAGState) -> RAGState:
            return {"context": [], "sources": []}

        # ── Nodo 2: recupera i chunk rilevanti dal vectorstore ─────────────
        def retrieve(state: RAGState) -> RAGState:
            question = state["messages"][-1].content
            docs     = retriever.invoke(question)
            context  = [doc.page_content for doc in docs]
            sources  = list({doc.metadata.get("source", "N/A") for doc in docs})
            return {"context": context, "sources": sources}

        # ── Nodo 3: reranking — riordina i chunk per rilevanza reale ───────
        def rerank(state: RAGState) -> RAGState:
            question = state["messages"][-1].content
            chunks   = state["context"]
            sources  = state["sources"]

            if not chunks:
                return {"context": [], "sources": []}

            # Calcola score di rilevanza per ogni chunk
            pairs  = [(question, chunk) for chunk in chunks]
            scores = reranker.predict(pairs)

            # Ordina per score decrescente e prendi i migliori
            ranked = sorted(
                zip(scores, chunks, sources if len(sources) == len(chunks) else [""]*len(chunks)),
                key=lambda x: x[0],
                reverse=True,
            )

            top_chunks  = [chunk  for _, chunk, _      in ranked[:TOP_K_RERANKED]]
            top_sources = [source for _, _,     source in ranked[:TOP_K_RERANKED]]

            print(f"  🔀 Reranking: {len(chunks)} → {len(top_chunks)} chunk")
            return {"context": top_chunks, "sources": top_sources}

        # ── Nodo 4: genera la risposta finale ──────────────────────────────
        def generate(state: RAGState) -> RAGState:
            messages = state["messages"]
            question = messages[-1].content
            context  = "\n\n".join(state["context"]) if state["context"] else "Nessun contesto disponibile."

            chain  = qa_prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
            return {"messages": [AIMessage(content=answer)]}

        # ── Costruzione grafo ──────────────────────────────────────────────
        graph = StateGraph(RAGState)
        graph.add_node("contextualize", contextualize)
        graph.add_node("retrieve",      retrieve)
        graph.add_node("rerank",        rerank)
        graph.add_node("generate",      generate)

        graph.add_edge(START,           "contextualize")
        graph.add_edge("contextualize", "retrieve")
        graph.add_edge("retrieve",      "rerank")
        graph.add_edge("rerank",        "generate")
        graph.add_edge("generate",       END)

        # ── Persistenza SQLite ─────────────────────────────────────────────
        import sqlite3
        conn   = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        memory = SqliteSaver(conn)
        return graph.compile(checkpointer=memory)

    def ask(self, question: str, thread_id: str = "default") -> dict:
        """Pone una domanda. thread_id separa conversazioni diverse."""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.app.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        answer  = result["messages"][-1].content
        sources = result.get("sources", [])
        return {"answer": answer, "sources": sources}

    def reset_memory(self, thread_id: str = "default"):
        """Azzera la memoria di un thread specifico."""
        import sqlite3
        conn = sqlite3.connect(SQLITE_PATH)
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()
        print(f"🔄 Memoria thread '{thread_id}' azzerata.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def run_cli(agent: LocalRAGAgent, thread_id: str):
    print(f"\n🤖 RAG Agent attivo | thread: '{thread_id}'")
    print("   Scrivi 'exit' per uscire, 'reset' per nuova sessione.\n")
    while True:
        try:
            query = input("Tu: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "reset":
            agent.reset_memory(thread_id)
            continue

        result = agent.ask(query, thread_id=thread_id)
        print(f"\n🤖 Risposta:\n{result['answer']}")
        if result["sources"]:
            print(f"\n📚 Sorgenti: {', '.join(result['sources'])}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RAG Chatbot Agent")
    parser.add_argument("--docs",      default="./docs",     help="Directory documenti")
    parser.add_argument("--rebuild",   action="store_true",  help="Forza rebuild vectorstore")
    parser.add_argument("--ui",        action="store_true",  help="Avvia interfaccia Gradio")
    parser.add_argument("--thread",    default="default",    help="ID sessione conversazione")
    args = parser.parse_args()

    agent = LocalRAGAgent(docs_dir=args.docs, force_rebuild=args.rebuild)

    if args.ui:
        from gradio_ui import launch_gradio
        launch_gradio(agent)
    else:
        run_cli(agent, thread_id=args.thread)