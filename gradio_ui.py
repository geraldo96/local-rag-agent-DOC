"""
Interfaccia Gradio per il Local RAG Agent
Avvia con: python rag_agent.py --docs ./docs --ui
"""

import gradio as gr
from pathlib import Path


CUSTOM_CSS = """
:root {
    --bg: #0a0f0a;
    --surface: #111a11;
    --border: #1e321e;
    --accent: #2d6a2d;
    --accent-bright: #4a9e4a;
    --accent2: #7bc67b;
    --text: #e0ede0;
    --muted: #7a9e7a;
    --font: 'JetBrains Mono', 'Fira Code', monospace;
}
body, .gradio-container { background: var(--bg) !important; font-family: var(--font); }
.chat-window { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
.message.user { background: linear-gradient(135deg, var(--accent), var(--accent-bright)) !important; border-radius: 12px 12px 2px 12px !important; }
.message.bot { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px 12px 12px 2px !important; }
.source-box { background: #070d07 !important; border: 1px solid var(--accent2) !important; border-radius: 8px !important; color: var(--accent2) !important; font-size: 0.8rem !important; }
button.primary { background: linear-gradient(135deg, var(--accent), var(--accent-bright)) !important; border: none !important; border-radius: 8px !important; color: white !important; font-weight: 600 !important; }
button.secondary { background: transparent !important; border: 1px solid var(--border) !important; border-radius: 8px !important; color: var(--muted) !important; }
input, textarea { background: var(--surface) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 8px !important; font-family: var(--font) !important; }
label { color: var(--muted) !important; font-size: 0.8rem !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; }
"""


def launch_gradio(agent):

    THREAD_ID = "gradio-session"

    def chat(message, history):
        if not message.strip():
            return history, "", ""
        try:
            result = agent.ask(message, thread_id=THREAD_ID)
            answer = result["answer"]
            sources = result["sources"]
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            sources_text = "\n".join(f"📄 {s}" for s in sources) if sources else "—"
        except Exception as e:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"❌ Errore: {e}"})
            sources_text = "—"
        return history, "", sources_text

    def reset_chat():
        agent.reset_memory(thread_id=THREAD_ID)
        return [], "", "Memoria azzerata ✓"

    def rebuild_index(docs_path):
        if not Path(docs_path).exists():
            return "❌ Directory non trovata."
        try:
            agent.db = None
            from rag_agent import build_vectorstore
            agent.db = build_vectorstore(docs_path, force_rebuild=True)
            agent.retriever = agent.db.as_retriever(
                search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15}
            )
            return f"✅ Indice ricostruito da: {docs_path}"
        except Exception as e:
            return f"❌ Errore: {e}"

    with gr.Blocks(title="Local RAG Agent") as demo:

        gr.HTML("""
        <div style="text-align:center; padding: 2rem 0 1rem;">
            <h1 style="font-family: 'JetBrains Mono', monospace; font-size: 1.8rem;
                       background: linear-gradient(135deg, #4a9e4a, #7bc67b);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       margin: 0; letter-spacing: -0.02em;">
                ◈ Local RAG Agent
            </h1>
            <p style="color: #7a9e7a; font-size: 0.85rem; margin-top: 0.5rem; font-family: monospace;">
                Completamente offline · Modello quantizzato · Documenti locali
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_classes=["chat-window"],
                    height=500,
                    show_label=False,
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Fai una domanda sui tuoi documenti...",
                        show_label=False,
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("Invia ▶", variant="primary", scale=1)

            with gr.Column(scale=1):
                gr.HTML('<p style="color:#7a9e7a; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Sorgenti recuperate</p>')
                sources_box = gr.Textbox(
                    value="—",
                    show_label=False,
                    lines=8,
                    interactive=False,
                    elem_classes=["source-box"],
                )

                gr.HTML('<hr style="border-color:#1e321e; margin: 1rem 0;">')
                gr.HTML('<p style="color:#7a9e7a; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Gestione indice</p>')

                docs_input = gr.Textbox(value="./docs", label="Directory documenti", lines=1)
                rebuild_btn = gr.Button("🔄 Ricostruisci indice", variant="secondary")
                rebuild_status = gr.Textbox(show_label=False, lines=2, interactive=False)

                gr.HTML('<hr style="border-color:#1e321e; margin: 1rem 0;">')
                reset_btn = gr.Button("🗑 Reset conversazione", variant="secondary")

        # ── Event bindings ─────────────────────────────────────────────────
        send_btn.click(chat, [msg_box, chatbot], [chatbot, msg_box, sources_box])
        msg_box.submit(chat, [msg_box, chatbot], [chatbot, msg_box, sources_box])
        reset_btn.click(reset_chat, outputs=[chatbot, msg_box, sources_box])
        rebuild_btn.click(rebuild_index, inputs=[docs_input], outputs=[rebuild_status])

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)