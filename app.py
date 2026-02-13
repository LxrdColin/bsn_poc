import gradio as gr
from rag_chat import (
    init_embedding_model,
    build_index,
    load_index,
    EMBEDDINGS_FILE,
    METADATA_FILE,
    answer_question,
)

# Backend beim Start vorbereiten
init_embedding_model()
if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
    build_index()
embeddings, metadata = load_index()


def chat_fn(message, history):
    # in app.py
    def chat_fn(message, history):
        answer, sources = answer_question(
            message,
            embeddings,
            metadata,
            allowed_categories=["beurteilung"]  # Demo: nur Beurteilung
        )

    try:
        answer, sources = answer_question(message, embeddings, metadata)
    except Exception as e:
        # Falls OpenAI-Key fehlt o.ä.
        return f"Im Backend ist ein Fehler aufgetreten: {e}"

    if sources:
        src_lines = "\n\nQuellen:\n" + "\n".join(
            f"- {s['source_file']} (Seite {s['page']})"
            for s in sources
        )
        answer = answer + src_lines

    return answer

demo = gr.ChatInterface(
    fn=chat_fn,
    title="BSN Handbuch – PoC Bot",
    description="Stelle Fragen zur Beurteilung (BAS70) oder zum sonderpädagogischen Förderbedarf (BAS50).",
)

if __name__ == "__main__":
    demo.launch()
