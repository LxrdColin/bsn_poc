# bas11_index_chat.py
"""
Experimenteller Index-Bot NUR über BAS11s.
Eigener Index-Ordner, damit der bestehende PoC-Index nicht angefasst wird.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============================================
# KONFIGURATION – EIGENER INDEX-ORDNER
# ============================================

BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdfs"

INDEX_DIR = BASE_DIR / "index_bas11"          # ← anderer Ordner!
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
METADATA_FILE = INDEX_DIR / "metadata.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
MIN_SCORE = 0.40

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = "gpt-4o-mini"

emb_model: SentenceTransformer | None = None

# nur BAS11s
BAS11_NAME = "BAS11s.pdf"

INDEX_PATTERNS = re.compile(
    r"(wo finde ich|in welchem dokument|in welchen dokumenten|"
    r"wo steht|welche vorschrift|welches dokument)",
    re.IGNORECASE,
)

# ============================================
# Hilfsfunktionen
# ============================================

def init_embedding_model() -> None:
    global emb_model
    if emb_model is None:
        print("[INFO] Lade Embedding-Modell (all-MiniLM-L6-v2) ...")
        emb_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings_batch(texts: List[str]) -> np.ndarray:
    assert emb_model is not None
    embs = emb_model.encode(texts, normalize_embeddings=True)
    return np.array(embs, dtype=np.float32)

def get_embedding(text: str) -> np.ndarray:
    assert emb_model is not None
    emb = emb_model.encode([text], normalize_embeddings=True)[0]
    return np.array(emb, dtype=np.float32)

def extract_text_from_pdf(path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append((i + 1, text))  # physische Seite
    return pages

def extract_logical_page(page_text: str, default: int) -> int:
    """
    Versucht 'Seite X' im Text zu finden.
    Fällt sonst auf die physische Seitennummer zurück.
    """
    m = re.search(r"Seite\s+(\d+)", page_text)
    if m:
        return int(m.group(1))
    return default

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# ============================================
# Index-Bau NUR für BAS11s
# ============================================

def build_index_bas11() -> None:
    print(f"[INFO] Baue BAS11-Index aus {PDF_DIR}")
    INDEX_DIR.mkdir(exist_ok=True)
    init_embedding_model()

    pdf_path = PDF_DIR / BAS11_NAME
    if not pdf_path.exists():
        raise RuntimeError(f"{BAS11_NAME} nicht im Ordner {PDF_DIR} gefunden.")

    all_chunks: List[str] = []
    all_metadata: List[Dict[str, Any]] = []

    pages = extract_text_from_pdf(pdf_path)
    for phys_page, page_text in pages:
        logical_page = extract_logical_page(page_text, phys_page)
        chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            all_chunks.append(chunk)
            all_metadata.append({
                "source_file": BAS11_NAME,
                "phys_page": phys_page,
                "page": logical_page,  # die im Dokument angegebene Seite
                "preview": chunk[:300].replace("\n", " ")
            })

    if not all_chunks:
        raise RuntimeError("Keine Text-Chunks erzeugt – ist BAS11s evtl. nur ein Scan?")

    print(f"[INFO] Erzeuge Embeddings für {len(all_chunks)} Chunks ...")
    embeddings = get_embeddings_batch(all_chunks)

    np.save(EMBEDDINGS_FILE, embeddings)
    with METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] BAS11-Index gebaut: {embeddings.shape[0]} Chunks.")
    print(f" - Embeddings: {EMBEDDINGS_FILE}")
    print(f" - Metadaten:  {METADATA_FILE}")

def load_index_bas11() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
        raise RuntimeError("BAS11-Index nicht vorhanden. Bitte build_index_bas11() ausführen.")
    embeddings = np.load(EMBEDDINGS_FILE)
    with METADATA_FILE.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return embeddings, metadata

# ============================================
# Retrieval + Antwort
# ============================================

def retrieve_index(query: str,
                   embeddings: np.ndarray,
                   metadata: List[Dict[str, Any]],
                   top_k: int = TOP_K) -> List[Dict[str, Any]]:
    query_emb = get_embedding(query)
    scores = embeddings @ query_emb
    sorted_idx = scores.argsort()[::-1]

    results = []
    for idx in sorted_idx:
        score = float(scores[idx])
        if score < MIN_SCORE:
            break
        m = metadata[idx].copy()
        m["score"] = score
        results.append(m)
        if len(results) >= top_k:
            break
    return results

def build_context(chunks_meta: List[Dict[str, Any]]) -> str:
    parts = []
    for m in chunks_meta:
        header = f"[Quelle: {m['source_file']} – Seite {m['page']}]"
        parts.append(header + "\n" + m["preview"])
    return "\n\n".join(parts)

def generate_index_answer(question: str, context: str) -> str:
    system_prompt = (
        "Du bist ein Assistent für das Inhaltsverzeichnis des BSN-Handbuchs (BAS11s).\n"
        "Du sollst NICHT den Inhalt der Vorschriften erklären, sondern nur sagen,\n"
        "in WELCHEN Dokumenten bestimmte Regelungen zu finden sind.\n"
        "Antworte NUR auf Grundlage des bereitgestellten Kontextes.\n"
        "Formatiere deine Antwort so:\n"
        "- Dokument: BASxxxs – kurzer Titel (ggf. Teil/Abschnitt, falls im Kontext genannt)\n"
        "Erfinde keine Dokumente."
    )
    user_prompt = (
        f"Frage:\n{question}\n\n"
        f"Ausschnitte aus BAS11s:\n{context}\n\n"
        "Liste alle passenden Dokumente (BAS-Codes) mit kurzem Titel auf. "
        "Wenn du nichts Passendes findest, sag das ausdrücklich."
    )

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content.strip()

def answer_index_question(query: str,
                          embeddings: np.ndarray,
                          metadata: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if not INDEX_PATTERNS.search(query):
        return (
            "Dieser Index-Chat sagt dir, in WELCHEN Dokumenten etwas steht "
            "(z.B. 'In welchem Dokument finde ich die Regelungen zur Beurteilung von Lehrkräften?').\n"
            "Für inhaltliche Fragen nutze den normalen BSN-Chat.",
            []
        )

    chunks_meta = retrieve_index(query, embeddings, metadata, TOP_K)
    if not chunks_meta:
        return (
            "Im Inhaltsverzeichnis (BAS11s) konnte ich dazu keinen passenden Eintrag finden.",
            []
        )

    context = build_context(chunks_meta)
    answer = generate_index_answer(query, context)
    return answer, chunks_meta

# ============================================
# Interaktiver CLI-Chat
# ============================================

def interactive_index_chat():
    init_embedding_model()

    if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
        print("[INFO] Kein BAS11-Index gefunden – baue neuen ...")
        build_index_bas11()

    embeddings, metadata = load_index_bas11()
    print("\n[READY] BAS11-Index-Chat ist bereit.")
    print("Beispiele: 'In welchem Dokument finde ich die Regelungen zur Beurteilung von Lehrkräften?'\n")

    while True:
        q = input("Index-Frage: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Tschau 👋")
            break

        print("\n[INFO] Suche im Inhaltsverzeichnis ...")
        try:
            answer, sources = answer_index_question(q, embeddings, metadata)
        except Exception as e:
            print(f"[FEHLER] {e}")
            break

        print("\nAntwort:")
        print(answer)
        print("\nVerwendete Quellen:")
        for s in sources:
            print(
                f"- {s['source_file']} (physisch Seite {s['phys_page']}, "
                f"Dokumentseite {s['page']}, Score {s['score']:.3f})"
            )

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    interactive_index_chat()
