import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = "gpt-4o-mini"

# ============================================
# KONFIGURATION
# ============================================

BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdfs"
INDEX_DIR = BASE_DIR / "index"

EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
METADATA_FILE = INDEX_DIR / "metadata.json"

CHUNK_SIZE = 1000      # Zeichen pro Chunk
CHUNK_OVERLAP = 200    # Überlappung
TOP_K = 3              # Anzahl Kontexte pro Antwort

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi3:mini"  # ggf. anpassen

# Sentence-Transformer Modell (wird einmalig geladen)
emb_model: SentenceTransformer | None = None

MIN_SCORE = 0.40  # Schwellwert für "passt wirklich"
GREETING_PATTERNS = re.compile(r"^(hi|hallo|hey|moin|servus|guten tag)\b", re.IGNORECASE)

# ============================================
# HILFSFUNKTIONEN
# ============================================

def is_greeting(text: str) -> bool:
    return bool(GREETING_PATTERNS.match(text.strip()))

def infer_category(source_file: str) -> str:
    """
    Leitet aus dem Dateinamen eine grobe Kategorie ab.
    Das hilft später beim Filtern (z.B. nur BAS70 durchsuchen).
    """
    name = source_file.lower()
    if name.startswith("bas70"):
        return "beurteilung"
    if name.startswith("bas50"):
        return "sopaed"
    if name.startswith("bas11"):
        return "index"
    return "other"

def init_embedding_model() -> None:
    global emb_model
    if emb_model is None:
        print("[INFO] Lade Embedding-Modell (all-MiniLM-L6-v2) ...")
        emb_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> np.ndarray:
    # Einzelnen Text zu Embedding (normiert)
    assert emb_model is not None
    emb = emb_model.encode([text], normalize_embeddings=True)[0]
    return np.array(emb, dtype=np.float32)


def get_embeddings_batch(texts: List[str]) -> np.ndarray:
    # Batch-Embeddings
    assert emb_model is not None
    embs = emb_model.encode(texts, normalize_embeddings=True)
    return np.array(embs, dtype=np.float32)


def extract_text_from_pdf(path: Path) -> List[Tuple[int, str]]:
    """
    Liest ein PDF und gibt eine Liste von (seitenzahl, text) zurück.
    """
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append((i + 1, text))  # Seiten bei 1 beginnend
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Einfache Chunk-Funktion auf Zeichenbasis mit Überlappung.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def build_index() -> None:
    """
    Lädt PDFs aus PDF_DIR, erzeugt Chunks + Embeddings und speichert sie.
    Für den PoC gerne erstmal nur 1 PDF in den Ordner legen.
    """
    print(f"[INFO] Baue Index aus Ordner: {PDF_DIR}")

    INDEX_DIR.mkdir(exist_ok=True)
    init_embedding_model()

    all_chunks: List[str] = []
    all_metadata: List[Dict[str, Any]] = []

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"Keine PDFs in {PDF_DIR} gefunden.")

    for pdf_path in pdf_files:
        print(f"[INFO] Lese PDF: {pdf_path.name}")
        pages = extract_text_from_pdf(pdf_path)

        for page_number, page_text in pages:
            chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                all_chunks.append(chunk)
                all_metadata.append({
                    "source_file": pdf_path.name,
                    "page": page_number,
                    "preview": chunk[:300].replace("\n", " "),
                    "category": infer_category(pdf_path.name)
                })

    if not all_chunks:
        raise RuntimeError("Keine Text-Chunks erzeugt – sind die PDFs evtl. nur Scans?")

    print(f"[INFO] Erzeuge Embeddings für {len(all_chunks)} Chunks ...")
    embeddings = get_embeddings_batch(all_chunks)

    np.save(EMBEDDINGS_FILE, embeddings)
    with METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] Index gebaut: {embeddings.shape[0]} Chunks.")
    print(f" - Embeddings: {EMBEDDINGS_FILE}")
    print(f" - Metadaten:  {METADATA_FILE}")


def load_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
        raise RuntimeError("Index nicht vorhanden. Bitte zuerst build_index() ausführen.")
    embeddings = np.load(EMBEDDINGS_FILE)
    with METADATA_FILE.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return embeddings, metadata


def retrieve(query: str,
             embeddings: np.ndarray,
             metadata: List[Dict[str, Any]],
             top_k: int = TOP_K,
             allowed_categories: List[str] | None = None) -> List[Dict[str, Any]]:
    """
    Findet die ähnlichsten Chunks über Cosine Similarity.
    Embeddings sind bereits normalisiert → Cosine = Dot-Product.
    Optional können Kategorien gefiltert werden (z.B. nur 'beurteilung').
    """
    query_emb = get_embedding(query)
    scores = embeddings @ query_emb  # (N,)

    # Indizes nach Score absteigend sortieren
    sorted_idx = scores.argsort()[::-1]

    results: List[Dict[str, Any]] = []
    for idx in sorted_idx:
        score = float(scores[idx])
        if score < MIN_SCORE:
            # ab hier wird es nur noch schlechter → abbrechen
            break

        meta = metadata[idx]
        if allowed_categories is not None:
            if meta.get("category") not in allowed_categories:
                continue

        m = meta.copy()
        m["score"] = score
        results.append(m)

        if len(results) >= top_k:
            break

    return results

def build_context(chunks_meta: List[Dict[str, Any]]) -> str:
    """
    Kontext-String für das LLM bauen.
    """
    parts = []
    for m in chunks_meta:
        header = f"[Quelle: {m['source_file']} – Seite {m['page']}]"
        parts.append(header + "\n" + m["preview"])
    return "\n\n".join(parts)


def generate_answer_with_ollama(question: str, context: str) -> str:
    """
    Fragt OpenAI (statt Ollama) mit Frage + Kontext.
    Der Funktionsname bleibt gleich, damit der Rest des Codes nicht geändert werden muss.
    """
    system_prompt = (
    "Du bist ein Assistent für ein schulrechtliches Handbuch.\n"
    "Antworte NUR auf Grundlage des bereitgestellten Kontextes.\n"
    "Wenn der Kontext für eine verlässliche Antwort nicht reicht, "
    "sage klar, dass keine eindeutige Regelung gefunden wurde und spekuliere nicht.\n"
    "Antworte sachlich und kurz auf Deutsch.\n"
    "Wenn nach zuständigen Personen oder Stellen gefragt wird, antworte in Stichpunkten der Form "
    "'Rolle: kurze Beschreibung der Aufgabe', und verwende möglichst die Originalbezeichnungen aus dem Kontext "
    "(z. B. Schulleiter, Direktor, Abteilungsleiter, Regionalschulamt, Beurteilungsbeitrag)."
    )

    user_prompt = (
        f"Frage:\n{question}\n\n"
        f"Kontext aus den Handbuch-PDFs:\n{context}\n\n"
        "Formuliere eine Antwort in eigenen Worten. "
        "Falls der Kontext nicht reicht, sag das deutlich."
    )

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return completion.choices[0].message.content.strip()

def answer_question(query: str,
                    embeddings: np.ndarray,
                    metadata: List[Dict[str, Any]],
                    allowed_categories: List[str] | None = None
                    ) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retriever + LLM zusammengesetzt.
    Optional können Kategorien gefiltert werden (z.B. nur 'beurteilung').
    """
    # 1) Begrüßung abfangen, ohne RAG zu bemühen
    if is_greeting(query):
        return (
            "Hallo! Ich kann Fragen zum BSN-Handbuch beantworten – aktuell vor allem zur Beurteilung "
            "(BAS70) und zum sonderpädagogischen Förderbedarf (BAS50). "
            "Stell mir z.B. eine Frage wie: 'Welche Beurteilungsarten gibt es?'",
            []
        )

    # 2) Retriever mit optionalem Kategorie-Filter
    chunks_meta = retrieve(
        query,
        embeddings,
        metadata,
        TOP_K,
        allowed_categories=allowed_categories,
    )

    if not chunks_meta:
        return (
            "Dazu konnte ich in den eingebundenen Dokumenten keine eindeutig passende Stelle finden. "
            "Vielleicht ist die Frage zu allgemein formuliert oder der Bereich ist noch nicht abgedeckt.",
            []
        )

    # 3) Kontext bauen + LLM befragen
    context = build_context(chunks_meta)
    answer = generate_answer_with_ollama(query, context)
    return answer, chunks_meta

def interactive_chat():
    """
    Konsolen-Chat mit RAG über die PDFs im Ordner.
    """
    init_embedding_model()

    # Index bei Bedarf bauen
    if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
        print("[INFO] Kein Index gefunden – baue neuen Index ...")
        build_index()

    embeddings, metadata = load_index()
    print("\n[READY] RAG-Chat über PDFs ist bereit. Tippe deine Frage (oder 'exit').\n")

    while True:
        q = input("Frage: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Auf wiedersehen!")
            break

        print("\n[INFO] Suche relevante Stellen ...")
        try:
            answer, sources = answer_question(
                q,
                embeddings,
                metadata,
                allowed_categories=["beurteilung"]  # nur BAS70*
            )
        except Exception as e:
            print(f"[FEHLER] {e}")
            break

        print("\nAntwort:")
        print(answer)
        print("\nVerwendete Quellen:")
        for s in sources:
            print(f"- {s['source_file']} (Seite {s['page']}, Score {s['score']:.3f})")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    interactive_chat()
