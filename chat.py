import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "phi3:mini"  # oder ein anderes kleines Modell, das du geladen hast


def ask_ollama(question: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Du antwortest kurz und auf Deutsch."},
            {"role": "user", "content": question},
        ],
        "stream": False,
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"].strip()


def main():
    print("Lokaler Ollama-Chat. Tippe 'exit' zum Beenden.\n")
    while True:
        q = input("Frage: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Ciao 👋")
            break

        try:
            answer = ask_ollama(q)
            print("\nAntwort:")
            print(answer)
            print("\n" + "-" * 50 + "\n")
        except Exception as e:
            print(f"Fehler bei der Anfrage: {e}")
            break


if __name__ == "__main__":
    main()
