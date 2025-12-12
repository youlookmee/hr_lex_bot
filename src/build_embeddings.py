import json
import pickle
import os
from pathlib import Path
from openai import OpenAI

# IMPORTANT:
# OpenAI API KEY must be set in system environment
# Example:
#   setx OPENAI_API_KEY "sk-xxxx"
#   export OPENAI_API_KEY="sk-xxxx"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parents[1]
LAW_PATH = BASE_DIR / "law_base.json"
EMB_DIR = BASE_DIR / "embeddings"

def main():
    if not LAW_PATH.exists():
        print(f"[ERROR] law_base.json not found at {LAW_PATH}")
        return

    EMB_DIR.mkdir(exist_ok=True)

    print("[INFO] Loading law_base.json ...")
    with open(LAW_PATH, "r", encoding="utf-8") as f:
        law = json.load(f)

    all_vectors = {}
    json_vectors = {}

    print(f"[INFO] Total articles: {len(law)}")
    print("[INFO] Generating embeddings...\n")

    for article_id, content in law.items():
        ru = content.get("ru", "")
        uz = content.get("uz", "")

        text = f"ID: {article_id}\nRU:\n{ru}\nUZ:\n{uz}"

        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        vec = emb.data[0].embedding

        all_vectors[article_id] = vec
        json_vectors[article_id] = vec

        print(f"[EMBEDDED] {article_id}")

    # Save pickle
    with open(EMB_DIR / "tk_vectors.pkl", "wb") as f:
        pickle.dump(all_vectors, f)

    # Save JSON version
    with open(EMB_DIR / "embeddings.json", "w", encoding="utf-8") as f:
        json.dump(json_vectors, f, ensure_ascii=False)

    print("\n[OK] Embeddings ready!")
    print("Saved:")
    print(f" - {EMB_DIR}/tk_vectors.pkl")
    print(f" - {EMB_DIR}/embeddings.json")

if __name__ == "__main__":
    main()
