import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from openai import OpenAI

# ----------------------------------
# CONFIG
# ----------------------------------
MODEL = "text-embedding-3-small"

BASE_DIR = "src"
OUT_DIR = "src/embeddings"

FILES = {
    "ru": "law_base_ru.json",
    "uz": "law_base_uz.json",
}

os.makedirs(OUT_DIR, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------------
# BUILD EMBEDDINGS
# ----------------------------------
def build(lang: str, filename: str):
    print(f"\n🔹 Обрабатываю {lang.upper()} ({filename})")

    path = os.path.join(BASE_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vectors = {}

    for article_id, text in tqdm(data.items()):
        if not text or not isinstance(text, str):
            continue

        emb = client.embeddings.create(
            model=MODEL,
            input=text
        )

        vectors[article_id] = np.array(
            emb.data[0].embedding,
            dtype=np.float32
        )

    out_file = os.path.join(OUT_DIR, f"embeddings_{lang}.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(vectors, f)

    print(f"✅ Сохранено: {out_file}")
    print(f"📦 Статей: {len(vectors)}")


if __name__ == "__main__":
    for lang, file in FILES.items():
        build(lang, file)

    print("\n🎉 ГОТОВО. Embeddings созданы.")
