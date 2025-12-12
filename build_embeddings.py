"""
build_embeddings.py
Run this locally where you have network access and your OPENAI_API_KEY in environment (.env).
This script creates embeddings for each article in lex_base.json and saves embeddings/tk_vectors.pkl
"""
import json, pickle, os
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment (.env) before running this script.")
client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"
LEX_BASE_JSON = "lex_base.json"
OUT_PATH = "embeddings/tk_vectors.pkl"
os.makedirs("embeddings", exist_ok=True)
with open(LEX_BASE_JSON, "r", encoding="utf-8") as f:
    lex = json.load(f)
vectors = {}
for art_id, texts in lex.items():
    text = (texts.get("ru","") + "\\n" + texts.get("uz","")).strip()
    if not text:
        continue
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    emb = resp.data[0].embedding
    vectors[art_id] = emb
    print("Embedded", art_id)
with open(OUT_PATH, "wb") as f:
    pickle.dump(vectors, f)
print("Saved embeddings to", OUT_PATH)
