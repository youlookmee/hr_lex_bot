#!/usr/bin/env python3
# build_embeddings.py
import os
import json
import pickle
from pathlib import Path
from openai import OpenAI
import numpy as np

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment or .env")

client = OpenAI(api_key=OPENAI_API_KEY)

BASE_PATH = Path(".")
infile = BASE_PATH / "src" / "law_base.json"
out_dir = BASE_PATH / "src" / "embeddings"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "tk_vectors.pkl"

with open(infile, "r", encoding="utf-8") as f:
    base = json.load(f)

vectors = {}
count = 0
for aid, texts in base.items():
    ru = texts.get("ru","")
    uz = texts.get("uz","")
    text_for_emb = (ru + "\n\n" + uz).strip()
    if not text_for_emb:
        continue
    resp = client.embeddings.create(model="text-embedding-3-small", input=text_for_emb)
    vec = resp.data[0].embedding
    vectors[aid] = vec
    count += 1
    if count % 50 == 0:
        print(f"Embedded {count}")

with open(out_file, "wb") as f:
    pickle.dump(vectors, f)

print(f"Saved {len(vectors)} embeddings to {out_file}")
