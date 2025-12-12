import re
import json
from pathlib import Path
import textract

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = BASE_DIR / "law_base.json"

def extract_text(path: Path):
    raw = textract.process(str(path))
    return raw.decode("utf-8", errors="ignore")

def split_articles_ru(text):
    pattern = r"(Статья\s+(\d{1,4}))"
    hits = list(re.finditer(pattern, text))
    out = {}
    for i, m in enumerate(hits):
        aid = m.group(2)
        start = m.start()
        end = hits[i+1].start() if i+1 < len(hits) else len(text)
        out[aid] = text[start:end].strip()
    return out

def split_articles_uz(text):
    pattern = r"(\d{1,4}\s*-\s*modda)"
    hits = list(re.finditer(pattern, text, re.IGNORECASE))
    out = {}
    for i, m in enumerate(hits):
        raw = m.group(0)
        aid = re.findall(r"\d+", raw)[0]
        start = m.start()
        end = hits[i+1].start() if i+1 < len(hits) else len(text)
        out[aid] = text[start:end].strip()
    return out

def main():
    ru_path = BASE_DIR / "tools" / "RU.doc"
    uz_path = BASE_DIR / "tools" / "UZ.doc"

    if not ru_path.exists() or not uz_path.exists():
        print("[ERROR] RU.doc or UZ.doc missing in src/tools/")
        return

    print("[INFO] extracting RU...")
    ru_text = extract_text(ru_path)
    print("[INFO] extracting UZ...")
    uz_text = extract_text(uz_path)

    print("[INFO] splitting articles...")
    ru_map = split_articles_ru(ru_text)
    uz_map = split_articles_uz(uz_text)

    all_ids = sorted(set(ru_map.keys()) | set(uz_map.keys()), key=lambda x: int(x))
    result = {}

    for aid in all_ids:
        ru = ru_map.get(aid, "")
        uz = uz_map.get(aid, "")

        result[aid] = {
            "ru": ru,
            "uz": uz
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("[OK] Completed!")
    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
