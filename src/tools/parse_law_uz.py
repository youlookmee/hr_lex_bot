import re
import json
from pathlib import Path
from docx import Document

BASE_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
OUT_FILE = BASE_DIR / "law" / "law_base_uz.json"

def read_docx(path: Path) -> str:
    doc = Document(path)
    lines = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            lines.append(text)
    return "\n".join(lines)

def parse_uz(text: str) -> dict:
    pattern = re.compile(r"(\d{1,4}\s*-\s*modda\.)", re.IGNORECASE)
    matches = list(pattern.finditer(text))

    articles = {}

    for i, m in enumerate(matches):
        raw = m.group(1)
        art_num = re.findall(r"\d+", raw)[0]
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_text = text[start:end].strip()

        article_text = re.sub(r"\n{2,}", "\n", article_text)
        article_text = article_text.replace("–", "-")

        articles[art_num] = article_text

    return articles

def main():
    uz_doc = TOOLS_DIR / "UZ.doc"
    if not uz_doc.exists():
        print("❌ UZ.doc не найден")
        return

    print("📖 Читаю UZ.doc ...")
    text = read_docx(uz_doc)

    print("✂️ Парсю статьи (UZ латиница) ...")
    articles = parse_uz(text)

    OUT_FILE.parent.mkdir(exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"✅ Готово: {OUT_FILE}")
    print(f"📊 Moddalar: {len(articles)}")

if __name__ == "__main__":
    main()
