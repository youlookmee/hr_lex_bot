import re
import json
from pathlib import Path
from docx import Document

BASE_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
OUT_FILE = BASE_DIR / "law" / "law_base_ru.json"

def read_docx(path: Path) -> str:
    doc = Document(path)
    lines = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            lines.append(text)
    return "\n".join(lines)

def parse_ru(text: str) -> dict:
    pattern = re.compile(r"(Статья\s+(\d{1,4})\.)")
    matches = list(pattern.finditer(text))

    articles = {}

    for i, m in enumerate(matches):
        art_num = m.group(2)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_text = text[start:end].strip()

        article_text = re.sub(r"\n{2,}", "\n", article_text)
        articles[art_num] = article_text

    return articles

def main():
    ru_doc = TOOLS_DIR / "RU.doc"
    if not ru_doc.exists():
        print("❌ RU.doc не найден")
        return

    print("📖 Читаю RU.doc ...")
    text = read_docx(ru_doc)

    print("✂️ Парсю статьи ...")
    articles = parse_ru(text)

    OUT_FILE.parent.mkdir(exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"✅ Готово: {OUT_FILE}")
    print(f"📊 Статей: {len(articles)}")

if __name__ == "__main__":
    main()
