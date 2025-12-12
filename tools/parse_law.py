#!/usr/bin/env python3
# parse_law.py
import re
import sys
import json
import time
import argparse
from pathlib import Path

try:
    from docx import Document
    HAVE_DOCX = True
except Exception:
    HAVE_DOCX = False

try:
    import textract
    HAVE_TEXTRACT = True
except Exception:
    HAVE_TEXTRACT = False

RESET = ""
BOLD = ""
GREEN = ""
YELLOW = ""
RED = ""
BLUE = ""
try:
    from colorama import init as _col_init, Fore, Style
    _col_init(autoreset=True)
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED = Fore.RED
    BLUE = Fore.CYAN
except Exception:
    RESET = BOLD = GREEN = YELLOW = RED = BLUE = ""

def log_info(msg):
    print(f"{BLUE}[INFO]{RESET} {msg}")
def log_ok(msg):
    print(f"{GREEN}[OK]{RESET} {msg}")
def log_warn(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")
def log_err(msg):
    print(f"{RED}[ERR]{RESET} {msg}")

def extract_text_from_docx(path: Path) -> str:
    if not HAVE_DOCX:
        raise RuntimeError("python-docx not installed. Install with: pip install python-docx")
    doc = Document(str(path))
    paragraphs = []
    for p in doc.paragraphs:
        paragraphs.append(p.text)
    return "\n".join(paragraphs)

def extract_text_generic(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".docx", ".docm", ".dotx"]:
        return extract_text_from_docx(path)
    if suffix == ".doc":
        if HAVE_TEXTRACT:
            return textract.process(str(path)).decode("utf-8", errors="ignore")
        else:
            raise RuntimeError("To read .doc files install textract (pip install textract) or convert to .docx")
    return path.read_text(encoding="utf-8", errors="ignore")

def clean_text_block(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = "\n".join([ln.strip() for ln in s.splitlines()])
    return s.strip()

def find_article_markers_ru(text: str):
    pattern = re.compile(r"(Статья\s+(\d{1,4})\.)", flags=re.IGNORECASE)
    return list(pattern.finditer(text))

def find_article_markers_uz(text: str):
    pattern = re.compile(r"((\d{1,4})\s*-\s*modda\.)", flags=re.IGNORECASE)
    return list(pattern.finditer(text))

def split_by_markers(full_text: str, markers):
    markers_sorted = sorted(markers, key=lambda m: m.start())
    result = {}
    for i, m in enumerate(markers_sorted):
        num = m.group(2)
        start = m.start()
        end = markers_sorted[i+1].start() if i+1 < len(markers_sorted) else len(full_text)
        seg = full_text[start:end].strip()
        result[num] = seg
    return result

def split_header_body(seg: str, ru=True):
    lines = seg.splitlines()
    if not lines:
        return "", ""
    header = lines[0].strip()
    body = "\n".join(lines[1:]).strip()
    if len(header) < 5 and len(lines) > 1:
        header = (lines[0] + " " + lines[1]).strip()
        body = "\n".join(lines[2:]).strip()
    return clean_text_block(header), clean_text_block(body)

def build_law_base(rutext: str, uztext: str, expected_count: int = None):
    log_info("Finding article markers in RU text...")
    ru_markers = find_article_markers_ru(rutext)
    log_info(f"Found {len(ru_markers)} RU markers (Статья).")

    log_info("Finding article markers in UZ text...")
    uz_markers = find_article_markers_uz(uztext)
    log_info(f"Found {len(uz_markers)} UZ markers (modda).")

    ru_map = split_by_markers(rutext, ru_markers)
    uz_map = split_by_markers(uztext, uz_markers)

    if expected_count:
        log_info(f"Expected count: {expected_count} articles.")
    log_info(f"RU parsed: {len(ru_map)} articles; UZ parsed: {len(uz_map)} articles")

    all_nums = sorted(set([int(x) for x in list(ru_map.keys()) + list(uz_map.keys())]))
    base = {}
    errors = []
    for n in all_nums:
        nk = str(n)
        ru_seg = ru_map.get(nk, "")
        uz_seg = uz_map.get(nk, "")
        ru_header, ru_body = split_header_body(ru_seg, ru=True) if ru_seg else ("", "")
        uz_header, uz_body = split_header_body(uz_seg, ru=False) if uz_seg else ("", "")
        if not ru_seg:
            errors.append(f"Missing RU article {nk}")
        if not uz_seg:
            errors.append(f"Missing UZ article {nk}")
        ru_text = (ru_header + "\n\n" + ru_body).strip() if ru_header or ru_body else ""
        uz_text = (uz_header + "\n\n" + uz_body).strip() if uz_header or uz_body else ""
        base[nk] = {"ru": clean_text_block(ru_text), "uz": clean_text_block(uz_text)}
    return base, errors

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    log_ok(f"Saved JSON to {path}")

def log_progress_example(base, sample_ids=[1,2,160,581]):
    log_info("Sample articles preview (short):")
    for k in sample_ids:
        sk = str(k)
        if sk in base:
            ru_snip = (base[sk]["ru"].replace("\n", " ")[:200] + "...") if base[sk]["ru"] else "(empty)"
            uz_snip = (base[sk]["uz"].replace("\n", " ")[:200] + "...") if base[sk]["uz"] else "(empty)"
            print(f"  [{sk}] RU: {ru_snip}")
            print(f"       UZ: {uz_snip}")
        else:
            print(f"  [{sk}] MISSING")

def main():
    parser = argparse.ArgumentParser(prog="parse_law.py", description="Parse RU.doc and UZ.doc into law_base.json")
    parser.add_argument("--ru", required=True, help="Path to RU.doc or RU.docx")
    parser.add_argument("--uz", required=True, help="Path to UZ.doc or UZ.docx (latin)")
    parser.add_argument("--out", default="law_base.json", help="Output JSON file")
    parser.add_argument("--expected", type=int, default=581, help="Expected number of articles (default 581)")
    parser.add_argument("--log", default="parse_log.txt", help="Log file path")
    args = parser.parse_args()

    start_time = time.time()
    log_info(f"Starting parse: RU={args.ru}, UZ={args.uz}")

    ru_path = Path(args.ru)
    uz_path = Path(args.uz)
    if not ru_path.exists() or not uz_path.exists():
        log_err("Input files not found. Please provide correct paths.")
        sys.exit(2)

    try:
        log_info(f"Extracting RU text from {ru_path} ...")
        ru_text = extract_text_generic(ru_path)
        ru_text = clean_text_block(ru_text)
        log_ok(f"Extracted RU text (len {len(ru_text)} chars).")
    except Exception as e:
        log_err(f"Failed to extract RU text: {e}")
        raise

    try:
        log_info(f"Extracting UZ text from {uz_path} ...")
        uz_text = extract_text_generic(uz_path)
        uz_text = clean_text_block(uz_text)
        log_ok(f"Extracted UZ text (len {len(uz_text)} chars).")
    except Exception as e:
        log_err(f"Failed to extract UZ text: {e}")
        raise

    log_info("Building combined law base...")
    base, errors = build_law_base(ru_text, uz_text, expected_count=args.expected)

    ru_count = sum(1 for k,v in base.items() if v["ru"])
    uz_count = sum(1 for k,v in base.items() if v["uz"])
    log_info(f"Parsed articles with RU text: {ru_count}, with UZ text: {uz_count}")

    if errors:
        log_warn("Some mismatches were detected:")
        for e in errors[:20]:
            log_warn("  " + e)
        log_warn(f"Total mismatches: {len(errors)} (first 20 shown)")

    out_path = Path(args.out)
    save_json(base, out_path)

    log_progress_example(base, sample_ids=[1,2,160,581])

    elapsed = time.time() - start_time
    log_ok(f"Done in {elapsed:.1f}s. Output: {out_path}. Log -> {args.log}")

    with open(args.log, "w", encoding="utf-8") as lf:
        lf.write(f"parse finished. ru_count={ru_count} uz_count={uz_count} errors={len(errors)}\n")
        if errors:
            for e in errors:
                lf.write(e + "\n")

if __name__ == "__main__":
    main()
