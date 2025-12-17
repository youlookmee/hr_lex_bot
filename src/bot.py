import os
import re
import json
import logging
import asyncio
from pathlib import Path

import numpy as np
from numpy.linalg import norm

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message
from aiogram.filters import CommandStart

from openai import OpenAI

# =========================
# ENV
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
LAW_DIR = BASE_DIR / "law"

RU_JSON = LAW_DIR / "law_base_ru.json"
UZ_JSON = LAW_DIR / "law_base_uz.json"

RU_EMB = LAW_DIR / "embeddings_ru.pkl"
UZ_EMB = LAW_DIR / "embeddings_uz.pkl"

LEX_LINK = "https://lex.uz/docs/6257291"

# =========================
# BOT
# =========================
bot = Bot(token=TELEGRAM_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# LOAD BASES
# =========================
def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

LAW_RU = load_json(RU_JSON)
LAW_UZ = load_json(UZ_JSON)

# =========================
# LOAD EMBEDDINGS (optional)
# =========================
def load_pkl(path: Path) -> dict:
    if path.exists():
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

EMB_RU = load_pkl(RU_EMB)
EMB_UZ = load_pkl(UZ_EMB)

# =========================
# UTILS
# =========================
def cosine(a, b):
    if norm(a) == 0 or norm(b) == 0:
        return -1.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))

def detect_lang(text: str):
    # кириллица -> ru
    if re.search(r"[А-Яа-яЁё]", text):
        return "ru"
    # узбекская латиница (ключевые слова)
    low = text.lower()
    if any(k in low for k in ["modda", "mehnat", "ish", "shartnoma", "bekor", "ta'til", "maosh"]):
        return "uz"
    # по умолчанию ru
    return "ru"

def extract_article_id(text: str):
    low = text.lower()
    # ru
    m = re.search(r"стат(?:ья|и)?\s*(\d{1,4})", low)
    if m:
        return m.group(1)
    # uz latin
    m = re.search(r"(\d{1,4})\s*-\s*modda|\bmodda\s*(\d{1,4})", low)
    if m:
        return m.group(1) or m.group(2)
    return None

async def semantic_pick(text: str, lang: str):
    # если embeddings нет — выходим
    vectors = EMB_RU if lang == "ru" else EMB_UZ
    if not vectors:
        return None

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    qv = np.array(emb.data[0].embedding)

    best_id, best_score = None, -1.0
    for aid, vec in vectors.items():
        sc = cosine(qv, np.array(vec))
        if sc > best_score:
            best_score, best_id = sc, aid

    # порог можно подкрутить
    return best_id if best_score >= 0.23 else None

# =========================
# HANDLERS
# =========================
@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "👋 Привет! Я HR-помощник по Трудовому кодексу РУз.\n\n"
        "Пиши вопрос на русском или узбекском (латиница).\n"
        "Примеры:\n"
        "• статья 157\n"
        "• 157 modda\n"
        "• увольнение по инициативе работника\n"
        "• mehnat shartnomasini bekor qilish\n\n"
        f"Источник: {LEX_LINK}"
    )

@dp.message(F.text)
async def handle(message: Message):
    text = message.text.strip()
    if not text:
        return

    lang = detect_lang(text)
    law = LAW_RU if lang == "ru" else LAW_UZ

    # 1) явный номер статьи
    aid = extract_article_id(text)
    article_text = None

    if aid and aid in law:
        article_text = law.get(aid)

    # 2) semantic search
    if not article_text:
        try:
            pick = await semantic_pick(text, lang)
            if pick and pick in law:
                aid = pick
                article_text = law.get(pick)
        except Exception:
            logging.exception("Semantic search failed")

    # 3) GPT ответ
    system_msg = (
        "Ты дружелюбный HR-консультант по Трудовому кодексу Узбекистана. "
        "Отвечай просто и по делу. Если есть статья — объясни её смысл и практику."
    )

    if article_text:
        user_msg = (
            f"Язык: {lang}\n"
            f"Вопрос: {text}\n\n"
            f"Текст статьи:\n{article_text}\n\n"
            "Дай краткое объяснение и практический совет."
        )
    else:
        user_msg = (
            f"Язык: {lang}\n"
            f"Вопрос: {text}\n\n"
            "Статья не найдена. Дай практический совет и укажи источник."
        )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        logging.exception("GPT error: %s", e)
        answer = "Сервис временно недоступен. Попробуйте позже."

    # 4) финал
    header = f"Найдена статья: {aid}\n\n" if article_text and aid else ""
    await message.answer(f"{header}{answer}\n\nИсточник: {LEX_LINK}")

# =========================
# RUN
# =========================
async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
