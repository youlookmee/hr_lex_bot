import logging
import os
import json
import re
import sqlite3
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import CommandStart
from openai import OpenAI
from numpy.linalg import norm
import asyncio


# ---------------------------
#   НАСТРОЙКИ
# ---------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("❗ TELEGRAM_TOKEN не найден в .env")

if not OPENAI_API_KEY:
    raise RuntimeError("❗ OPENAI_API_KEY не найден в .env")

bot = Bot(token=TELEGRAM_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------
#  ЗАГРУЗКА БАЗЫ СТАТЕЙ + EMBEDDINGS
# ---------------------------
with open("lex_base.json", "r", encoding="utf-8") as f:
    LEX_BASE = json.load(f)

VECTORS = {}
if os.path.exists("embeddings/tk_vectors.pkl"):
    import pickle
    with open("embeddings/tk_vectors.pkl", "rb") as f:
        VECTORS = pickle.load(f)


# ---------------------------
#    ЯЗЫКОВЫЕ КНОПКИ
# ---------------------------
INLINE_LANG = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🇷🇺 Русский", callback_data="lang_ru"),
            InlineKeyboardButton(text="🇺🇿 Ўзбекча", callback_data="lang_uz"),
        ]
    ]
)

user_lang = {}  # user_id → "ru" / "uz"


# ---------------------------
#   КОСИНУСНОЕ СХОДСТВО
# ---------------------------
def cosine(a, b):
    if norm(a) == 0 or norm(b) == 0:
        return -1
    return float(np.dot(a, b) / (norm(a) * norm(b)))


# -------------------------------------------------
#   START
# -------------------------------------------------
@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Выберите язык / Tilni tanlang:",
        reply_markup=INLINE_LANG
    )


# -------------------------------------------------
#   ОБРАБОТКА КНОПОК ЯЗЫКА
# -------------------------------------------------
@dp.callback_query(F.data.startswith("lang_"))
async def lang_picker(callback):
    uid = callback.from_user.id

    if callback.data == "lang_ru":
        user_lang[uid] = "ru"
        await callback.message.answer("Язык установлен: Русский 🇷🇺")

    elif callback.data == "lang_uz":
        user_lang[uid] = "uz"
        await callback.message.answer("Til o‘rnatildi: Ўзбекча 🇺🇿")

    await callback.answer()


# -------------------------------------------------
#   ГЛАВНАЯ ЛОГИКА GPT + ПОИСК СТАТЬИ
# -------------------------------------------------
@dp.message(F.text)
async def handle_text(message: Message):
    uid = message.from_user.id
    q = message.text.strip()

    # 1 — Проверка языка
    if uid not in user_lang:
        await message.answer("Выберите язык:", reply_markup=INLINE_LANG)
        return

    lang = user_lang[uid]

    await message.chat.do("typing")

    # 2 — Поиск статьи по номеру
    article_id = None

    # RU → "статья 160"
    m1 = re.search(r"стат(ья|и)?\s*(\d+)", q.lower())

    # UZ → "160-модда"
    m2 = re.search(r"(\d+)\s*-\s*модда", q.lower())

    if m1:
        article_id = m1.group(2)
    elif m2:
        article_id = m2.group(1)

    # 3 — Семантический поиск
    if not article_id and VECTORS:
        try:
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=q
            )
            qvec = np.array(emb.data[0].embedding)

            best_score = -999
            best_id = None

            for aid, vec in VECTORS.items():
                score = cosine(qvec, np.array(vec))
                if score > best_score:
                    best_score = score
                    best_id = aid

            article_id = best_id

        except Exception as e:
            logging.exception("Ошибка embeddings: %s", e)

    # 4 — Получение текста найденной статьи
    article_text = ""

    if article_id and str(article_id) in LEX_BASE:
        article_text = LEX_BASE[str(article_id)].get(lang, "")

    # 5 — GPT отвечает с объяснением
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — HR-консультант по Трудовому кодексу РУз. "
                        "Отвечай коротко, понятно, простым языком. "
                        "Если передана статья — объясни её. "
                        "Всегда добавляй: ⚠️ Ответ носит справочный характер."
                    )
                },
                {
                    "role": "user",
                    "content": f"Вопрос: {q}\n\nТекст статьи:\n{article_text}"
                }
            ],
            temperature=0.25
        )

        answer = completion.choices[0].message.content

    except Exception as e:
        logging.exception("GPT ERROR: %s", e)
        answer = "⚠️ Ошибка GPT. Попробуйте позже."

    # 6 — Отправляем ответ
    await message.answer(answer)

    # 7 — ЛОГИРУЕМ
    try:
        conn = sqlite3.connect("logs.db")
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS logs(
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            username TEXT,
            question TEXT,
            answer TEXT,
            article INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        INSERT INTO logs(user_id, username, question, answer, article)
        VALUES(?,?,?,?,?)
        """, (
            uid,
            message.from_user.username or "",
            q,
            answer,
            int(article_id) if article_id else None
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.exception("Ошибка логов: %s", e)


# -------------------------------------------------
#   ЗАПУСК ПОЛЛИНГА
# -------------------------------------------------
async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
