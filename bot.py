import logging
import os
import json
import re
import sqlite3
import numpy as np
from numpy.linalg import norm

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import CommandStart

from openai import OpenAI
import asyncio


# ----------------------------------
# ENV TOKENS
# ----------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("❗ TELEGRAM_TOKEN не найден в .env")
if not OPENAI_API_KEY:
    raise RuntimeError("❗ OPENAI_API_KEY не найден в .env")


# ----------------------------------
# BOT INIT
# ----------------------------------
bot = Bot(
    token=TELEGRAM_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)

dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------------
# LOAD TK DATABASE + EMBEDDINGS
# ----------------------------------
with open("lex_base.json", "r", encoding="utf-8") as f:
    LEX_BASE = json.load(f)

VECTORS = {}
if os.path.exists("embeddings/tk_vectors.pkl"):
    import pickle
    with open("embeddings/tk_vectors.pkl", "rb") as f:
        VECTORS = pickle.load(f)


# ----------------------------------
# LANG KEYBOARD
# ----------------------------------
INLINE_LANG = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🇷🇺 Русский", callback_data="lang_ru"),
            InlineKeyboardButton(text="🇺🇿 Ўзбекча", callback_data="lang_uz"),
        ]
    ]
)

user_lang = {}  # user_id → "ru" / "uz"


# ----------------------------------
# COSINE SIMILARITY
# ----------------------------------
def cosine(a, b):
    if norm(a) == 0 or norm(b) == 0:
        return -1
    return float(np.dot(a, b) / (norm(a) * norm(b)))


# ----------------------------------
# GPT HR CLASSIFIER
# ----------------------------------
async def classify_hr(text: str) -> str:
    """Возвращает HR или NOT_HR."""
    try:
        comp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content":
                    "Ты — бинарный классификатор. Классифицируй текст только одним словом: HR или NOT_HR.\n"
                    "HR — если вопрос связан с работой, увольнением, отпуском, больничным, зарплатой, "
                    "трудовым договором, Меҳнат кодексом, правами сотрудников.\n"
                    "NOT_HR — если приветствие, эмоции, бытовой текст, смайлы."
                },
                {"role": "user", "content": text}
            ]
        )

        result = comp.choices[0].message.content.strip()
        return "HR" if "HR" in result else "NOT_HR"

    except Exception as e:
        logging.exception("Ошибка классификатора")
        return "HR"  # безопасно — лучше дать HR-ответ, чем лишний привет


# ----------------------------------
# START
# ----------------------------------
@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "Выберите язык / Tilni tanlang:",
        reply_markup=INLINE_LANG
    )


# ----------------------------------
# LANGUAGE PICKER
# ----------------------------------
@dp.callback_query(F.data.startswith("lang_"))
async def lang_picker(callback):
    uid = callback.from_user.id

    # Установка языка
    if callback.data == "lang_ru":
        user_lang[uid] = "ru"
        await callback.message.edit_text("Язык установлен: Русский 🇷🇺")

    elif callback.data == "lang_uz":
        user_lang[uid] = "uz"
        await callback.message.edit_text("Til o‘rnatildi: Ўзбекча 🇺🇿")

    await callback.answer()


# ----------------------------------
# MAIN HANDLER
# ----------------------------------
@dp.message(F.text)
async def answer_user(message: Message):
    uid = message.from_user.id
    text = message.text.strip()

    if uid not in user_lang:
        await message.answer("Выберите язык:", reply_markup=INLINE_LANG)
        return

    lang = user_lang[uid]
    await message.chat.do("typing")

    # -------------------------------------
    # 0. ЯВНОЕ УКАЗАНИЕ СТАТЬИ / МОДДА
    # -------------------------------------
    explicit_article = None

    m1 = re.search(r"стат(ья|и)?\s*(\d+)", text.lower())
    m2 = re.search(r"(\d{1,4})\s*-\s*модда", text.lower())
    m3 = re.search(r"модда\s*(\d{1,4})", text.lower())
    m4 = re.search(r"(\d{1,4})\s*модда", text.lower())

    if m1:
        explicit_article = m1.group(2)
    elif m2:
        explicit_article = m2.group(1)
    elif m3:
        explicit_article = m3.group(1)
    elif m4:
        explicit_article = m4.group(1)

    # -------------------------------------
    # HR CLASSIFICATION
    # -------------------------------------
    if explicit_article:
        classification = "HR"
    else:
        classification = await classify_hr(text)

    # -------------------------------------
    # NOT HR — дружелюбный ответ
    # -------------------------------------
    if classification == "NOT_HR":
        if lang == "ru":
            await message.answer(
                "Привет! 👋 У меня всё отлично — я HR-бот по Трудовому кодексу Узбекистана. "
                "Если у вас есть вопрос по увольнению, отпуску, больничному, трудовым договорам или "
                "правам работников — спрашивайте!\n\n"
                "Источник: https://lex.uz/docs/6257291"
            )
        else:
            await message.answer(
                "Салом! 👋 Ҳаммаси жойида — мен Ўзбекистан Меҳнат кодекси бўйича HR-ботман. "
                "Агар ишдан бўшатиш, таътил, касаллик, меҳнат шартномалари ёки ходим ҳуқуқлари ҳақида савол бўлса — сўранг!\n\n"
                "Манба: https://lex.uz/docs/6257291"
            )
        return

    # -------------------------------------
    # ARTICLE SEARCH
    # -------------------------------------
    article_id = None

    if explicit_article:
        article_id = explicit_article

    if not article_id:
        # Try semantic search
        if VECTORS:
            try:
                emb = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                qvec = np.array(emb.data[0].embedding)

                best_score = -999
                best_id = None
                for aid, vec in VECTORS.items():
                    score = cosine(qvec, np.array(vec))
                    if score > best_score:
                        best_score = score
                        best_id = aid

                # допустимый порог похожести
                if best_score > 0.25:
                    article_id = best_id

            except Exception as e:
                logging.exception("Semantic search error")

    # -------------------------------------
    # ARTICLE TEXT
    # -------------------------------------
    article_text = ""
    if article_id and str(article_id) in LEX_BASE:
        article_text = LEX_BASE[str(article_id)].get(lang, "")

    # -------------------------------------
    # GPT FINAL ANSWER
    # -------------------------------------
    try:
        comp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content":
                    "Ты — HR-консультант по Трудовому кодексу РУз. "
                    "Отвечай коротко, точно и без выдумывания норм. "
                    "Если статья есть — объясняй простыми словами. "
                    "Добавляй только: 'Источник: https://lex.uz/docs/6257291'."
                },
                {
                    "role": "user",
                    "content": f"Вопрос: {text}\n\nСтатья {article_id}:\n{article_text}"
                }
            ]
        )

        answer = comp.choices[0].message.content

    except Exception as e:
        logging.exception("GPT Error")
        answer = "Ошибка GPT. Попробуйте позже."

    # -------------------------------------
    # SEND ANSWER
    # -------------------------------------
    await message.answer(answer)

    # -------------------------------------
    # LOGGING
    # -------------------------------------
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
            text,
            answer,
            int(article_id) if article_id else None
        ))

        conn.commit()
        conn.close()

    except Exception:
        logging.exception("Log error")


# ----------------------------------
# RUN
# ----------------------------------
async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
