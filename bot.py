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

# ---------------------------
# ENV
# ---------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

# ---------------------------
# Bot init (Aiogram 3.7+)
# ---------------------------
bot = Bot(token=TELEGRAM_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Load lex_base + embeddings
# ---------------------------
with open("lex_base.json", "r", encoding="utf-8") as f:
    LEX_BASE = json.load(f)

VECTORS = {}
if os.path.exists("embeddings/tk_vectors.pkl"):
    import pickle
    with open("embeddings/tk_vectors.pkl", "rb") as f:
        VECTORS = pickle.load(f)

# ---------------------------
# Inline language keyboard
# ---------------------------
INLINE_LANG = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🇷🇺 Русский", callback_data="lang_ru"),
     InlineKeyboardButton(text="🇺🇿 Ўзбекча", callback_data="lang_uz")]
])

user_lang = {}  # user_id -> "ru"/"uz"

# ---------------------------
# utils
# ---------------------------
def cosine(a, b):
    a = np.array(a); b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

def is_short_non_hr(text: str) -> bool:
    # quick heuristic: very short messages (emoji/hi) — treat as non-HR probable
    t = text.strip()
    if len(t) <= 3:
        return True
    # contains only emoji / punctuation
    if re.fullmatch(r"[\W_]+", t):
        return True
    return False

# ---------------------------
# Start handler
# ---------------------------
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("Выберите язык / Tilni tanlang:", reply_markup=INLINE_LANG)

# ---------------------------
# Language picker (remove keyboard after choice)
# ---------------------------
@dp.callback_query(F.data.startswith("lang_"))
async def lang_picker(callback):
    uid = callback.from_user.id

    if callback.data == "lang_ru":
        user_lang[uid] = "ru"
        await callback.message.edit_reply_markup(reply_markup=None)  # remove inline keyboard
        await callback.message.answer("Язык установлен: Русский 🇷🇺")
    else:
        user_lang[uid] = "uz"
        await callback.message.edit_reply_markup(reply_markup=None)
        await callback.message.answer("Til o‘rnatildi: Ўзbekча 🇺🇿")

    await callback.answer()

# ---------------------------
# Classification via GPT: is this HR-related?
# ---------------------------
async def classify_hr(question: str) -> str:
    """
    Return "HR" or "NOT_HR".
    Uses a deterministic prompt (temperature=0).
    """
    try:
        prompt = (
            "Кратко классифицируй: является ли следующий запрос ВОПРОСОМ ПО ТРУДОВОМУ КОДЕКСУ/HR (ответи только 'HR' или 'NOT_HR').\n\n"
            f"Текст: {question}\n\n"
            "Правила: если вопрос явно про увольнение, отпуск, больничный, зарплату, компенсации, дисциплину, "
            "приём/увольнение, сокращение, режимы работы, права работника — ответь HR. "
            "Если это бытовое сообщение (привет, как дела, emoji), личный чат, благодарность или неясный короткий текст — ответь NOT_HR."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты классификатор. Отвечай строго 'HR' или 'NOT_HR'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=8
        )
        out = resp.choices[0].message.content.strip().upper()
        if out.startswith("HR"):
            return "HR"
        return "NOT_HR"
    except Exception as e:
        logging.exception("Classifier error: %s", e)
        # fallback heuristics
        if is_short_non_hr(question):
            return "NOT_HR"
        # fallback: if contains HR keywords -> HR
        keywords = ["увол", "увольн", "отпуск", "больнич", "договор", "труд", "зарп", "компенс", "дисципл", "сокращ", "работник", "сотрудник", "статья", "модда"]
        low = question.lower()
        if any(k in low for k in keywords):
            return "HR"
        return "NOT_HR"

# ---------------------------
# Main message handler
# ---------------------------
@dp.message(F.text)
async def answer_user(message: Message):
    uid = message.from_user.id
    text = message.text.strip()

    if uid not in user_lang:
        await message.answer("Привет! Пожалуйста, выберите язык сначала:", reply_markup=INLINE_LANG)
        return

    lang = user_lang[uid]
    await message.chat.do("typing")

    # Quick non-HR checks
    if is_short_non_hr(text):
        if lang == "ru":
            reply = "Привет! 👋 У меня всё отлично — я бот-помощник по Трудовому кодексу Республики Узбекистан. Задайте вопрос про увольнение, отпуск, больничный, график или права работников — помогу!"
        else:  # uz
            reply = "Салом! 👋 Ҳаммаси жойида — мен Ўзбекистон Меҳнат кодекси бўйича ёрдамчиман. Эҳтирозлар, ишдан бўшатиш, таътил, касаллик ва ходим ҳуқуқлари ҳақида сўрашинг — ёрдам бераман!"
        await message.answer(reply + "\n\nИсточник: https://lex.uz/docs/6257291")
        return

    # Ask GPT classifier whether this is HR-related
    classification = await classify_hr(text)

    if classification == "NOT_HR":
        # Friendly reply (variant 2 style)
        if lang == "ru":
            reply = ("Привет! 👋 У меня всё отлично — я HR-бот по Трудовому кодексу Узбекистана. "
                     "Если у вас есть вопрос по увольнению, отпуску, больничному, трудовым договорам или правам работников — спрашивайте.")
        else:
            reply = ("Салом! 👋 Ҳаммаси жойида — мен Ўзбекистон Меҳнат кодекси бўйича HR-ботман. "
                     "Агар ишдан бўшатиш, таътил, касаллик, меҳнат шартномалари ёки ходим ҳуқуқлари ҳақида савол бўлса — сўранг.")
        await message.answer(reply + "\n\nИсточник: https://lex.uz/docs/6257291")
        return

    # If here => classified as HR question
    # -------------------------------
    # 1) try extract article number (RU or UZ)
    article_id = None
    m1 = re.search(r"стат(ья|и)?\s*(\d+)", text.lower())
    m2 = re.search(r"(\d{1,4})\s*-\s*модда", text.lower())
    if m1:
        article_id = m1.group(2)
    elif m2:
        article_id = m2.group(1)

    # 2) semantic search if no article number
    if not article_id and VECTORS:
        try:
            emb = client.embeddings.create(model="text-embedding-3-small", input=text)
            qvec = np.array(emb.data[0].embedding)
            best_score = -999
            best_aid = None
            for aid, vec in VECTORS.items():
                score = cosine(qvec, np.array(vec))
                if score > best_score:
                    best_score = score
                    best_aid = aid
            # threshold: require some minimal similarity to accept (avoid random match)
            if best_score is not None and best_score >= 0.23:
                article_id = best_aid
            else:
                article_id = None
        except Exception as e:
            logging.exception("Embeddings search error: %s", e)
            article_id = None

    # 3) get article text if found
    article_text = ""
    if article_id and str(article_id) in LEX_BASE:
        article_text = LEX_BASE[str(article_id)].get(lang, "")

    # 4) Prepare strict system/user messages to avoid invented numbers
    system_msg = (
        "Ты — HR-консультант по Трудовому кодексу Республики Узбекистан. "
        "Если тебе передали текст статьи — ОТВЕЧАЙ ИСКЛЮЧИТЕЛЬНО на основе этого текста. "
        "НИКАКИХ других номеров статей не придумывай и не называй. "
        "Если статья не передана, честно скажи: 'Статья не найдена в локальной базе; рекомендую посмотреть на lex.uz' и не придумывай номер. "
        "Отвечай кратко, понятно и давай практические шаги для кадровика."
    )

    if article_text:
        user_msg = (f"Вопрос: {text}\n\n"
                    f"Найдена статья #{article_id} в локальной базе. Используй только этот текст для ответа.\n\n"
                    f"Текст статьи:\n{article_text}\n\n"
                    "Дай краткое понятное объяснение и практические шаги.")
    else:
        user_msg = (f"Вопрос: {text}\n\n"
                    "Статья не найдена в локальной базе. Не придумывай номера статей. Дай общий практический совет и порекомендуй посмотреть https://lex.uz/docs/6257291 для актуальной редакции.")

    # 5) Call GPT for the final answer
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.15,
            max_tokens=700
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        logging.exception("GPT error: %s", e)
        if article_text:
            answer = "Сервис временно недоступен. Ниже — найденная статья:\n\n" + article_text
        else:
            answer = "Сервис временно недоступен. Попробуйте позже."

    # 6) Send response with header & lex link
    footer = "\n\nИсточник и актуальная редакция: https://lex.uz/docs/6257291"
    if article_id and article_text:
        header = f"Найдена статья: {article_id}\n\n"
        await message.answer(header + answer + footer)
    else:
        await message.answer(answer + footer)

    # 7) Logging
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
            (("Найдена статья: " + str(article_id) + "\n\n") if article_id else "") + answer,
            int(article_id) if article_id else None
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.exception("Logging error: %s", e)

# ---------------------------
# Run
# ---------------------------
async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
