# src/bot.py
"""
HR Law Bot (вариант: живой стиль A — как ChatGPT)
Требования:
 - src/law_base.json  (в формате {"157": {"ru": "...", "uz": "..."}, ...})
 - src/embeddings/tk_vectors.pkl  (pickle: dict article_id -> vector)
 - OPENAI_API_KEY и TELEGRAM_TOKEN в окружении
 - Установленные зависимости: aiogram, openai, numpy, python-dotenv (опционально)
"""

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

# -----------------------
# Конфигурация
# -----------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN not set in environment")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

# Пути (работаем внутри /app/ или корня репо)
BASE_DIR = Path(__file__).resolve().parent
LAW_PATH = BASE_DIR / "law_base.json"
EMBED_PATH = BASE_DIR / "embeddings" / "tk_vectors.pkl"

LEX_LINK = "https://lex.uz/docs/6257291"

# -----------------------
# Инициализация
# -----------------------
bot = Bot(token=TELEGRAM_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# Загрузка law_base.json
# -----------------------
if LAW_PATH.exists():
    with open(LAW_PATH, "r", encoding="utf-8") as f:
        try:
            LEX_BASE = json.load(f)
        except Exception as e:
            logging.exception("Не удалось загрузить law_base.json: %s", e)
            LEX_BASE = {}
else:
    logging.warning("law_base.json not found at %s", LAW_PATH)
    LEX_BASE = {}

# -----------------------
# Загрузка embeddings (если есть)
# -----------------------
VECTORS = {}
if EMBED_PATH.exists():
    try:
        import pickle
        with open(EMBED_PATH, "rb") as f:
            VECTORS = pickle.load(f)
    except Exception:
        logging.exception("Не удалось загрузить embeddings; semantic search отключён.")
        VECTORS = {}
else:
    logging.info("Embeddings not found: %s — semantic search disabled.", EMBED_PATH)

# -----------------------
# Утилиты
# -----------------------
def cosine(a, b):
    a = np.array(a); b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

# простой набор ключевых слов для узбекского (латиница)
UZ_KEYWORDS = ["modda", "mehnat", "ishchi", "ish", "mehnat kodeksi", "qonun", "ta'til", "maosh", "ishlash", "bekor"]
# русский ключевые слова (кириллица)
RU_KEYWORDS = ["статья", "труд", "работник", "увольн", "отпуск", "больнич", "договор", "маpшрут", "оклад"]

def detect_language_by_text(text: str):
    text = (text or "").strip()
    if not text:
        return None
    # кириллица -> русский
    if re.search(r"[А-Яа-яЁё]", text):
        return "ru"
    low = text.lower()
    # явные узбекские слова (latin)
    for k in UZ_KEYWORDS:
        if k in low:
            return "uz"
    # латинские буквы и нерусские слова -> предположим uz
    if re.search(r"[A-Za-z]", text):
        if any(k in low for k in ["ish", "modda", "mehnat", "qonun", "salom"]):
            return "uz"
    return None

# распознать явную ссылку на статью
def extract_explicit_article(text: str):
    if not text:
        return None
    low = text.lower()
    m = re.search(r"стат(?:ья|и)?\s*(\d{1,4})", low)
    if m:
        return m.group(1)
    m = re.search(r"(\d{1,4})\s*-\s*modda", low) or re.search(r"(\d{1,4})\s+modda\b", low)
    if m:
        return m.group(1)
    m = re.search(r"modda\s*(\d{1,4})", low)
    if m:
        return m.group(1)
    return None

# -----------------------
# Сохраняем выбор языка в памяти (в рамках сессии)
# -----------------------
USER_LANG = {}  # user_id -> "ru"/"uz"

# -----------------------
# /start
# -----------------------
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Salom! / Привет! 👋\n\n"
        "Я — HR-помощник по Трудовому кодексу Республики Узбекистан.\n"
        "Задай вопрос простым языком (например: «увольнение по инициативе работника», «modda 157» или «mehnat shartnomasini bekor qilish»).\n\n"
        "Agar tilni o‘zgartirmoqchi bo‘lsang — yoz 'uz' yoki 'ru'.\n"
    )

# команда для явной установки языка
@dp.message(F.text & F.regex(r"^(ru|uz)\b", flags=re.IGNORECASE))
async def set_lang_command(message: Message):
    code = message.text.strip().lower().split()[0]
    if code in ("ru", "uz"):
        USER_LANG[message.from_user.id] = code
        if code == "ru":
            await message.answer("Язык выбран: Русский 🇷🇺")
        else:
            await message.answer("Til tanlandi: Oʻzbekcha (latin) 🇺🇿")
    else:
        await message.answer("Напишите 'ru' или 'uz' чтобы выбрать язык.")

# -----------------------
# Классификатор HR / NOT_HR (GPT как детерминированный классификатор)
# -----------------------
async def classify_hr(question: str) -> str:
    # коротко: HR или NOT_HR
    try:
        prompt = (
            "Классифицируй коротко: HR или NOT_HR. HR — вопросы по Трудовому кодексу/трудовым отношениям (увольнение, отпуск, оплата, дисциплина и т.д.). "
            "NOT_HR — приветствия, эмоции, не относящиеся к трудовому праву сообщения.\n\n"
            f"Текст: {question}\n\nОтветь строго 'HR' или 'NOT_HR'."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"Ты — бинарный классификатор. Отвечай 'HR' или 'NOT_HR'."},
                {"role":"user","content":prompt}
            ],
            temperature=0.0,
            max_tokens=6
        )
        out = resp.choices[0].message.content.strip().upper()
        if out.startswith("HR"):
            return "HR"
        return "NOT_HR"
    except Exception:
        low = (question or "").lower()
        if any(x in low for x in ["увол", "отпуск", "бол", "договор", "труд", "работник", "modda", "ish", "mehnat"]):
            return "HR"
        return "NOT_HR"

# -----------------------
# Основной обработчик
# -----------------------
@dp.message(F.text)
async def handle_message(message: Message):
    uid = message.from_user.id
    text = (message.text or "").strip()
    if not text:
        await message.answer("Напишите вопрос по Трудовому кодексу или 'ru'/'uz' для выбора языка.")
        return

    # 1) явная команда выбора языка
    if text.strip().lower() in ("ru", "uz"):
        USER_LANG[uid] = text.strip().lower()
        await message.answer("Язык сохранён.")
        return

    # 2) получаем язык: сначала сохранённый, потом детекция
    lang = USER_LANG.get(uid) or detect_language_by_text(text)

    # 3) если не удалось определить — спросим
    if not lang:
        await message.answer(
            "Не смог определить язык. Напишите 'ru' для русского или 'uz' для узбекского (latin), затем отправьте вопрос снова.\n\n"
            "Tilni aniqlay olmadim. Iltimos, 'ru' yoki 'uz' deb yozing."
        )
        return

    # сохраним выбор
    USER_LANG[uid] = lang

    # 4) короткая классификация HR/NOT_HR
    classification = "HR"
    try:
        classification = await classify_hr(text)
    except Exception:
        classification = "HR"

    if classification == "NOT_HR":
        if lang == "ru":
            await message.answer(
                "Привет! Я HR-бот по Трудовому кодексу. Задайте конкретный вопрос про увольнение, отпуск, больничный или трудовой договор — помогу."
                f"\n\nИсточник: {LEX_LINK}"
            )
        else:
            await message.answer(
                "Salom! Men Mehnat Kodeksi bo‘yicha yordamchiman. Iltimos, aniq savol yozing — men yordam beraman."
                f"\n\nManba: {LEX_LINK}"
            )
        return

    # 5) попытка найти явную статью
    explicit = extract_explicit_article(text)
    found_article = None
    article_text = ""

    if explicit and explicit in LEX_BASE:
        found_article = explicit
        article_text = LEX_BASE.get(found_article, {}).get(lang) or LEX_BASE.get(found_article, {}).get("ru") or ""
    elif explicit:
        # explicit указан, но нет в базе
        found_article = None
        article_text = ""

    # 6) semantic search если нет явной статьи
    if not found_article and VECTORS:
        try:
            emb = client.embeddings.create(model="text-embedding-3-small", input=text)
            qvec = np.array(emb.data[0].embedding)
            best = None
            best_score = -1.0
            for aid, vec in VECTORS.items():
                sc = cosine(qvec, np.array(vec))
                if sc > best_score:
                    best_score = sc
                    best = aid
            # порог — эмпирический, можно подстроить
            if best is not None and best_score >= 0.23:
                found_article = str(best)
                article_text = LEX_BASE.get(found_article, {}).get(lang) or LEX_BASE.get(found_article, {}).get("ru") or ""
        except Exception:
            logging.exception("Ошибка эмбеддингового поиска")

    # 7) составляю prompt для ответа: если статья найдена — даём её + объясняем; если не найдена — даём практический совет + ссылку
    system_msg = (
        "Ты — дружелюбный и практичный HR-консультант по Трудовому кодексу Узбекистана. "
        "Отвечай простым языком, давай понятные шаги для кадровика и сотрудника. "
        "Если есть текст статьи — используй только его и не придумывай дополнительных номеров."
    )

    if found_article and article_text:
        user_msg = (
            f"Язык: {lang}\nВопрос: {text}\n\nНайдена статья #{found_article} (локальная база). Текст статьи:\n{article_text}\n\n"
            "Дай краткий, понятный ответ и практические рекомендации для кадровика."
        )
    else:
        user_msg = (
            f"Язык: {lang}\nВопрос: {text}\n\nСтатья не найдена в локальной базе. Не придумывай номер статьи. "
            "Дай практический, понятный совет и порекомендуй проверить актуальную редакцию на lex.uz."
        )

    # 8) вызов модели
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.2,
            max_tokens=700
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        logging.exception("GPT error: %s", e)
        if found_article and article_text:
            answer = f"Не удалось получить ответ от модели. Но вот найденная статья:\n\n{article_text}"
        else:
            answer = "Сервис временно недоступен. Попробуйте позже."

    # 9) финальный текст: добавляем ссылку на lex.uz
    footer = f"\n\nИсточник: {LEX_LINK}"
    if found_article:
        header = f"Найдена статья: {found_article}\n\n"
        await message.answer(header + answer + footer)
    else:
        await message.answer(answer + footer)

# -----------------------
# Запуск
# -----------------------
async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
