import os
import json
import re
import pickle
import logging
import numpy as np
from numpy.linalg import norm

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message
from aiogram.filters import CommandStart

from openai import OpenAI
import asyncio

# ================== ENV ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("❌ Не заданы TELEGRAM_TOKEN или OPENAI_API_KEY")

# ================== INIT ==================
bot = Bot(
    token=TELEGRAM_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)

# ================== LOAD BASES ==================
with open("src/law_base_ru.json", encoding="utf-8") as f:
    LAW_RU = json.load(f)

with open("src/law_base_uz.json", encoding="utf-8") as f:
    LAW_UZ = json.load(f)

with open("src/embeddings/embeddings_ru.pkl", "rb") as f:
    VEC_RU = pickle.load(f)

with open("src/embeddings/embeddings_uz.pkl", "rb") as f:
    VEC_UZ = pickle.load(f)

# ================== UTILS ==================
def cosine(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))

def detect_lang(text: str) -> str:
    if re.search(r"[а-яА-Я]", text):
        return "ru"
    return "uz"

def semantic_search(text, vectors):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    q = np.array(emb.data[0].embedding)

    best_id, best_score = None, -1
    for k, v in vectors.items():
        score = cosine(q, np.array(v))
        if score > best_score:
            best_id, best_score = k, score

    return best_id if best_score > 0.25 else None

# ================== HANDLERS ==================
@dp.message(CommandStart())
async def start(msg: Message):
    await msg.answer(
        "👋 Salom / Привет!\n\n"
        "Я HR-бот по Трудовому кодексу Узбекистана.\n"
        "Просто задай вопрос на RU или UZ."
    )

@dp.message(F.text)
async def handle(msg: Message):
    text = msg.text.strip()
    lang = detect_lang(text)

    law = LAW_RU if lang == "ru" else LAW_UZ
    vec = VEC_RU if lang == "ru" else VEC_UZ

    article_id = None

    # Явный номер статьи
    m = re.search(r"(\d+)", text)
    if m and m.group(1) in law:
        article_id = m.group(1)
    else:
        article_id = semantic_search(text, vec)

    article_text = law.get(article_id, "") if article_id else ""

    system_prompt = (
        "Ты HR-консультант по Трудовому кодексу Узбекистана. "
        "Отвечай понятно, по делу, давай практические советы."
    )

    user_prompt = (
        f"Вопрос: {text}\n\n"
        f"Статья: {article_id}\n{article_text}"
        if article_text else text
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    answer = resp.choices[0].message.content.strip()
    answer += "\n\nИсточник: https://lex.uz"

    await msg.answer(answer)

# ================== RUN ==================
async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
