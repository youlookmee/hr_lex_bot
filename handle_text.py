@dp.message(F.text)
async def handle_text(message: Message):
    uid = message.from_user.id
    lang = user_lang.get(uid)

    if not lang:
        await message.answer("Выберите язык:", reply_markup=INLINE_LANG)
        return

    question = message.text.strip()
    await message.chat.do("typing")

    # ---------- 1. Поиск статьи по номеру ----------
    article = None

    # RU вариант: "статья 160"
    m = re.search(r"стат(ья|и)?\s*(\d+)", question.lower())
    # UZ вариант: "160-модда"
    u = re.search(r"(\d+)\s*-\s*модда", question.lower())

    if m:
        article = m.group(2)
    elif u:
        article = u.group(1)

    # ---------- 2. Семантический поиск ----------
    if not article and VECTORS and client:
        try:
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=question
            )
            qvec = np.array(emb.data[0].embedding)

            best_id = None
            best_score = -999

            for aid, vec in VECTORS.items():
                score = cosine(qvec, np.array(vec))
                if score > best_score:
                    best_score = score
                    best_id = aid

            article = best_id
        except Exception as e:
            logging.exception("Ошибка embeddings: %s", e)

    # ---------- 3. Получаем текст статьи ----------
    article_text = ""
    if article and article in LEX_BASE:
        article_text = LEX_BASE[article].get(lang, "")

    # ---------- 4. GPT отвечает пользователю ----------
    gpt_answer = ""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — консультант по Трудовому кодексу РУз. "
                        "Отвечай простым языком. "
                        "Если найдена статья, объясни её коротко. "
                        "В конце ВСЕГДА добавляй: ⚠️ Ответ носит справочный характер."
                    )
                },
                {
                    "role": "user",
                    "content": f"Вопрос: {question}\n\nТекст статьи:\n{article_text}"
                }
            ],
            temperature=0.2
        )

        gpt_answer = completion.choices[0].message.content

    except Exception as e:
        logging.exception("GPT ошибка: %s", e)
        gpt_answer = "⚠️ Ошибка GPT. Попробуйте позже."

    # ---------- 5. Отправляем пользователю ----------
    await message.answer(gpt_answer)

    # ---------- 6. Логирование ----------
    try:
        import sqlite3
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
        """, (uid, message.from_user.username or "", question, gpt_answer, int(article) if article else None))

        conn.commit()
        conn.close()
    except:
        pass
