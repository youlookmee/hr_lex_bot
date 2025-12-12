from fastapi import FastAPI
import sqlite3
app = FastAPI()
DB="logs.db"
def conn():
    return sqlite3.connect(DB)
@app.get("/")
def index():
    c=conn().cursor()
    c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='logs'")
    if c.fetchone()[0]==0:
        return {"msg":"No logs yet"}
    c=conn().cursor()
    rows=c.execute("SELECT COUNT(*) FROM logs").fetchone()
    total=rows[0]
    top = conn().cursor().execute("SELECT article, COUNT(*) as cnt FROM logs WHERE article IS NOT NULL GROUP BY article ORDER BY cnt DESC LIMIT 10").fetchall()
    return {"total": total, "top_articles": top}
