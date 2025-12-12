import os
from dotenv import load_dotenv
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LEX_BASE_DOC_ID = "6257291"
LEX_BASE_JSON = "lex_base.json"
EMBEDDINGS_PATH = "embeddings/tk_vectors.pkl"
CHAT_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
