import os
from dotenv import load_dotenv

load_dotenv()

PORT = os.getenv("PORT")
GEMINI_KEY = os.getenv("GEMINI_KEY")
ZOHO_MYSQL_HOST=os.getenv("ZOHO_MYSQL_HOST")
ZOHO_MYSQL_PORT=os.getenv("ZOHO_MYSQL_PORT")
ZOHO_MYSQL_USER=os.getenv("ZOHO_MYSQL_USER")
ZOHO_MYSQL_PASSWORD=os.getenv("ZOHO_MYSQL_PASSWORD")
ZOHO_MYSQL_DB=os.getenv("ZOHO_MYSQL_DB")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")