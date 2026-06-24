from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import GEMINI_KEY

embeddingModel = GoogleGenerativeAIEmbeddings(
    api_key=GEMINI_KEY,
    model="gemini-embedding-2-preview"
)