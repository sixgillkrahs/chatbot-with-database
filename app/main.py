from fastapi import FastAPI
from app.routers.langchain import router as langchain_router
from app.routers.handle import router as handle_router

app = FastAPI()
app.include_router(langchain_router)
app.include_router(handle_router)
