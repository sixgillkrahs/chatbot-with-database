from pydantic import BaseModel

class InputLangchain(BaseModel):
    userMessage: str