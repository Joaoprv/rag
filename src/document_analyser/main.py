from fastapi import FastAPI
from document_analyser.routes import chat, file

app = FastAPI()

app.include_router(file.router)
app.include_router(chat.router)