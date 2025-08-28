import asyncio
import json

from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.schema import HumanMessage, AIMessage

from agent import *
from logger.logger import log_answer

app = FastAPI()

# graph = None

class ChatRequest(BaseModel):
    message: str
    model_name: str

class TextInput(BaseModel):
    text: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    state = {"messages": [HumanMessage(content=request.message)]}
    await init_retriever()
    graph = build_graph(request.model_name)
    result = await graph.ainvoke(state)

    state["messages"].extend(result["messages"])

    ai_response = next(
        (msg.content for msg in reversed(result["messages"]) if isinstance(msg, AIMessage)),
        "Ошибка: нет ответа от модели"
    )
    log_answer(ai_response)
    return {"response": ai_response}
