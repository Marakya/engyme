import asyncio
import json
import datetime
from functools import partial

# LangGraph
from langgraph.graph import StateGraph, MessagesState, START, END

# LangChain чат и инструменты
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import tool  # если нужен декоратор tool для агентов

import json

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

import asyncio

from dotenv import load_dotenv
import os

from logger.logger import log_rag_interaction, log_model


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


async def get_retriever(collection_name: str = "collection",  k: int = 5):
    """
    Асинхронно возвращает ретривер для Chroma.
    """

    embedding = await asyncio.to_thread(OpenAIEmbeddings, api_key=OPENAI_API_KEY)
    vector_store = await asyncio.to_thread(
        Chroma,
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory="./chroma_db"
    )
    return vector_store.as_retriever(search_kwargs={"k": k})

# retriever = Nones

async def init_retriever():
    global retriever
    retriever = await get_retriever()


@tool("search_data")
async def search_regulation(query: str):
    """
    Найди в базе знаний нормативных документов информацию, соответствующую запросу пользователя. 
    Дай подробный ответ на основе контекста. В конце обязательно перечисли метаданные каждого источника.
    """
    docs = await retriever.aget_relevant_documents(query)
    context_parts = []

    for doc in docs:
        metadata_dict = doc.metadata.copy()

        if "parent_doc" in metadata_dict:
            text = metadata_dict["parent_doc"]
            metadata_dict.pop("parent_doc")
        else:
            text = doc.page_content

        metadata = json.dumps(metadata_dict, ensure_ascii=False)
        context_parts.append(f"Текст: {text}\nМетаданные: {metadata}")

    log_rag_interaction(query, text, metadata)
    return "\n\n".join(context_parts)


@tool
async def search_citation(query: str):
    """
    Найди и процитируй дословно пункты нормативной документации из базы знаний, которые относятся к запросу пользователя. 
    Сохрани оригинальное оформление и структуру документа (нумерация, абзацы). В конце обязательно перечисли метаданные каждого источника.
    """
    docs = await retriever.aget_relevant_documents(query)
    context_parts = []

    for doc in docs:
        metadata_dict = doc.metadata.copy()

        if "parent_doc" in metadata_dict:
            text = metadata_dict["parent_doc"]
            metadata_dict.pop("parent_doc")
        else:
            text = doc.page_content

        metadata = json.dumps(metadata_dict, ensure_ascii=False)
        context_parts.append(f"Текст: {text}\nМетаданные: {metadata}")
    log_rag_interaction(query, text, metadata)
    return "\n\n".join(context_parts)


@tool
async def summary(query: str):
    """
    Сделай подробное описание указанного раздела или документа из базы знаний нормативной документации. Используй простые формулировки, сохраняя точный смысл. 
    В конце обязательно перечисли метаданные каждого источника.
    """
    docs = await retriever.aget_relevant_documents(query)
    context_parts = []

    for doc in docs:
        metadata_dict = doc.metadata.copy()

        if "parent_doc" in metadata_dict:
            text = metadata_dict["parent_doc"]
            metadata_dict.pop("parent_doc")
        else:
            text = doc.page_content

        metadata = json.dumps(metadata_dict, ensure_ascii=False)
        context_parts.append(f"Текст: {text}\nМетаданные: {metadata}")
    log_rag_interaction(query, text, metadata)
    return "\n\n".join(context_parts)


@tool
async def semantic_search(query: str):
    """
    Проанализируй предоставленный текст на соответствие требованиям нормативной документации из базы знаний. 
    Укажи:
    - какие пункты и положения документа соответствуют содержанию текста;
    - какие пункты нарушены или отсутствуют;
    - общий вывод о соответствии.
    В конце обязательно перечисли метаданные каждого источника.
    """
    docs = await retriever.aget_relevant_documents(query)
    context_parts = []

    for doc in docs:
        metadata_dict = doc.metadata.copy()

        if "parent_doc" in metadata_dict:
            text = metadata_dict["parent_doc"]
            metadata_dict.pop("parent_doc")
        else:
            text = doc.page_content

        metadata = json.dumps(metadata_dict, ensure_ascii=False)
        context_parts.append(f"Текст: {text}\nМетаданные: {metadata}")

    log_rag_interaction(query, text, metadata)
    return "\n\n".join(context_parts)


@tool
async def search_plan(query: str):
    """
    Сформируй пошаговый план действий в соответствии с требованиями нормативной документации из базы знаний. 
    План должен быть конкретным, последовательным, со ссылками на соответствующие пункты и разделы документации. 
    Учитывай цель и контекст, указанный пользователем.
    В конце обязательно перечисли метаданные каждого источника.
    """
    docs = await retriever.aget_relevant_documents(query)
    context_parts = []

    for doc in docs:
        metadata_dict = doc.metadata.copy()

        if "parent_doc" in metadata_dict:
            text = metadata_dict["parent_doc"]
            metadata_dict.pop("parent_doc")
        else:
            text = doc.page_content

        metadata = json.dumps(metadata_dict, ensure_ascii=False)
        context_parts.append(f"Текст: {text}\nМетаданные: {metadata}")

    log_rag_interaction(query, text, metadata)
    return "\n\n".join(context_parts)


async def call_model(state: MessagesState, model_name: str, *, tools_map=None, store=None):

    if 'gpt' in model_name:
        model = init_chat_model(model=model_name,api_key=OPENAI_API_KEY,temperature=0.1)
    else:
        model = ChatGoogleGenerativeAI(model=model_name,google_api_key=GEMINI_API_KEY,temperature=0.1,convert_system_message_to_human=True)

    system_msg = "Ты ассистент, который должен поддерживать непрерывный диалог с пользователем"
    model_with_tools = model.bind_tools(list(tools_map.values()))
    response = await model_with_tools.ainvoke(
        [{"role": "system", "content": system_msg}] + state["messages"])
    log_model(model_name)
    return {"messages": [response]}


async def tool_node(state: MessagesState, *, tools_map=None, store=None):
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None)
    results = []

    if tool_calls:
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            tool_fn = tools_map.get(name)
            if tool_fn:
                out = await tool_fn.ainvoke(args)
            else:
                out = "Unknown tool"

            results.append(
                ToolMessage(
                    content=str(out),
                    name=name,
                    tool_call_id=call.get("id") or "unknown_id"
                )
            )

    return {"messages": results}


def should_continue(state: MessagesState) -> str:
    last_msg = state["messages"][-1]
    return "tools" if getattr(last_msg, "tool_calls", None) else "end"


def build_graph(model_name):
    tools_map = {
        t.name: t
        for t in [search_regulation, search_citation, summary, semantic_search, search_plan]
    }

    builder = StateGraph(MessagesState)

    builder.add_node("call_model", partial(call_model, tools_map=tools_map, model_name=model_name))
    builder.add_node("tool_node", partial(tool_node, tools_map=tools_map))

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, {
        "tools": "tool_node",
        "end": END,
    })
    builder.add_edge("tool_node", "call_model")
    return builder.compile()
