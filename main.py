import uuid
import json
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.store.base import BaseStore

# Базовая модель
model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

# DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

async with (
    AsyncPostgresStore.from_conn_string(DB_URI) as store,
    AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    # await store.setup()
    # await checkpointer.setup()

    @tool
    def get_weather(city: str) -> str:
        """Погода в городе"""
        return f"Погода в {city} - солнечно, 25°C"

    @tool
    def get_time() -> str:
        """Текущее местное время (время сервера). Используйте, когда пользователь спрашивает о времени."""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tools_map: Dict[str, Any] = {t.name: t for t in [get_weather, get_time]}


    async def build_system_message(state: MessagesState, store: BaseStore, user_id: str) -> str:
        """Готовим system-промпт с краткой выжимкой памяти."""
        namespace = ("memories", user_id)
        last_user_text = str(state["messages"][-1].content)
        memories = await store.asearch(namespace, query=last_user_text)
        info = "\n".join([d.value.get("data", "") for d in memories]) if memories else ""
        system_msg = (
            "Ты ассистент"
            + (f" Информация о пользователе (из памяти): {info}" if info else "")
        )
        return system_msg

    async def call_model(
        state: MessagesState,
        config: RunnableConfig,
        *,
        store: BaseStore,
    ):
        user_id = config["configurable"]["user_id"]

        system_msg = await build_system_message(state, store, user_id)

        model_with_tools = model.bind_tools(list(tools_map.values()))

        response = await model_with_tools.ainvoke(
            [{"role": "system", "content": system_msg}] + state["messages"]
        )
        return {"messages": [response]}

    async def tool_node(
        state: MessagesState,
        config: RunnableConfig,
        *,
        store: BaseStore,
    ):
        user_id = config["configurable"]["user_id"]

        last_msg = state["messages"][-1]
        results = []

        # Для моделей LangChain tool_calls лежит в атрибуте AIMessage.tool_calls
        tool_calls = getattr(last_msg, "tool_calls", None)

        if tool_calls:
            for call in tool_calls:
                name = call.get("name")
                raw_args = call.get("args") or {}

                # Аргументы могут быть строкой (JSON) или dict — нормализуем
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {}
                else:
                    args = dict(raw_args)

                # Авто-подстановка user_id для нашего инструмента имени, если не указан
                if name == "get_user_name" and "user_id" not in args:
                    args["user_id"] = str(user_id)

                tool_fn = tools_map.get(name)
                if tool_fn is None:
                    # Неизвестный инструмент — вернём заглушку
                    results.append(
                        {
                            "role": "tool",
                            "name": name or "unknown_tool",
                            "content": f"Tool '{name}' is not available.",
                        }
                    )
                    continue

                # Синхронный или асинхронный инструмент — вызываем корректно
                if getattr(tool_fn, "ainvoke", None):
                    out = await tool_fn.ainvoke(args)
                else:
                    out = tool_fn.invoke(args)

                # Возвращаем сообщение tool для модели
                results.append({"role": "tool", "name": name, "content": str(out)})

        return {"messages": results}

    # Условная маршрутизация: если модель попросила инструмент(ы) — идём в tool_node, иначе завершаем
    def should_continue(state: MessagesState) -> str:
        last_msg = state["messages"][-1]
        has_tools = bool(getattr(last_msg, "tool_calls", None))
        return "tools" if has_tools else "end"

    # -----------------------------
    # Сборка графа
    # -----------------------------
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "tools": "tool_node",
            "end": END,
        },
    )
    # После инструментов снова к модели — чтобы она использовала результаты tool и выдала финальный ответ
    builder.add_edge("tool_node", "call_model")

    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )

    # -----------------------------
    # Примеры запуска
    # -----------------------------


    # 3) Пример вызова weather-инструмента (модель сама выберет get_weather)
    config = {"configurable": {"thread_id": "3", "user_id": "1"}}
    for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "what's the weather in Paris?"}]},
        config,
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()

