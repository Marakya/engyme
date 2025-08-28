# import logging

# logging.basicConfig(
#     filename='logger/logs.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     encoding='utf-8'
# )

# def log_rag_interaction(query: str, context_parts: list):
#     logging.info("Запрос пользователя: %s", query)
#     logging.info("Контекст: %s", context_parts)


# def log_answer(answer: str):
#     logging.info("Ответ LLM: %s", answer)


import logging


rag_logger = logging.getLogger("rag_logger")
rag_logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logger/rag_logs.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

rag_logger.addHandler(file_handler)


def log_rag_interaction(query: str, text: str, metadata: str):
    rag_logger.info("Запрос пользователя: %s", query)
    rag_logger.info("Контекст: %s", text)
    rag_logger.info("Метаданные: %s", metadata)


def log_answer(answer: str):
    rag_logger.info("Ответ LLM: %s", answer)


def log_model(model_name: str):
    rag_logger.info("Используемая модель: %s", model_name)