import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model

# Установи ключ Google API заранее
# os.environ["GOOGLE_API_KEY"] = "твой_api_ключ"

async def test_gemini():
    # GEMINI_API_KEY = "AIzaSyDZ6e-UJg2Pf8l4d39MA7g7FHqKXzIh7vQ"
    # GEMINI_MODEL = "gemini-2.5-flash" 


    # model = ChatGoogleGenerativeAI(
    #     model=GEMINI_MODEL,
    #     google_api_key=GEMINI_API_KEY,
    #     temperature=0.1,
    #     convert_system_message_to_human=True)
    API_KEY = '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'
    model = init_chat_model(
        model="gpt-4o-mini",
        api_key=API_KEY,
        temperature=0.1
    )

    try:
        response = await model.ainvoke([
            {"role": "user", "content": "Привет! Проверим работу модели."}
        ])
        print("Модель доступна! Ответ:")
        print(response.content)
    except Exception as e:
        print("Ошибка при вызове Gemini:")
        print(e)

asyncio.run(test_gemini())
