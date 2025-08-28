from parsing import image_to_json, table_to_json, parse_html
from uuid import uuid4
import json
import uuid
from langchain_core.documents import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema import Document as Document_proc
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio

            

def process_value(data):
    documents = []

    def traverse(obj, keys_path):
        if isinstance(obj, dict):
            for key, value in obj.items():
                traverse(value, keys_path + [key])
        elif isinstance(obj, str): 
            doc = Document(
                page_content=obj,
                metadata={"Местоположение": keys_path}
            )
            documents.append(doc)
        else:
            doc = Document(
                page_content=str(obj),
                metadata={"Местоположение": keys_path}
            )
            documents.append(doc)

    traverse(data, [])
    return documents


def extract_data_json(json_path, function, processor):
    documents = []
    try:
        data = processor(json_path)
        documents = function(data)

    except Exception as e:
        print(f"Ошибка при обработке {json_path}: {e}")
        return [], []

    
    records = []
    for doc in documents:
        record = {
            "id": str(uuid4()),
            "text": doc.page_content,
            "metadata": doc.metadata
        }
        records.append(record)
       

    ids = [record['id'] for record in records]
    documents_texts = [record['text'] for record in records]
    metas = [record['metadata'] for record in records]
    return ids, documents_texts, metas


def process_json_images(data):
    documents = []

    for item in data:
        if not isinstance(item, dict):
            continue

        keys = list(item.keys())
     
        if len(keys) >= 3:
            text_keys = [keys[-3], keys[-2]]
        else:
            text_keys = keys  

        page_content = "\n".join(str(item[k]) for k in text_keys if k in item)
        metadata = {k: item[k] for k in item if k not in text_keys}

        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    return documents


def process_json_tables(data):
    documents = []

    for item in data:
        if not isinstance(item, dict):
            continue

        page_content = json.dumps(item, ensure_ascii=False)

        doc = Document(page_content=page_content, metadata={})
        documents.append(doc)

    return documents


def chunk_texts(ids, documents_texts, documents_metadata):
    """
    ids: список уникальных id документов
    documents_texts: список текстов (page_content)
    documents_metadata: список метаданных (dict) для каждого документа
    """
    chunked_documents = []
    chunked_ids = []

    for idx, text, metadata in zip(ids, documents_texts, documents_metadata):
        doc = Document(page_content=text, metadata=metadata)

        metadata_safe = {k: (", ".join(v) if isinstance(v, list) else v) for k, v in metadata.items()}
        doc = Document(page_content=text, metadata=metadata_safe)


        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            length_function=len,
            keep_separator=True
        )

        chunks = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            chunk_doc = Document(page_content=chunk, metadata=doc.metadata)
            chunked_documents.append(chunk_doc)
            chunked_ids.append(f"{idx}_chunk_{i}")

    return chunked_ids, chunked_documents


def chunk_texts_parent(ids, documents_texts, documents_metadata):
    """
    ids: список уникальных id документов
    documents_texts: список текстов (page_content)
    documents_metadata: список метаданных (dict) для каждого документа
    """
    chunked_documents = []
    chunked_ids = []

    for idx, text, metadata in zip(ids, documents_texts, documents_metadata):
        metadata_safe = {k: (", ".join(v) if isinstance(v, list) else v) for k, v in metadata.items()}
        metadata_safe["parent_doc"] = text

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            length_function=len,
            keep_separator=True
        )

        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            chunk_doc = Document_proc(
                page_content=chunk,
                metadata=metadata_safe)
            chunked_documents.append(chunk_doc)
            chunked_ids.append(f"{idx}_chunk_{i}")

    return chunked_ids, chunked_documents


def get_vector_db(ids, documents_texts, documents_metadata, ids_img, images_texts, images_metadata, ids_table, tables_texts, tables_metadata, user_query):
    collection_name = f"collection"

    embedding = OpenAIEmbeddings(api_key="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")

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory="./chroma_db")

    chunked_ids, chunked_documents = chunk_texts_parent(ids, documents_texts, documents_metadata)
    vector_store.add_documents(documents=chunked_documents, ids=chunked_ids)

    chunked_ids_img, chunked_documents_img = chunk_texts(ids_img, images_texts, images_metadata)
    vector_store.add_documents(documents=chunked_documents_img, ids=chunked_ids_img)

    chunked_ids_table, chunked_documents_table = chunk_texts(ids_table, tables_texts, tables_metadata)
    vector_store.add_documents(documents=chunked_documents_table, ids=chunked_ids_table)

    # documents = chunked_documents + chunked_documents_img + chunked_documents_table

    # chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    # bm25_retriever = BM25Retriever.from_documents(documents=documents, k=5)
    # ensemble_retriever = EnsembleRetriever(
    #     retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
    # )
    # context = ensemble_retriever.invoke(user_query)

    # return chroma_retriever

# def get_retriever():
#     collection_name = f"collection"

#     embedding = OpenAIEmbeddings(api_key="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")
   
#     vector_store = Chroma(
#         collection_name=collection_name,
#         embedding_function=embedding,
#         persist_directory="./chroma_db"
#     )
  
#     chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

#     return chroma_retriever




async def get_retriever(collection_name: str = "collection", k: int = 5):
    """
    Асинхронно возвращает ретривер для Chroma.
    """

    embedding = await asyncio.to_thread(OpenAIEmbeddings, api_key="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")
    vector_store = await asyncio.to_thread(
        Chroma,
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory="./chroma_db"
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


def main(json_path, img_path, table_path, user_query):
    ids, documents_texts, metadata = extract_data_json(json_path, process_value, parse_html)
    ids_img, images_texts, images_metadata = extract_data_json(img_path, process_json_images, image_to_json)
    ids_table, tables_texts, tables_metadata = extract_data_json(table_path, process_json_tables, table_to_json)
    get_vector_db(ids, documents_texts, metadata, ids_img, images_texts, images_metadata, ids_table, tables_texts, tables_metadata, user_query)
    # print(context)

# main('https://files.stroyinf.ru/Data2/1/4293842/4293842059.htm','C:\\documents\\engyme\\images\\НП-068-05_Изображения.xlsx', 'C:\\documents\\engyme\\tables\\НП-068-05.xlsx', 'кляка')