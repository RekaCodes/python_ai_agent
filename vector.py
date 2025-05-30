import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

df = pd.read_csv("student_habits_performance.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row['student_id'] + " " + str(row['exam_score']) + " " +  str(row['study_hours_per_day']),
            metadata={"age":str(row['age']), "gender": str(row['gender'])},
            id=str(i)   
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="student_habits_performance",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever()