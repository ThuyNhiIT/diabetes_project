# backend/src/rag_api.py
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from pydantic import BaseModel
from fastapi import APIRouter

BASE_DIR = Path(__file__).parents[1]
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_PATH = BASE_DIR / "models" / "diabetes_faiss_index"

# ✅ Load embeddings giống lúc build index
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ Load FAISS với embeddings
vectorstore = FAISS.load_local(
    str(INDEX_PATH),
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

router = APIRouter()

class Question(BaseModel):
    query: str

@router.post("/ask")
def ask_diabetes(question: Question):
    answer = qa_chain.run(question.query)
    return {"answer": answer}
