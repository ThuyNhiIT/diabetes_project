# backend/src/rag_api.py
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ===========================
# Load biến môi trường
# ===========================
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("⚠️ OPENAI_API_KEY chưa được cấu hình trong file .env")

# ===========================
# Config model
# ===========================
MODEL_NAME = "gpt-3.5-turbo"  # hoặc gpt-4
EMBEDDING_MODEL = "text-embedding-3-small"

# ===========================
# FAISS index path
# ===========================
FAISS_INDEX_PATH = BASE_DIR / "models" / "diabetes_faiss_index"

# ===========================
# Load FAISS index
# ===========================
print(f"📂 Đang load FAISS index từ: {FAISS_INDEX_PATH}")
vectorstore = None
qa_chain = None
try:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS index load thành công!")
except Exception as e:
    print(f"❌ Lỗi khi load FAISS index: {e}")

# ===========================
# Prompt template
# ===========================
prompt_template = """
Bạn là một chuyên gia y tế về bệnh tiểu đường.
Dựa vào thông tin trong phần "Ngữ cảnh" bên dưới, hãy trả lời câu hỏi của người dùng:
- Trả lời chi tiết, rõ ràng, đầy đủ.
- Nếu không chắc chắn hoặc thông tin không có, hãy nói rõ.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Trả lời:
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# ===========================
# Init LLM + RetrievalQA
# ===========================
if vectorstore:
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=OPENAI_API_KEY
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# ===========================
# API Router
# ===========================
router = APIRouter()

class QuestionRequest(BaseModel):
    question: str = Field(None, description="Câu hỏi của người dùng")
    query: str = Field(None, description="Alias cho 'question', để tương thích với frontend cũ")

@router.post("/ask")
async def ask_question(req: QuestionRequest):
    if not qa_chain:
        raise HTTPException(status_code=500, detail="FAISS index chưa sẵn sàng.")

    # Ưu tiên lấy 'question', fallback sang 'query'
    user_question = req.question or req.query
    if not user_question:
        raise HTTPException(status_code=422, detail="Cần truyền 'question' hoặc 'query'.")

    try:
        result = qa_chain.invoke({"query": user_question})
        return {"answer": result.get("result", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
