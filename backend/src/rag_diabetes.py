import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Load biến môi trường
BASE_DIR = Path(__file__).parents[1]
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY chưa được cấu hình trong .env")

INDEX_PATH = BASE_DIR / "models" / "diabetes_faiss_index"

# Load FAISS index đã tạo
print("📂 Đang load FAISS index từ:", INDEX_PATH)
vectorstore = FAISS.load_local(str(INDEX_PATH), embeddings=None, allow_dangerous_deserialization=True)

# Tạo retriever + LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

def answer_question(query: str) -> str:
    return qa_chain.run(query)

if __name__ == "__main__":
    q = "Phòng ngừa tiểu đường type 2"
    print("Q:", q)
    print("A:", answer_question(q))
