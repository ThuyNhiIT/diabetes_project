import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load biến môi trường
BASE_DIR = Path(__file__).parents[1]
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY chưa được cấu hình trong .env")

DATA_PATH = BASE_DIR / "data" / "diabetes_knowledge.txt"
INDEX_PATH = BASE_DIR / "models" / "diabetes_faiss_index"

print("📄 Đang đọc dữ liệu từ:", DATA_PATH)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Chia nhỏ văn bản
print("✂️ Đang chia nhỏ dữ liệu...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.create_documents([raw_text])

# Tạo embeddings & FAISS index
print("⚙️ Đang tạo embeddings và FAISS index...")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(docs, embeddings)

# Lưu index
print("💾 Đang lưu index vào:", INDEX_PATH)
vectorstore.save_local(str(INDEX_PATH))
print("✅ Hoàn thành tạo FAISS index!")
