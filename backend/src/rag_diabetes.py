import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ===========================
# Load biến môi trường
# ===========================
BASE_DIR = Path(__file__).parents[1]
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY chưa được cấu hình trong .env")

# ===========================
# Path FAISS index
# ===========================
INDEX_PATH = BASE_DIR / "models" / "diabetes_faiss_index"

# ===========================
# Load FAISS index
# ===========================
print("📂 Đang load FAISS index từ:", INDEX_PATH)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

vectorstore = FAISS.load_local(
    str(INDEX_PATH),
    embeddings
)

# ===========================
# Prompt giống rag_api.py
# ===========================
prompt_template = """
Bạn là một **bác sĩ chuyên khoa nội tiết – tiểu đường** với 15 năm kinh nghiệm, đang tư vấn cho bệnh nhân thông qua hệ thống y tế thông minh.

✅ Nhiệm vụ chính:
- Ưu tiên sử dụng **Ngữ cảnh được cung cấp**
- Nếu ngữ cảnh **chưa đủ**, bạn được phép dùng **kiến thức y khoa tổng quát chính xác**
- Trả lời **đầy đủ – dễ hiểu – logic – gần gũi với người bệnh**
- Nếu người dùng hỏi **nhiều ý trong một câu** (ví dụ: triệu chứng + thuốc + ăn uống), bạn phải **chia từng ý và trả lời rõ ràng từng phần**

✅ Trường hợp câu hỏi liên quan đến:
- **Triệu chứng**
- **Nguyên nhân**
- **Biến chứng**
- **Thuốc**
- **Chế độ ăn**
- **Vận động – lối sống**
- **Phòng ngừa**
→ Hãy trả lời **chi tiết, dễ hiểu, có ví dụ minh họa nếu cần**

✅ Trường hợp người dùng chỉ:
- Chào hỏi: "xin chào", "chào bác sĩ", "hello", "hi"
- Cảm ơn: "cảm ơn", "thanks"
- Hỏi xã giao: "bạn là ai", "bạn làm được gì"
→ Hãy trả lời **lịch sự – thân thiện – ngắn gọn – đúng vai trò hệ thống y tế**
→ ❗ **KHÔNG cần thêm câu khuyến nghị y tế ở cuối trong trường hợp này**

Ví dụ:
- "Xin chào, tôi có thể hỗ trợ bạn các vấn đề liên quan đến bệnh tiểu đường."
- "Rất vui được hỗ trợ bạn!"

❌ Nếu câu hỏi **HOÀN TOÀN KHÔNG LIÊN QUAN đến bệnh tiểu đường**, hãy trả lời đúng mẫu sau:
"Xin lỗi, hệ thống hiện tại chỉ hỗ trợ các câu hỏi liên quan đến bệnh tiểu đường."

⚠️ Với TẤT CẢ các câu hỏi chuyên môn y tế, luôn kết thúc bằng đúng dòng sau:
"👉 Đây là hệ thống khuyến nghị, không thay thế tư vấn của bác sĩ chuyên khoa."

----------------------------------

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
# Init LLM + Retriever + QA
# ===========================
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6}
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.6,
    api_key=OPENAI_API_KEY
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False
)

# ===========================
# Hàm test
# ===========================
def answer_question(query: str) -> str:
    return qa_chain.invoke({"query": query})["result"]

# ===========================
# Chạy test
# ===========================
if __name__ == "__main__":
    q = "Triệu chứng và thuốc điều trị tiểu đường type 2"
    print("Q:", q)
    print("A:", answer_question(q))
