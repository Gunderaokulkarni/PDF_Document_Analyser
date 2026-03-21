from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os

load_dotenv()


# -----------------------------
# PDF TEXT
# -----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


# -----------------------------
# CHUNKING
# -----------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# -----------------------------
# VECTOR STORE
# -----------------------------
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(text_chunks, embeddings)

    save_path = os.path.join(os.getcwd(), "faiss_index")
    db.save_local(save_path)


# -----------------------------
# FORMAT CHAT HISTORY (FIXED)
# -----------------------------
def format_chat_history(chat_history):
    formatted = ""

    for item in chat_history:
        # Case 1: tuple ("question", "answer")
        if isinstance(item, tuple) and len(item) == 2:
            human, ai = item
            formatted += f"User: {human}\nAssistant: {ai}\n"

        # Case 2: dict {"role": "...", "content": "..."}
        elif isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            if role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"

        # Case 3: plain string
        elif isinstance(item, str):
            formatted += f"{item}\n"

    return formatted


# -----------------------------
# LLM CHAIN
# -----------------------------
def get_conversational_chain():

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Answer the question using the provided context. "
         "If the answer is not in the context, say: "
         "'Answer is not available in the context.'"),

        ("user", 
         "Chat History:\n{chat_history}\n\n"
         "Context:\n{context}\n\n"
         "Question:\n{question}")
    ])

    model = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    chain = prompt | model | StrOutputParser()

    return chain


# -----------------------------
# USER QUERY
# -----------------------------
def user_input(user_question, chat_history):

    if not user_question.strip():
        return "Please enter a valid question."

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db_path = os.path.join(os.getcwd(), "faiss_index")

    if not os.path.exists(db_path):
        return "Please upload and process a PDF first."

    db = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(user_question)

    context = "\n".join([doc.page_content for doc in docs])
    context = context if context else "No relevant context found."
    context = context[:5000]

    formatted_history = format_chat_history(chat_history)

    chain = get_conversational_chain()

    response = chain.invoke({
        "context": context,
        "chat_history": formatted_history,
        "question": user_question
    })

    return response
