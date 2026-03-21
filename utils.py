from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os
from config import GROQ_API_KEY


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
        model_name="mistralai/Mistral-7B-Instruct-v0.2"
    )

    db = FAISS.from_texts(text_chunks, embeddings)
    db.save_local("faiss_index")


# -----------------------------
# FORMAT CHAT HISTORY
# -----------------------------
def format_chat_history(chat_history):
    formatted = ""
    for msg in chat_history[-4:]:  # limit history
        if hasattr(msg, "content"):
            role = msg.__class__.__name__
            if "Human" in role:
                formatted += f"User: {msg.content}\n"
            else:
                formatted += f"Assistant: {msg.content}\n"
    return formatted


# -----------------------------
# CHAIN (FIXED)
# -----------------------------
def get_conversational_chain():

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the question using the provided context. "
         "If not found, say 'Answer is not available in the context.'"),

        ("user",
         "Chat History:\n{chat_history}\n\n"
         "Context:\n{context}\n\n"
         "Question:\n{question}")
    ])

    model = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.3,
        api_key=GROQ_API_KEY
    )

    chain = prompt | model | StrOutputParser()

    return chain


# -----------------------------
# USER QUERY (FIXED)
# -----------------------------
def user_input(user_question, chat_history):

    if not user_question.strip():
        return "Please enter a valid question."

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists("faiss_index"):
        return "Please upload and process a PDF first."

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(user_question)

    context = "\n".join([doc.page_content for doc in docs])
    context = context if context else "No relevant context found."
    context = context[:4000]

    formatted_history = format_chat_history(chat_history)

    chain = get_conversational_chain()

    try:
        response = chain.invoke({
            "chat_history": formatted_history,
            "context": context,
            "question": user_question
        })
    except Exception as e:
        print("ERROR:", e)
        return str(e)

    return response
