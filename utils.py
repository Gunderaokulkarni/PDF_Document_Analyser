from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os

# Load environment variables
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
# FORMAT CHAT HISTORY
# -----------------------------
def format_chat_history(chat_history):
    formatted = []
    for human, ai in chat_history:
        formatted.append({"role": "user", "content": human})
        formatted.append({"role": "assistant", "content": ai})
    return formatted


# -----------------------------
# CHAIN
# -----------------------------
def get_conversational_chain():

    prompt_template = """
    Answer the question using the provided context.
    If answer is not in context, say "answer is not available in the context".

    Chat History:
    {chat_history}

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGroq(
        model_name="qwen/qwen3-32b",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "context", "question"]
    )

    chain = prompt | model | StrOutputParser()

    return chain


# -----------------------------
# USER QUERY
# -----------------------------
def user_input(user_question, chat_history):

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

    # ✅ NEW STYLE (LangChain latest)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(user_question)

    context = "\n".join([doc.page_content for doc in docs])

    chain = get_conversational_chain()

    chain.invoke({
    "context": context,
    "chat_history": chat_history,
    "question": user_question
})

    return response
