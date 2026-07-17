import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DOCS_PATH = os.path.join(os.path.dirname(__file__), "invoice_docs.txt")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "docs_collection"
MODEL_NAME = "qwen2.5:1.5b"
EMBEDDING_MODEL = "nomic-embed-text"

app = FastAPI(title="RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    source_documents: list[str]


def build_vectorstore():
    loader = TextLoader(DOCS_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore):
    prompt_template = PromptTemplate(
        template=(
            "You are an invoice data extraction assistant. "
            "Answer the question using ONLY the information in the context below. "
            "Extract the exact value, number, name, or code as it appears in the context. "
            "Do not confuse different fields — each line in the table is a separate item. "
            "If the context does not contain the answer, say 'I don't know'.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


print("Building vector store from invoice_docs.txt...")
vectorstore = build_vectorstore()
rag_chain, retriever = build_rag_chain(vectorstore)
print("RAG pipeline ready.")


@app.post("/query", response_model=QueryResponse)
def query_docs(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer = rag_chain.invoke(request.question)

    source_docs_raw = retriever.invoke(request.question)
    source_docs = [doc.page_content for doc in source_docs_raw]

    return QueryResponse(answer=answer, source_documents=source_docs)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
