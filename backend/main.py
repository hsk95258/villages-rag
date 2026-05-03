import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import load_rag_chain

app = FastAPI(title="Villages RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever, chain = load_rag_chain()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.get("/")
def root():
    return {"status": "Villages RAG API is running"}

@app.post("/ask")
def ask(request: QueryRequest):
    answer = chain.invoke(request.question)
    docs = retriever.invoke(request.question)
    sources = [doc.metadata for doc in docs]
    return QueryResponse(answer=answer, sources=sources)