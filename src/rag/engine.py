"""RAG Engine — Q&A over milling machine SOPs using LangChain + ChromaDB."""

import os, yaml
from pathlib import Path
from loguru import logger
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from src.data.sop_documents import get_sop_documents

PROMPT = PromptTemplate(
    template="""You are ShopFloor AI, an expert manufacturing assistant for CNC milling operations.
You help operators and engineers with questions about predictive maintenance, tool wear,
temperature monitoring, power/torque specs, failure modes (TWF, HDF, PWF, OSF, RNF),
quality variants (L/M/H), and KPI definitions.

Use the context below. If the answer isn't there, say so — never invent specs.
Cite which document the information comes from.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)


class RAGEngine:
    def __init__(self, config: dict = None):
        if config is None:
            with open("configs/config.yaml") as f:
                config = yaml.safe_load(f)
        self.config = config
        self.collection = config["chromadb"]["collection_name"]

        host = os.getenv("CHROMA_HOST", config["chromadb"]["host"])
        port = int(os.getenv("CHROMA_PORT", config["chromadb"]["port"]))
        try:
            self.client = chromadb.HttpClient(host=host, port=port)
            logger.info(f"ChromaDB connected at {host}:{port}")
        except Exception:
            os.makedirs("data/vectordb", exist_ok=True)
            self.client = chromadb.PersistentClient(path="data/vectordb")
            logger.info("Using local ChromaDB")

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
        self.qa_chain = None

    def ingest(self, docs: list[dict] = None) -> int:
        docs = docs or get_sop_documents()
        lc_docs = [
            Document(page_content=d["content"], metadata={"title": d["title"], "category": d["category"]})
            for d in docs
        ]
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(lc_docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks, embedding=self.embeddings,
            client=self.client, collection_name=self.collection,
        )
        logger.info(f"Ingested {len(chunks)} chunks from {len(docs)} documents")
        return len(chunks)

    def build_chain(self):
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                client=self.client, collection_name=self.collection,
                embedding_function=self.embeddings,
            )
        llm_cfg = self.config["llm"]
        llm = ChatOpenAI(model=llm_cfg["model"], temperature=llm_cfg["temperature"], max_tokens=llm_cfg["max_tokens"])
        retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 8})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever,
            return_source_documents=True, chain_type_kwargs={"prompt": PROMPT},
        )
        logger.info("QA chain ready")

    def ask(self, question: str) -> dict:
        if not self.qa_chain:
            self.build_chain()
        result = self.qa_chain.invoke({"query": question})
        sources = [{"title": d.metadata.get("title", ""), "excerpt": d.page_content[:200]} for d in result.get("source_documents", [])]
        return {"question": question, "answer": result["result"], "sources": sources, "n_sources": len(sources)}


if __name__ == "__main__":
    engine = RAGEngine()
    engine.ingest()
    for q in ["What causes heat dissipation failure?", "When should I replace the tool?", "What is the power failure threshold?"]:
        r = engine.ask(q)
        print(f"\nQ: {q}\nA: {r['answer'][:300]}...")
