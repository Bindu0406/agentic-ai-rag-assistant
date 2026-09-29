import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Local MiniLM embeddings (384-dim)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to Pinecone vector database
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Document retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Groq LLM using the verified active model
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2
)

# Prompt template grounded to eBook context
system_prompt = (
    "You are an expert AI assistant answering questions based strictly on the provided eBook context.\n"
    "Context:\n{context}\n\n"
    "If the answer cannot be found in the context, say that the information is not present in the document."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def query_agent(question: str) -> str:
    return rag_chain.invoke(question)

if __name__ == "__main__":
    test_question = "What is agentic AI?"
    print(f"Testing Question: {test_question}\n")
    response = query_agent(test_question)
    print("Response:\n", response)