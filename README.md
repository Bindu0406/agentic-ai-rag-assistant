# 🤖 Agentic AI Assistant: Production-Grade RAG Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3-1C3C3C?style=flat)](https://www.langchain.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Serverless_Vector_DB-000000?style=flat)](https://www.pinecone.io/)
[![Groq](https://img.shields.io/badge/Groq-LPU_Hardware-F05A28?style=flat)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_UI-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

An end-to-end Retrieval-Augmented Generation (RAG) system built to ingest, index, and conduct hallucination-free question answering over complex domain technical literature (*Agentic AI* eBook).

Designed with a **cost-optimized, high-throughput architecture** utilizing local 384-dimensional dense embeddings to eliminate per-token embedding costs, paired with Groq LPU hardware acceleration for low-latency response generation.

---

## 📸 System Interface

![Agentic AI Assistant Demo](assets/demo.png)

---

## 🏗️ Architecture & Pipeline Flow

```text
   [eBook PDF Source]
           │
           ▼
     [PyPDFLoader] 
           │
           ▼
[Recursive Text Splitter] (chunk_size=1000, chunk_overlap=200)
           │
           ▼
[sentence-transformers: all-MiniLM-L6-v2] (Local Dense 384-dim Embeddings)
           │
           ▼
[Pinecone Serverless Vector Database] (Metric: Cosine Similarity)
           │
   ┌───────┴────────────────────────┐
   ▼                                ▼
[User Input via Streamlit] ──> [Semantic Search (Top-k=3)]
                                    │
                                    ▼
                     [Grounded Context + Prompt Template]
                                    │
                                    ▼
                     [Groq LPU Inference: Qwen 27B / Llama]
                                    │
                                    ▼
                      [Streamlit Rendered Answer]