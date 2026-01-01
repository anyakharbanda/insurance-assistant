# AI-Powered Insurance Policy Q&A Assistant  
**RAG-based Insurance Document Understanding System**


## Overview

Insurance policies are long and difficult to read.  
This project converts policy PDFs into a **conversational assistant** that answers user questions using the actual policy text.

Users can upload **one or two PDF policies**, ask natural queries, and the AI responds with **clear, contextual answers** based on retrieved policy chunks.

This project demonstrates a full **Retrieval-Augmented Generation (RAG) pipeline** with embeddings + LLM-based answer generation.


## Key Features

- Upload insurance policy PDFs
- Ask natural-language questions
- AI generates answers grounded in document content
- Semantic similarity retrieval (not keyword-based)
- Compare policy clauses (when 2 PDFs uploaded)
- Built with Streamlit for an interactive UI


## Core Concepts Used

### NLP
- Hybrid segmentation (regex + rule-based)
- Text chunking for long policy documents
- Semantic embeddings using **BAAI/bge-small-en-v1.5**
- Context ranking using cosine similarity

### Deep Learning
- 768-dimension embeddings for semantic meaning
- Embedding vectors enable accurate retrieval

### Machine Learning
- Cosine similarity for top-k context retrieval
- Embedding search to find relevant clauses

### LLM (Generative AI)
- **Gemini 2.5 Flash** used for answer formulation
- Summarizes long sections into human-friendly outputs
- Compares clauses between two policies when needed
