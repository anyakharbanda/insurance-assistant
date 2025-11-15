AI-Powered Insurance Policy Q&A Assistant

A smart RAG-based system that helps users understand complex insurance policies.
Users can upload one or two PDFs, ask natural-language questions, and get clear, AI-generated answers based on the policy content.

🚀 Overview

This project converts long insurance documents into a searchable, conversational assistant using:

PDF Text Extraction

Policy Segmentation (NLP)

Semantic Embeddings (Deep Learning)

Similarity Search

Gemini LLM Answer Generation

Streamlit Web App

🧠 Core Concepts Used
✔ NLP

Hybrid segmentation using regex + rule-based NLP

Text chunking for large documents

Semantic search using embeddings

✔ Deep Learning

BAAI/bge-small-en-v1.5 transformer for generating 768-d embeddings

Embeddings encode meaning, enabling better matching than keyword search

✔ Machine Learning

Cosine similarity to retrieve relevant policy sections

Rank top-k chunks based on semantic closeness to user query

✔ LLM (Generative AI)

Gemini 2.5 Flash used to:

Read retrieved context

Generate clear, human-friendly answers

Compare two policies

🏗️ Pipeline
PDF → Extract Text → Segment Policy → Generate Embeddings → 
Cosine Similarity → Retrieve Relevant Chunks → Gemini Answer → Streamlit UI

📂 Folder Structure
data/
   segmented_policies/
   policy_embeddings.json
segment_policies_hybrid.py
generate_embeddings.py
main.py (Streamlit App)

🧰 Technologies Used

Python
SentenceTransformers (BAAI model)
Google Gemini API
PyPDF2
scikit-learn
NumPy
Streamlit

▶ How to Run
pip install -r requirements.txt
streamlit run main.py

Upload a policy → Ask your question → Get AI-generated answers.

🌟 Why This Project Matters

Turns complex insurance PDFs into interactive Q&A

Helps customers understand coverage, exclusions, claims

Shows real-world use of RAG (Retrieval-Augmented Generation)

Combines NLP + Deep Learning + LLMs + UI
