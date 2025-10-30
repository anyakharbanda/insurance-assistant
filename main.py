import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# ===============================
# 🔑 CONFIGURE GEMINI API KEY HERE
# ===============================
genai.configure(api_key="AIzaSyCz1cSWFSMOFibAy9Yb7Dqq5SFZHEMZZZ0")  # ← paste your Gemini key

# -------------------------------
# Helper functions
# -------------------------------

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


@st.cache_data(show_spinner=False)
def split_text(text, max_chars=1500):
    """Split text into smaller overlapping chunks."""
    chunks = []
    for i in range(0, len(text), max_chars):
        chunk = text[i:i + max_chars]
        chunks.append(chunk)
    return chunks


@st.cache_data(show_spinner=False)
def embed_texts_with_gemini(texts):
    """Generate embeddings for all text chunks."""
    model = "models/text-embedding-004"
    embeddings = []
    for t in texts:
        try:
            result = genai.embed_content(model=model, content=t)
            embeddings.append(result["embedding"])
        except Exception as e:
            st.warning(f"Embedding error: {e}")
            embeddings.append(np.zeros(768))
    return np.array(embeddings)


def find_most_relevant_chunks(query, chunks, embeddings, top_k=3):
    """Find the most relevant chunks to the query using cosine similarity."""
    query_emb = genai.embed_content(model="models/text-embedding-004", content=query)["embedding"]
    sims = cosine_similarity([query_emb], embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


def generate_answer_with_gemini(query, context):
    """Generate smart, natural answers from Gemini using context."""
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    prompt = f"""
    You are an expert insurance assistant.

    Here are relevant parts of the user's insurance policy:
    {context}

    Question: {query}

    Please answer clearly and briefly. 
    - Do not just copy text from the policy.
    - Provide a short, human-friendly explanation.
    - If applicable, include eligibility, coverage decision, or related details.
    """
    response = model.generate_content(prompt)
    return response.text.strip() if response and response.text else "No clear answer found."


# -------------------------------
# Streamlit App
# -------------------------------

st.set_page_config(page_title="Insurance Policy QA", page_icon="📘", layout="centered")
st.title("📘 Smart Insurance Policy QA Assistant")

st.markdown("""
Upload your **insurance policy PDF**, and then ask any question like:
> “46-year-old male, knee surgery in Pune, 3-month policy”

The system will read your policy, understand it semantically, and give a **short, clear** answer — not just copied clauses.
""")

uploaded_file = st.file_uploader("📄 Upload your insurance policy (PDF)", type=["pdf"])

if uploaded_file:
    st.info("Extracting text from your PDF...")
    policy_text = extract_text_from_pdf(uploaded_file)

    if len(policy_text) < 100:
        st.warning("⚠️ Couldn’t extract enough text from this PDF. Try another file.")
    else:
        st.success("✅ Policy uploaded and processed!")

        # Split & embed
        st.info("Creating embeddings (this will be cached for faster next runs)...")
        chunks = split_text(policy_text)
        embeddings = embed_texts_with_gemini(chunks)
        st.success("✅ Embeddings ready!")

        # Ask question
        query = st.text_input("❓ Ask a question about your policy:")

        if query:
            with st.spinner("Analyzing your policy..."):
                relevant_chunks = find_most_relevant_chunks(query, chunks, embeddings)
                context = "\n\n".join(relevant_chunks)
                answer = generate_answer_with_gemini(query, context)

            st.markdown("### 💬 Answer:")
            st.write(answer)

            with st.expander("🔍 View retrieved policy context"):
                st.write(context)
else:
    st.info("Please upload your insurance policy PDF to start.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Gemini 2.5")

