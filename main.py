import streamlit as st
import google.generativeai as genai
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
 
# ===============================
# 🔑 CONFIGURE GEMINI API KEY (from Streamlit secrets)
# ===============================
genai.configure(api_key=st.secrets["general"]["GEMINI_API_KEY"])
 
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
        except Exception:
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
 
st.set_page_config(page_title="Insurance Policy Q&A", layout="centered")
st.title("Smart Insurance Policy Q&A Assistant")
 
st.markdown("""
Upload **one or two insurance policies (PDF)**, and ask questions like:
> “What’s the maternity coverage difference?”
> “Which offers better hospitalization benefits?”
 
The system will analyze your policy content and give clear, human answers.
""")
 
# Option for single or dual mode
mode = st.radio("Choose mode:", ["Single Policy Q&A", "Compare Two Policies"])
 
if mode == "Single Policy Q&A":
    uploaded_file = st.file_uploader("Upload your insurance policy (PDF)", type=["pdf"])
 
    if uploaded_file:
        with st.spinner("Extracting text from your PDF..."):
            policy_text = extract_text_from_pdf(uploaded_file)
 
        if len(policy_text) < 100:
            st.warning("Couldn’t extract enough text from this PDF. Try another file.")
        else:
            st.success("Policy uploaded and processed!")
 
            chunks = split_text(policy_text)
            embeddings = embed_texts_with_gemini(chunks)
 
            query = st.text_input("Ask a question about your policy:")
 
            if query:
                with st.spinner("Analyzing your policy..."):
                    relevant_chunks = find_most_relevant_chunks(query, chunks, embeddings)
                    context = "\n\n".join(relevant_chunks)
                    answer = generate_answer_with_gemini(query, context)
 
                st.markdown("### Answer:")
                st.write(answer)
 
                with st.expander("View retrieved policy context"):
                    st.write(context)
    else:
        st.info("Please upload your insurance policy PDF to start.")
 
else:  # 🟢 Compare Two Policies Mode
    col1, col2 = st.columns(2)
    with col1:
        policy1 = st.file_uploader("Upload Policy 1 (PDF)", type=["pdf"], key="p1")
    with col2:
        policy2 = st.file_uploader("Upload Policy 2 (PDF)", type=["pdf"], key="p2")
 
    if policy1 and policy2:
        with st.spinner("Processing both policies..."):
            text1 = extract_text_from_pdf(policy1)
            text2 = extract_text_from_pdf(policy2)
 
            chunks1, chunks2 = split_text(text1), split_text(text2)
            emb1, emb2 = embed_texts_with_gemini(chunks1), embed_texts_with_gemini(chunks2)
 
        query = st.text_input("Ask a comparison question (e.g., 'Which offers better maternity coverage?')")
 
        if query:
            with st.spinner("Comparing both policies..."):
                rel1 = find_most_relevant_chunks(query, chunks1, emb1)
                rel2 = find_most_relevant_chunks(query, chunks2, emb2)
 
                context1, context2 = "\n\n".join(rel1), "\n\n".join(rel2)
 
                model = genai.GenerativeModel("models/gemini-2.5-flash")
                compare_prompt = f"""
                You are an expert insurance advisor comparing two policies.
 
                Policy A:
                {context1}
 
                Policy B:
                {context2}
 
                Question: {query}
 
                Please answer clearly:
                - Which policy offers better coverage or benefits for this query?
                - Highlight key differences in a short, user-friendly way.
                - Avoid repeating identical clauses.
                """
                comparison = model.generate_content(compare_prompt)
 
            st.markdown("###Comparison Result")
            st.write(comparison.text.strip() if comparison and comparison.text else "No clear comparison found.")
 
            with st.expander("View policy contexts"):
                st.subheader("Policy 1 Relevant Context")
                st.write(context1)
                st.subheader("Policy 2 Relevant Context")
                st.write(context2)
    else:
        st.info("Upload both policies to enable comparison.")
 
st.markdown("---")
 