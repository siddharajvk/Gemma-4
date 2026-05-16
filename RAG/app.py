import os
import fitz
import faiss
import ollama
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OLLAMA MODEL NAME
OLLAMA_MODEL = "gemma4:e2b"


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Healthcare RAG",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )

    return model


embedding_model = load_embedding_model()


# =========================================================
# PDF LOADER
# =========================================================

def load_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:

        text += page.get_text()

    return text


# =========================================================
# TEXT CHUNKING
# =========================================================

def create_chunks(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        chunks.append(chunk)

    return chunks


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

def create_embeddings(chunks):

    embeddings = embedding_model.encode(chunks)

    return embeddings


# =========================================================
# CREATE FAISS INDEX
# =========================================================

def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(
            embeddings,
            dtype=np.float32
        )
    )

    return index


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(
    query,
    index,
    chunks,
    top_k=3
):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        np.array(
            query_embedding,
            dtype=np.float32
        ),
        top_k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        retrieved_chunks.append(
            chunks[idx]
        )

    return retrieved_chunks


# =========================================================
# GENERATE ANSWER USING OLLAMA
# =========================================================

def generate_answer(
    question,
    context
):

    prompt = f"""
You are a healthcare AI assistant.

Answer ONLY using the healthcare context provided below.

If the answer is not available in the context, say:
'I could not find this information in the document.'

Healthcare Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(

        model=OLLAMA_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return answer


# =========================================================
# UI
# =========================================================

st.title(
    "🩺 Healthcare RAG"
)

st.write(
    "Upload a PDF and ask questions."
)







# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload  PDF",
    type=["pdf"]
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    pdf_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:

        f.write(uploaded_file.read())

    st.success(
        "PDF Uploaded Successfully"
    )


    # =====================================================
    # PDF PROCESSING
    # =====================================================

    with st.spinner(
        "Processing PDF..."
    ):

        text = load_pdf(pdf_path)

        chunks = create_chunks(text)

        embeddings = create_embeddings(chunks)

        index = create_faiss_index(
            embeddings
        )

    st.success(
        "PDF Processed Successfully"
    )


    # =====================================================
    # QUESTION INPUT
    # =====================================================

    question = st.text_input(
        "Ask a Healthcare Question"
    )


    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    if st.button(
        "Generate Answer"
    ):

        if question.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Generating Answer..."
            ):

                retrieved_chunks = retrieve_chunks(
                    question,
                    index,
                    chunks
                )

                context = "\n".join(
                    retrieved_chunks
                )

                answer = generate_answer(
                    question,
                    context
                )

            # =============================================
            # DISPLAY ANSWER
            # =============================================

            st.subheader("Generated Answer")

            st.write(answer)


            # =============================================
            # DISPLAY RETRIEVED CONTEXT
            # =============================================

            with st.expander(
                "Retrieved Context"
            ):

                for i, chunk in enumerate(
                    retrieved_chunks
                ):

                    st.write(
                        f"Chunk {i+1}"
                    )

                    st.write(chunk)

                    st.write(
                        "--------------------------------"
                    )