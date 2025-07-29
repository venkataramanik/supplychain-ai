# rag_contract_summarizer.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # Changed from AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap # For pretty printing long texts

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="📄 Supply Chain Contract Q&A (RAG)")

st.title("📄 Supply Chain Contract Q&A (RAG)")

# --- Business Context & Highlights ---
st.header("🌟 Business Context & Highlights")
st.markdown("""
- **Problem:** Procurement, legal, and operational teams spend countless hours manually sifting through complex supplier contracts, legal documents, and internal policies to extract specific clauses, verify terms, or answer compliance questions. This is time-consuming, prone to human error, and delays critical decision-making.
- **Solution:** This pilot demonstrates how **Python + Open-Source RAG** can build an intelligent system that quickly retrieves and summarizes relevant information from a vast repository of enterprise documents, directly addressing natural language queries.
- **Impact:** By leveraging the power of Generative AI grounded in proprietary data, we can transform static document archives into dynamic, queryable knowledge bases, empowering faster, more accurate, and consistent access to critical information.
""")

st.subheader("💡 Why This Problem Matters")
st.markdown("""
- **Risk Mitigation:** Ensures quick access to crucial clauses (e.g., force majeure, liability, termination) during disruptions, protecting the business from financial and operational risks.
- **Operational Efficiency:** Drastically reduces the manual effort and time spent by procurement, legal, and compliance teams on document review and information extraction.
- **Negotiation Power:** Enables procurement to rapidly access historical contract terms, pricing structures, and performance clauses to inform better negotiation strategies.
- **Compliance & Audit:** Provides verifiable, grounded answers directly from source documents, simplifying internal audits and ensuring adherence to regulatory requirements.
- **New Employee Onboarding:** Accelerates the learning curve for new team members by providing instant, accurate answers to policy and contract-related questions.
""")

st.subheader("📈 KPIs Improved")
st.markdown("""
| KPI Category | Improvement |
| :------------------------- | :-------------------------------------------------------------- |
| **Information Retrieval Time** | Reduce time to find a specific clause from minutes to seconds.    |
| **Contract Review Cycle Time** | Speed up initial review of new supplier contracts by X%.         |
| **Query Resolution Accuracy** | Higher percentage of natural language questions answered correctly. |
| **"Hallucination" Rate** | Near-zero instances of AI generating factually incorrect info.  |
| **Employee Productivity** | Quantifiable hours saved by teams no longer manually searching. |
| **Compliance Adherence** | Improved ability to quickly verify and demonstrate adherence.   |
""")

st.subheader("🛠️ Tech Stack (Pilot)")
st.markdown("""
- **Python:** For core RAG pipeline orchestration, data processing, and integration logic.
- **`transformers`:** To load and manage the Large Language Model (LLM) for generation.
- **`sentence-transformers`:** For generating high-quality semantic embeddings (numerical representations) of text.
- **`faiss-cpu`:** (Facebook AI Similarity Search) As a lightweight, in-memory vector database for fast similarity search.
- **`torch`:** The underlying deep learning framework powering the models.
- **Streamlit:** For building an instant, interactive web user interface.
""")

st.divider() # Visual separator

# --- 1. Simulate Data (Knowledge Base: Supplier Contracts) ---
# In a real-world scenario, these would be loaded from actual contract documents (PDFs, DOCX, etc.)
# and preprocessed (e.g., using PyPDF2, unstructured, or similar libraries).

CONTRACT_DOCUMENTS = [
    {
        "id": "contract_001",
        "title": "Master Supply Agreement - Supplier A",
        "text": """
        This Master Supply Agreement ("Agreement") is entered into on January 1, 2023,
        between Acme Corp (Buyer) and Global Parts Inc. (Supplier A).

        **Section 3.1: Payment Terms.** Buyer shall pay Supplier A within 60 days of receipt
        of a valid invoice. Invoices shall be submitted monthly for goods shipped.
        Late payments will incur a penalty of 1.5% per month.

        **Section 4.2: Lead Time.** Standard lead time for all orders is 30 calendar days
        from the date of purchase order confirmation. Expedited orders may be
        negotiated at a premium, reducing lead time to 15 days, subject to availability.

        **Section 7.1: Force Majeure.** Neither party shall be liable for any failure or delay
        in performance under this Agreement due to causes beyond its reasonable control,
        including, but not limited to, acts of God, war, terrorism, riots, embargoes,
        fires, floods, earthquakes, or other natural disasters, strikes, or government regulations.
        The affected party must notify the other party within 5 business days of the event.

        **Section 9.0: Termination.** Either party may terminate this Agreement with 90 days
        written notice for convenience. Termination for cause requires 30 days notice
        and an opportunity to cure the breach.
        """
    },
    {
        "id": "contract_002",
        "title": "Service Level Agreement - Logistics Partner B",
        "text": """
        This Service Level Agreement ("SLA") is effective from February 15, 2024,
        between Acme Corp (Client) and Swift Logistics Solutions (Logistics Partner B).

        **Article 2: Delivery Performance.** Swift Logistics Solutions commits to an
        on-time delivery rate of 98% for all shipments within the continental US.
        Failure to meet this KPI for two consecutive quarters will result in a 5%
        reduction in monthly service fees.

        **Article 3: Communication Protocol.** All critical shipment updates (delays,
        damage, re-routes) must be communicated via automated email and SMS within
        1 hour of the event occurring. A dedicated account manager will be available
        during business hours (9 AM - 5 PM EST).

        **Article 5: Insurance Coverage.** Shipments are insured up to $10,000 per incident
        by Swift Logistics Solutions. Higher value shipments require separate,
        additional insurance arranged by the Client.

        **Article 8: Dispute Resolution.** Any disputes arising from this SLA shall first
        be attempted to be resolved through good-faith negotiation. If unresolved
        within 30 days, disputes will proceed to binding arbitration in Delaware.
        """
    },
    {
        "id": "contract_003",
        "title": "Software License Agreement - ERP Vendor C",
        "text": """
        This Software License Agreement ("SLA") is dated March 10, 2023,
        between Acme Corp and Innovate ERP Systems (Vendor C).

        **Clause 2.1: Licensing Fees.** Annual licensing fees for the ERP software
        are $150,000, payable in advance on the anniversary date of this agreement.
        Fees are subject to a 3% annual increase.

        **Clause 3.2: Support & Maintenance.** Vendor C provides 24/7 technical support
        with a guaranteed response time of 4 hours for critical issues.
        Software updates and patches are included in the annual fee.

        **Clause 7.0: Data Privacy.** Vendor C agrees to comply with all applicable
        data privacy regulations, including GDPR and CCPA, regarding any data
        processed through the ERP system. Data will not be shared with third parties
        without explicit consent.
        """
    }
]

# --- 2. RAG Pipeline Components ---

# 2.1. Document Preprocessing and Chunking
# For simplicity, we'll chunk by paragraphs/sections.
def chunk_document(doc_id, doc_title, doc_text, chunk_size_chars=500):
    chunks = []
    current_chunk = ""
    sentences = doc_text.split('\n\n') # Split by paragraph for this example

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 2 < chunk_size_chars: # +2 for newline
            current_chunk += (sentence + "\n\n")
        else:
            if current_chunk:
                chunks.append({
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "text": current_chunk.strip()
                })
            current_chunk = sentence + "\n\n"

    if current_chunk: # Add the last chunk
        chunks.append({
            "doc_id": doc_id,
            "doc_title": doc_title,
            "text": current_chunk.strip()
        })
    return chunks

all_chunks = []
for doc in CONTRACT_DOCUMENTS:
    all_chunks.extend(chunk_document(doc["id"], doc["title"], doc["text"]))


# 2.2. Embedding Model (Retriever)
# Use st.cache_resource to cache the model loading
@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# Load the embedding model globally (this function is already cached)
embedding_model_instance = load_embedding_model()


# 2.3. Vector Store (FAISS Index)
# Use st.cache_resource to cache the FAISS index creation
@st.cache_resource
def build_faiss_index(texts): # Removed 'model' parameter
    # Get the cached embedding model instance inside the function
    model = load_embedding_model() # This ensures the cached model is used
    embeddings = model.encode(texts, convert_to_tensor=False) # Get numpy array
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # L2 distance for similarity search
    index.add(embeddings)
    return index

# Pass only the hashable list of text content for building the index
faiss_index = build_faiss_index([chunk["text"] for chunk in all_chunks])


# 2.4. LLM for Generation
# Use st.cache_resource to cache the LLM loading
@st.cache_resource
def load_llm():
    # Using a small T5 model for text generation, suitable for CPU/limited GPU
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # FIX: Changed from AutoModelForCausalLM to AutoModelForSeq2SeqLM for T5 models
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

llm_tokenizer, llm_model = load_llm()


# --- 3. RAG Pipeline Function ---
def run_rag_pipeline(query, top_k=3):
    """
    Executes the RAG pipeline:
    1. Embeds the query.
    2. Retrieves top_k relevant chunks from FAISS.
    3. Constructs a prompt with retrieved context.
    4. Generates an answer using the LLM.
    """
    # Use the globally loaded (and cached) embedding model instance
    query_embedding = embedding_model_instance.encode([query], convert_to_tensor=False)

    # 2. Retrieve top_k relevant chunks
    distances, indices = faiss_index.search(query_embedding, top_k)
    # Use all_chunks here to get the original dictionary (with doc_id, doc_title, text)
    retrieved_chunks = [all_chunks[i] for i in indices[0]]

    # 3. Construct prompt for LLM
    context = "\n\n".join([f"Document: {chunk['doc_title']}\nContent: {chunk['text']}" for chunk in retrieved_chunks])

    prompt = f"""
    You are a helpful assistant specialized in supply chain contracts.
    Answer the following question based ONLY on the provided context.
    If the answer cannot be found in the context, state that you don't have enough information.

    Context:
    {context}

    Question: {query}

    Answer:
    """

    # 4. Generate answer using LLM
    input_ids = llm_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).input_ids

    llm_model.eval()
    with torch.no_grad():
        output_ids = llm_model.generate(
            input_ids,
            max_new_tokens=200,
            num_beams=5,
            early_stopping=True
        )

    answer = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return answer, retrieved_chunks

# --- Streamlit UI for RAG Interaction ---

st.header("1. Knowledge Base (Simulated Contracts)")
st.info(f"Loaded {len(CONTRACT_DOCUMENTS)} simulated contracts and chunked them into {len(all_chunks)} pieces.")

# Using an expander to keep the contract list tidy
with st.expander("View Glimpse of Contracts"):
    for doc in CONTRACT_DOCUMENTS:
        st.markdown(f"**{doc['title']}** (`{doc['id']}`)")
        st.code(textwrap.fill(doc['text'][:200] + '...', width=80), language='text') # Show first 200 chars
        st.markdown("---")

st.header("2. Ask a Question about the Contracts")
user_query = st.text_area(
    "Enter your question about the contracts:",
    "What are the payment terms for Supplier A and what happens if I pay late?"
)

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Retrieving and generating answer..."):
            answer, retrieved_chunks = run_rag_pipeline(user_query)

        st.subheader("🤖 Generated Answer:")
        st.success(answer)

        st.subheader("🔍 Retrieved Context (for Transparency):")
        st.info("The following sections from the contracts were used to generate the answer:")
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Chunk {i+1} from '{chunk['doc_title']}' (`{chunk['doc_id']}`):**")
            st.code(textwrap.fill(chunk['text'], width=100), language='text') # Wrap text for better display
            st.markdown("---")
    else:
        st.warning("Please enter a question.")

st.divider() # Visual separator

# --- Next Steps Section ---
st.header("🚀 Next Steps (Scaling with High-Performance AI & Advanced Capabilities)")
st.markdown("""
This demonstration uses open-source components that can run on a CPU for simplicity.
However, for real-world enterprise and supply chain problems with massive datasets and
demanding performance requirements, **specialized hardware and optimized software stacks** are essential.

Here's how they would elevate this RAG pipeline:
""")

st.subheader("Hardware & Performance")
st.markdown("""
- **Enterprise-Scale Vector Database:** Transition from in-memory `faiss-cpu` to a robust, accelerated vector database (e.g., solutions utilizing powerful GPUs) to handle millions/billions of documents and support real-time ingestion/updates.
- **Larger, Optimized LLMs:** Integrate more powerful and nuanced LLMs (e.g., Llama 3, Mixtral) for superior understanding and generation, served efficiently via optimized inference engines (e.g., using libraries like TensorRT-LLM) for low-latency, high-throughput inference on high-performance computing platforms.
""")

st.subheader("Advanced Data & Models")
st.markdown("""
- **Multimodal Document Understanding:** Enhance the data ingestion pipeline using advanced document intelligence tools to accurately extract information from complex contract formats including tables, charts, scanned PDFs, and images, ensuring no valuable information is missed.
- **Custom Embedding Models:** Potentially fine-tune or train custom embedding models with domain-specific contract data to further improve retrieval relevance for highly specialized legal terminology.
- **Advanced Reranking:** Implement sophisticated reranking models to ensure the absolute most relevant chunks are presented to the LLM, even among highly similar candidates.
""")

st.subheader("Integration & MLOps")
st.markdown("""
- **Agentic Workflows Integration:** Embed this RAG system as a "tool" within a larger **AI Agent** (e.g., a "Procurement Agent" or "Contract Compliance Agent") capable of autonomously querying, analyzing, and taking actions based on contract terms, streamlining complex, multi-step business processes.
- **Security & Access Control:** Integrate with enterprise security frameworks to ensure RAG responses respect user permissions and data confidentiality for sensitive contract information.
- **Monitoring & Evaluation (MLOps):** Implement robust monitoring for RAG performance (e.g., retrieval accuracy, generation quality, latency, hallucination rate) and set up pipelines for continuous improvement, leveraging enterprise-grade MLOps capabilities.
""")

st.markdown("""
By leveraging high-performance computing and specialized AI software, enterprises can build highly accurate, scalable, and responsive Generative AI solutions that truly solve complex supply chain problems.
""")
