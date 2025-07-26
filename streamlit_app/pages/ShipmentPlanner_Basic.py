import streamlit as st

st.set_page_config(page_title="Rating & Mode Selection", layout="wide")

# Navigation
st.page_link("pages/TransportationSuite.py", label="⬅ Back to Transportation Suite")
st.page_link("Home.py", label="🏠 Back to Home")

# Image
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/2/21/I40TruckCTI.JPG",
    caption="Parcel vs LTL Shipping Modes",
    width=700  # Adjust size as needed (e.g., 500-800)
)




# --- TITLE ---
st.title("🚚 Rating & Mode Selection (Cost Engine)")

# --- BUSINESS CONTEXT ---
st.header("Business Problem")
st.markdown("""
Transportation costs can account for **30-40% of logistics expenses**. 
Companies often **choose sub-optimal modes (Parcel vs LTL vs FTL)** due to static rules or outdated pricing models, 
leading to cost inefficiencies.

**Key Challenges:**
- Manual mode selection leads to overpaying for premium modes.
- Lack of dynamic cost visibility (weight, volume, distance).
- Inability to simulate "what-if" scenarios quickly.

**KPIs Impacted:**
- **Cost per shipment** (primary metric).
- **Carrier selection accuracy** (better matching of service level vs cost).
- **Cost-to-service ratio** (overall logistics efficiency).
""")

# --- WHY THIS APPROACH ---
st.header("Why This Approach?")
st.markdown("""
We start with **rating and mode selection** because:
- **Rating is the foundation**: Carriers determine cost based on weight, distance, and base rates.
- **Mode optimization (Parcel vs. LTL)** can reduce costs by **10–25%** for typical mid-size networks.
- It demonstrates how simple cost models and linear equations can yield measurable ROI.
""")

# --- TOOLS USED ---
st.header("Tools Used")
st.markdown("""
- **Streamlit:** For interactive dashboards.
- **pandas:** To handle shipment datasets.
- **Python cost functions:** For transparent rating logic.
- **OpenAI (LLM):** Accelerated the prototyping of this solution.
""")

# --- MATH BEHIND IT ---
st.header("Math Behind It")
st.markdown(r"""
The cost for each mode is modeled as:

\[
Cost = Base + \alpha \cdot Weight + \beta \cdot Distance
\]

Where:
- **Base** = fixed cost per shipment.
- **\(\alpha\)** = cost per unit of weight.
- **\(\beta\)** = cost per unit of distance.
""")

# --- AI & ML ANGLE ---
st.header("How AI & ML Enhance Rating")
st.markdown("""
- **ML Models:** Predict dynamic rates using historical cost data and demand patterns.
- **LLMs:** Automatically parse carrier tariff PDFs and generate cost tables.
- **Generative AI:** Rapidly prototypes cost models and business logic (like this demo).
""")

# --- FUN FACT ---
st.header("Fun Fact")
st.markdown("""
> UPS reduced millions in operational costs by applying routing and cost optimization, 
including policies like **minimizing left turns** — a simple yet powerful optimization.
""")

# --- DEMO PLACEHOLDER ---
st.header("Demo")
st.info("Interactive cost rating demo **coming soon**.")

