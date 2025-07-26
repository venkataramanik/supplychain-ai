import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Multi-Modal & Green Logistics", layout="wide")

# --- TITLE ---
st.title("🌱 Multi-Modal & Green Logistics")

# --- BUSINESS CONTEXT ---
st.header("Business Problem")
st.markdown("""
Choosing between **road, rail, air, and ocean** involves balancing **cost, lead time, and carbon emissions**.  
Companies aiming for sustainability must optimize **multi-modal transportation** while minimizing CO₂ output 
and meeting delivery targets.

**KPIs Impacted:**
- **Total cost per shipment.**
- **CO₂ emissions per ton-mile.**
- **On-time delivery performance.**
""")

# --- WHY THIS APPROACH ---
st.header("Why This Approach?")
st.markdown("""
Multi-modal optimization is not just about cost — it's about **cost + sustainability + service**.  
This pilot:
- Demonstrates trade-offs between cost and emissions.
- Enables decision-making that aligns with corporate sustainability goals.
- Uses multi-objective optimization (MOO) to find **Pareto-optimal solutions**.
""")

# --- TOOLS USED ---
st.header("Tools Used")
st.markdown("""
- **PuLP (Linear Programming):** For multi-objective cost-emission optimization.
- **pandas:** To model mode-wise cost and emission factors.
- **Plotly:** For visualizing cost vs. CO₂ trade-offs.
""")

# --- MATH BEHIND IT ---
st.header("Math Behind It")
st.markdown(r"""
Multi-modal routing can be modeled as a **multi-objective optimization** problem:

\[
Minimize \ (Cost, CO₂)
\]

Subject to:
\[
x_{road} + x_{rail} + x_{air} + x_{ocean} = Demand
\]
Where **\(x_{mode}\)** represents the share of shipments by each mode.
""")

# --- AI & ML ANGLE ---
st.header("How AI & ML Enhance Green Logistics")
st.markdown("""
- **ML Models:** Predict carbon impact of different mode combinations.
- **LLMs:** Summarize regulatory compliance (e.g., IMO 2023 emissions rules).
- **Reinforcement Learning:** Optimize multi-modal planning under uncertain demand.
""")

# --- FUN FACT ---
st.header("Fun Fact")
st.markdown("""
> **Maersk and DHL** now report **real-time CO₂ emissions per shipment**, 
using algorithms that combine routing and sustainability analytics.
""")

# --- DEMO PLACEHOLDER ---
st.header("Demo")
st.info("Interactive multi-modal optimization demo **coming soon**.")
