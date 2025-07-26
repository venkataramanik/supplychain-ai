import streamlit as st

st.set_page_config(page_title="Cross-Dock & Multi-Echelon Routing", layout="wide")

# Navigation
st.page_link("pages/TransportationSuite.py", label="⬅ Back to Transportation Suite")
st.page_link("Home.py", label="🏠 Back to Home")

# Image
st.image("https://upload.wikimedia.org/wikipedia/commons/e/e4/Warehouse_icon.png",
         caption="Cross-docking and multi-echelon logistics", use_container_width=True)



# --- TITLE ---
st.title("🔄 Cross-Dock & Multi-Echelon Routing")

# --- BUSINESS CONTEXT ---
st.header("Business Problem")
st.markdown("""
Shipments often pass through **hubs or cross-dock facilities** before reaching the end customer. 
Optimizing the **line-haul (Plant → Hub)** and **last-mile (Hub → Customer)** routes together 
is significantly more complex than planning direct deliveries.

**KPIs Impacted:**
- **Hub utilization and throughput.**
- **Line-haul cost reduction.**
- **Lead time and order cycle time.**
""")

# --- WHY THIS APPROACH ---
st.header("Why This Approach?")
st.markdown("""
Cross-docking reduces warehousing needs and improves delivery speed, but it requires **coordinated routing** 
between upstream and downstream legs. Multi-echelon routing considers **multiple layers** of the network 
(e.g., Plant → DC → Cross-Dock → Customer).

This approach:
- Optimizes **end-to-end network flow**.
- Reduces **total transportation cost** and **inventory holding cost**.
""")

# --- TOOLS USED ---
st.header("Tools Used")
st.markdown("""
- **OR-Tools / PuLP:** For solving multi-echelon routing and consolidation.
- **NetworkX:** For visualizing hub-and-spoke networks.
- **pandas:** For managing network flow data.
""")

# --- MATH BEHIND IT ---
st.header("Math Behind It")
st.markdown(r"""
Multi-echelon routing can be modeled as a **network flow problem**:

\[
Minimize \ \sum_{i,j} C_{ij} x_{ij}
\]
Subject to:
\[
\sum_{j} x_{ij} = Demand_i
\]
\[
\sum_{i} x_{ij} = Supply_j
\]

Where **\(x_{ij}\)** is the flow from node i to node j.
""")

# --- AI & ML ANGLE ---
st.header("How AI & ML Enhance Cross-Docking")
st.markdown("""
- **ML Models:** Predict hub congestion and demand spikes for better planning.
- **LLMs:** Automate network design scenario generation.
- **Reinforcement Learning:** Continuously learn optimal consolidation and transfer strategies.
""")

# --- FUN FACT ---
st.header("Fun Fact")
st.markdown("""
> **Amazon's fulfillment network** is a multi-echelon system with cross-docks, 
regional hubs, and last-mile stations, all optimized with real-time algorithms.
""")

# --- DEMO PLACEHOLDER ---
st.header("Demo")
st.info("Interactive cross-dock routing demo **coming soon**.")
