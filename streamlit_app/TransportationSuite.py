import streamlit as st

st.set_page_config(page_title="Transportation Optimization Suite", layout="wide")

# --- INTRO ---
st.title("🚛 Transportation Optimization Suite")
st.markdown("""
Transportation networks face **multi-dimensional challenges** — rising costs, strict SLAs, sustainability goals, 
and increasing delivery complexity. Industry leaders solve these using a **blend of AI, ML, and optimization** to 
orchestrate millions of shipments in real time.

This suite showcases **business-first AI solutions** with a mix of:
- **Dynamic cost rating & mode selection** (Parcel vs LTL vs FTL).
- **Vehicle routing optimization (VRP & its variants).**
- **Cross-dock and multi-echelon planning.**
- **Multi-modal & sustainability trade-offs.**
- **Dynamic, event-driven re-routing.**
""")

st.divider()

# --- AI & ML LAYER ---
st.header("How AI & ML Enhance Transportation Planning")
st.markdown("""
- **Predictive ETA & Demand Models:** Machine learning (XGBoost, ARIMA, or deep learning) predicts delays and costs.
- **Reinforcement Learning:** Adaptive routing engines learn from real-time conditions (like ride-sharing systems).
- **Generative AI (LLMs):** Automates carrier rate table parsing and rapidly prototypes optimization models.
- **Graph Neural Networks (GNNs):** Cutting-edge approach for solving large-scale VRPs.
""")

st.divider()

# --- PILOT OVERVIEW ---
st.header("Pilots in This Suite")
st.markdown("""
1. **Rating & Mode Selection (Cost Engine):** Cost optimization using linear cost models.  
2. **VRP + Multi-Stop Routing:** Graph optimization for shortest route selection.  
3. **Time Windows & SLA (VRPTW):** On-time delivery optimization with penalty avoidance.  
4. **Cross-Dock & Multi-Echelon Routing:** Hub-and-spoke consolidation planning.  
5. **Multi-Modal & Green Logistics:** Cost vs. CO₂ trade-off optimization.  
6. **Dynamic Re-Routing:** Real-time dispatch adjustments.
""")

st.divider()

st.info("Each pilot includes **business context, math foundations, tool rationale, and a working interactive demo.**")

