import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dynamic Re-Routing (Real-Time)", layout="wide")
# Navigation link to go back
st.page_link("../TransportationSuite.py", label="⬅ Back to Transportation Suite")


# --- TITLE ---
st.title("⚡ Dynamic Re-Routing (Real-Time Optimization)")

# --- BUSINESS CONTEXT ---
st.header("Business Problem")
st.markdown("""
Static transportation plans often fail due to **real-time disruptions** like traffic jams, weather, 
vehicle breakdowns, or last-minute orders. Companies such as **ride-sharing platforms and 
e-commerce giants** rely on **dynamic routing engines** to adjust plans on the fly.

**KPIs Impacted:**
- **Average delivery delay time.**
- **Fleet utilization %.**
- **Customer satisfaction (NPS).**
""")

# --- WHY THIS APPROACH ---
st.header("Why This Approach?")
st.markdown("""
Dynamic re-routing enables:
- **Real-time adjustment of routes** as conditions change.
- **Proactive avoidance** of delays and SLA violations.
- **Higher asset utilization** and customer service improvements.
""")

# --- TOOLS USED ---
st.header("Tools Used")
st.markdown("""
- **OR-Tools (Dynamic VRP):** For continuous re-optimization.
- **SimPy or simulation libraries:** To simulate dynamic events.
- **pandas:** To handle route and vehicle state data.
""")

# --- MATH BEHIND IT ---
st.header("Math Behind It")
st.markdown(r"""
Dynamic VRP is often modeled using **rolling horizon optimization** or **reinforcement learning (RL):**

\[
\pi^* = \arg\max_{\pi} \mathbb{E}[ \sum_{t=0}^{T} r_t ]
\]
Where:
- **\(\pi\)** is the routing policy.
- **\(r_t\)** is the reward (e.g., minimized delay or cost) at time step t.
""")

# --- AI & ML ANGLE ---
st.header("How AI & ML Enhance Dynamic Routing")
st.markdown("""
- **Predictive Models:** Real-time ETA predictions using ML models (XGBoost, LSTMs).
- **Reinforcement Learning:** Continuously learns optimal routing adjustments.
- **LLMs:** Generate quick "what-if" re-routing scenarios when disruptions occur.
""")

# --- FUN FACT ---
st.header("Fun Fact")
st.markdown("""
> **Uber's dispatch engine** dynamically reassigns drivers every few seconds, 
solving a complex VRP variant with real-time events and stochastic demands.
""")

# --- DEMO PLACEHOLDER ---
st.header("Demo")
st.info("Interactive dynamic re-routing demo **coming soon**.")
