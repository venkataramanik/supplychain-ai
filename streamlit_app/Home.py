import streamlit as st

st.set_page_config(page_title="SupplyChain.ai", layout="wide")

st.title("🚀 SupplyChain.ai")
st.subheader("AI-Powered Optimization Pilots for Supply Chain & Logistics")

st.markdown("""
Welcome to **SupplyChain.ai**, a portfolio of prototypes showcasing how 
**AI, Machine Learning, and Optimization** can solve real-world supply chain challenges.

Explore the key pilots below:
""")

st.divider()

# --- LINK TO TRANSPORTATION SUITE ---
st.header("Transportation Optimization Suite")
st.page_link("TransportationSuite.py", label="🚛 Go to Transportation Optimization Suite")

st.divider()

# --- DIRECT LINKS TO PILOTS ---
st.header("Direct Pilot Links")
st.page_link("pages/ShipmentPlanner_Basic.py", label="1. Rating & Mode Selection (Cost Engine)")
st.page_link("pages/ShipmentPlanner_VRP.py", label="2. Multi-Stop Routing (VRP)")
st.page_link("pages/ShipmentPlanner_VRPTW.py", label="3. Time Windows & SLA (VRPTW)")
st.page_link("pages/ShipmentPlanner_CrossDock.py", label="4. Cross-Dock & Multi-Echelon")
st.page_link("pages/ShipmentPlanner_MultiModal.py", label="5. Multi-Modal & Green Logistics")
st.page_link("pages/ShipmentPlanner_Dynamic.py", label="6. Dynamic Re-Routing")

st.info("More pilots and demos coming soon.")
