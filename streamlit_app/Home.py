import streamlit as st

st.set_page_config(page_title="SupplyChain.ai", layout="wide")

st.title("🚀 SupplyChain.ai")
st.subheader("AI-Powered Optimization Pilots for Supply Chain & Logistics")

st.markdown("""
Welcome to **SupplyChain.ai**, a portfolio of prototypes showcasing how 
**AI, Machine Learning, and Optimization** can solve real-world supply chain challenges.

This portfolio demonstrates:
- **Transportation Optimization:** Mode selection, rating, and vehicle routing.
- **Network Design:** Facility location and cost-service trade-offs.
- **Scenario Simulations:** Tariff impact and dynamic re-routing.

**Explore the pilots below:**
""")

st.divider()

# --- LINK TO TRANSPORTATION SUITE ---
st.header("Transportation Optimization Suite")
st.page_link("streamlit_app/TransportationSuite.py", label="🚛 Go to Transportation Optimization Suite")

# Optional: Direct links to sub-pilots
st.markdown("""
**Direct Pilot Links:**
- [Rating & Mode Selection](ShipmentPlanner_Basic.py)
- [Multi-Stop Routing (VRP)](ShipmentPlanner_VRP.py)
- [Time Windows & SLA (VRPTW)](ShipmentPlanner_VRPTW.py)
- [Cross-Dock & Multi-Echelon](ShipmentPlanner_CrossDock.py)
- [Multi-Modal & Green Logistics](ShipmentPlanner_MultiModal.py)
- [Dynamic Re-Routing](ShipmentPlanner_Dynamic.py)
""")

st.info("More pilots and demos coming soon.")
