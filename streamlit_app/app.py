import streamlit as st

st.set_page_config(page_title="SupplyChain.ai", layout="wide")

# ---------------------------------------------------------
# Custom Sidebar Navigation (This will be the ONLY sidebar content)
# ---------------------------------------------------------
with st.sidebar:
    st.title("SupplyChain.ai Pilots")
    st.markdown("---")

    st.subheader("⚙️ Optimization Models")
    st.page_link("pages/TransportationSuite.py", label="🚛 Transportation Optimization Suite")
    st.page_link("pages/ShipmentPlanner_Basic.py", label="📍 Basic Shipment Planner")
    st.page_link("pages/ShipmentPlanner_VRP.py", label="🚚 Vehicle Routing Problem (VRP)")
    st.page_link("pages/ShipmentPlanner_VRPTW.py", label="⏰ VRPTW (Time Windows)")
    st.page_link("pages/ShipmentPlanner_MultiModal.py", label="🚢 Multi-Modal Shipment Planner")
    st.page_link("pages/ShipmentPlanner_CrossDock.py", label="🔄 Cross-Dock & Multi-Echelon Routing")
    st.page_link("pages/ShipmentPlanner_Dynamic.py", label="⚡ Dynamic Re-Routing")
    st.page_link("pages/NetworkDesign.py", label="🌐 Network Design & Optimization")
    
    st.markdown("---")
    
    st.subheader("📊 Data Analysis & Insights")
    st.page_link("pages/DemandVolatility.py", label="📈 Demand Volatility Assessment")
    st.page_link("pages/SupplierRisk.py", label="🛡️ Supplier Performance & Risk Profiling")

    st.markdown("---")
    # Link back to the main content of this app (streamlit_app.py itself)
    st.page_link("streamlit_app.py", label="🏠 Back to Home Page") 


# ---------------------------------------------------------
# Main Content (Home Page)
# ---------------------------------------------------------
st.title("SupplyChain.ai")
st.subheader("AI-Powered Optimization Pilots for Supply Chain & Logistics")

st.markdown("""
Welcome to **SupplyChain.ai** — a portfolio of interactive prototypes demonstrating how  
**AI, Machine Learning, and Optimization** can tackle real-world supply chain and logistics challenges.
""")

# ---------------------------------------------------------
# About Me Section
# ---------------------------------------------------------
st.markdown("""
---

### **About Me**
Hi, I’m **Venkat Krishnan**, a supply chain transformation leader with 20+ years of experience.  
I am passionate about blending **business strategy with AI/ML and optimization** to solve  
challenges in transportation, warehousing, and global trade.

This portfolio highlights:
- **Business problem framing** — turning logistics pain points into optimization problems.
- **Rapid prototyping** using open-source AI/ML and optimization libraries.
- **Storytelling with data** — showing how these solutions create measurable business impact.

[**Connect with me on LinkedIn**](https://www.linkedin.com/in/venkrish1/)
""")

# ---------------------------------------------------------
# Why I Built This
# ---------------------------------------------------------
st.markdown("""
---

### **Why I Built This**
I created **SupplyChain.ai** to showcase:
- How quickly we can build **AI-powered pilots** for supply chain problems.
- The **business value of optimization** (cost savings, SLA compliance, network efficiency).
- The potential of **open-source tools** like Python, OR-Tools, and Streamlit for enterprise solutions.
""")

# ---------------------------------------------------------
# Explore Pilots (This section is now redundant as navigation is in sidebar)
# ---------------------------------------------------------
st.markdown("---")
st.markdown("## **Explore the Pilots**")
st.info("Please use the **sidebar navigation** to explore the different AI-powered pilots.")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
---

""", unsafe_allow_html=True)
