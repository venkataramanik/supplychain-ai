import streamlit as st
from st_pages import Page, Section, add_page_title, add_pages_with_data

st.set_page_config(page_title="SupplyChain.ai", layout="wide")

# This adds the title to the current page (Home) and sets up the sidebar structure.
# It MUST be called before any other st.sidebar calls or page_link calls.
add_page_title()

# Define your pages and their structure using st_pages
add_pages_with_data(
    [
        # Main Home Page (this file itself)
        Page("app.py", "🏠 Home", "house"), # The icon is optional

        # Optimization Models Section
        Section("⚙️ Optimization Models", "gear"),
        Page("pages/TransportationSuite.py", "🚛 Transportation Optimization Suite"),
        Page("pages/ShipmentPlanner_Basic.py", "📍 Basic Shipment Planner"),
        Page("pages/ShipmentPlanner_VRP.py", "🚚 Vehicle Routing Problem (VRP)"),
        Page("pages/ShipmentPlanner_VRPTW.py", "⏰ VRPTW (Time Windows)"),
        Page("pages/ShipmentPlanner_MultiModal.py", "🚢 Multi-Modal Shipment Planner"),
        Page("pages/ShipmentPlanner_CrossDock.py", "🔄 Cross-Dock & Multi-Echelon Routing"),
        Page("pages/ShipmentPlanner_Dynamic.py", "⚡ Dynamic Re-Routing"),
        Page("pages/NetworkDesign.py", "🌐 Network Design & Optimization"),
        
        # Data Analysis & Insights Section
        Section("📊 Data Analysis & Insights", "bar_chart"),
        Page("pages/DemandVolatility.py", "📈 Demand Volatility Assessment"),
        Page("pages/SupplierRisk.py", "🛡️ Supplier Performance & Risk Profiling"),
    ]
)

# The rest of your main content for the Home page
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
