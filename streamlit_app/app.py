import streamlit as st
from st_pages import Page, Section, add_page_title, add_pages_with_data
import streamlit_shadcn_ui as ui

st.set_page_config(page_title="SupplyChain.ai", layout="wide")

# Set up sidebar navigation
add_page_title()
add_pages_with_data(
    [
        Page("app.py", "🏠 Home", "house"),
        Section("⚙️ Optimization Models", "gear"),
        Page("pages/TransportationSuite.py", "🚛 Transportation Optimization Suite"),
        Page("pages/ShipmentPlanner_Basic.py", "📍 Basic Shipment Planner"),
        Page("pages/ShipmentPlanner_VRP.py", "🚚 Vehicle Routing Problem (VRP)"),
        Page("pages/ShipmentPlanner_VRPTW.py", "⏰ VRPTW (Time Windows)"),
        Page("pages/ShipmentPlanner_MultiModal.py", "🚢 Multi-Modal Shipment Planner"),
        Page("pages/ShipmentPlanner_CrossDock.py", "🔄 Cross-Dock & Multi-Echelon Routing"),
        Page("pages/ShipmentPlanner_Dynamic.py", "⚡ Dynamic Re-Routing"),
        Page("pages/NetworkDesign.py", "🌐 Network Design & Optimization"),
        Section("📊 Data Analysis & Insights", "bar_chart"),
        Page("pages/DemandVolatility.py", "📈 Demand Volatility Assessment"),
        Page("pages/SupplierRisk.py", "🛡️ Supplier Performance & Risk Profiling"),
    ]
)

# -------------------------------
# ✨ Main UI Styling with Shadcn
# -------------------------------
ui.badges(
    badge_list=[("SupplyChain.ai", "default"), ("AI", "secondary"), ("Optimization", "destructive")],
    class_name="mb-2 flex gap-2",
    key="home_badges"
)

st.header("SupplyChain.ai")
ui.text_block("AI-Powered Optimization Pilots for Supply Chain & Logistics", class_name="text-lg text-muted-foreground mb-4")

# Welcome Block
with ui.card(title="Welcome", key="card_welcome") as c:
    st.markdown("""
    Welcome to **SupplyChain.ai** — a portfolio of interactive prototypes demonstrating how  
    **AI, Machine Learning, and Optimization** can tackle real-world supply chain and logistics challenges.
    """)

# About Me Block
with ui.card(title="About Me", key="card_about") as c:
    st.markdown("""
    Hi, I’m **Venkat Krishnan**, a supply chain transformation leader with 20+ years of experience.  
    I am passionate about blending **business strategy with AI/ML and optimization** to solve  
    challenges in transportation, warehousing, and global trade.

    This portfolio highlights:
    - **Business problem framing** — turning logistics pain points into optimization problems.
    - **Rapid prototyping** using open-source AI/ML and optimization libraries.
    - **Storytelling with data** — showing how these solutions create measurable business impact.

    [**Connect with me on LinkedIn**](https://www.linkedin.com/in/venkrish1/)
    """)

# Why I Built This Block
with ui.card(title="Why I Built This", key="card_why") as c:
    st.markdown("""
    I created **SupplyChain.ai** to showcase:
    - How quickly we can build **AI-powered pilots** for supply chain problems.
    - The **business value of optimization** (cost savings, SLA compliance, network efficiency).
    - The potential of **open-source tools** like Python, OR-Tools, and Streamlit for enterprise solutions.
    """)

# Explore Pilots Block
with ui.card(title="Explore Pilots", key="card_pilots") as c:
    st.markdown("Please use the **sidebar navigation** to explore the different AI-powered pilots.")

# Optional: footer
st.markdown("---")
