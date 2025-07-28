import streamlit as st
from st_pages import Page, Section, add_page_title, add_pages_with_data
import streamlit_shadcn_ui as ui

# Set page layout
st.set_page_config(page_title="SupplyChain.ai", layout="wide")
add_page_title()

# Sidebar navigation setup
add_pages_with_data(
    [
        Page("home.py", "🏠 Home", "house"),

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

# --------------------------------------------------------------------------------
# 🏷️ Header Section with Shadcn Badges
# --------------------------------------------------------------------------------
ui.badges(
    badge_list=[("SupplyChain.ai", "default"), ("AI/ML", "secondary"), ("Optimization", "destructive")],
    class_name="mb-2 flex gap-2",
    key="home_badges"
)
st.header("SupplyChain.ai")
st.caption("AI-powered optimization pilots for logistics & supply chain leaders")

# --------------------------------------------------------------------------------
# 🚀 Welcome & CTA
# --------------------------------------------------------------------------------
st.markdown("### 👋 Welcome")
st.markdown("""
Welcome to **SupplyChain.ai** — a curated set of interactive pilots that demonstrate how  
**AI, Machine Learning, and Optimization** can solve complex supply chain and logistics challenges.
""")

with ui.element("div", className="flex gap-2 mt-2", key="cta_buttons"):
    ui.element("button", text="Get Started", className="btn btn-primary", key="btn_start")
    ui.element("link_button", text="GitHub", url="https://github.com/your-org/your-repo", variant="outline", key="btn_github")

# --------------------------------------------------------------------------------
# 🧠 About the Creator
# --------------------------------------------------------------------------------
st.markdown("### 👤 About Me")
st.markdown("""
Hi, I’m **Venkat Krishnan**, a supply chain transformation leader with 20+ years of experience.  
I'm passionate about applying **business strategy, AI/ML, and optimization** to problems in transportation, warehousing, and global trade.
""")

# --------------------------------------------------------------------------------
# 🧭 What's Inside
# --------------------------------------------------------------------------------
st.markdown("### 🗂️ What’s Inside")
st.markdown("""
This portfolio includes:
- 🚛 **Transportation Optimization Suite**  
- ⏰ **Time Window & VRP Planning Tools**  
- 🌐 **Network Design & Cross-Docking Models**  
- 📊 **Analytics: Supplier Risk & Demand Volatility**

👉 Use the **sidebar** to explore each solution!
""")

# --------------------------------------------------------------------------------
# 💡 Why This Matters
# --------------------------------------------------------------------------------
st.markdown("### 💡 Why I Built This")
st.markdown("""
I created **SupplyChain.ai** to showcase:
- How to build AI pilots *fast* using Streamlit and Python
- The power of optimization to drive **cost savings and service levels**
- The potential of **open-source libraries** (like OR-Tools, Pyomo, and Pandas) in real-world supply chain use cases
""")

# --------------------------------------------------------------------------------
# 📬 Footer
# --------------------------------------------------------------------------------
st.markdown("---")
st.caption("🔗 [LinkedIn](https://www.linkedin.com/in/venkrish1/) | Built with 💡 using Streamlit & Shadcn UI")
