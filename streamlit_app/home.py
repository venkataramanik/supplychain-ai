import streamlit as st
from st_pages import Page, Section, add_page_title, show_pages

st.set_page_config(page_title="SupplyChain.ai", layout="wide")

# Sidebar navigation
add_page_title()
show_pages([
    Page("home.py", "🏠 Home", "house"),
    Section("⚙️ Optimization Models", "gear"),
    Page("pages/1_TransportationSuite.py", "Transportation Suite", "🚛"),
    Page("pages/2_ShipmentPlanner_Basic.py", "Shipment Planner – Basic", "📦"),
    Page("pages/3_ShipmentPlanner_VRP.py", "VRP Optimization", "🛻"),
    Page("pages/4_ShipmentPlanner_VRPTW.py", "VRPTW – Time Windows", "⏱"),
    Page("pages/5_ShipmentPlanner_MultiModal.py", "Multi-Modal Planner", "🚢"),
    Page("pages/6_ShipmentPlanner_CrossDock.py", "Cross-Dock Routing", "🔄"),
    Page("pages/7_ShipmentPlanner_Dynamic.py", "Dynamic Re-Routing", "⚡"),
    Page("pages/8_NetworkDesign.py", "Network Design", "🌐"),
    Section("📊 Analytics", "chart-line"),
    Page("pages/9_DemandVolatility.py", "Demand Volatility", "📈"),
    Page("pages/10_SupplierRisk.py", "Supplier Risk", "🛡️")
])
