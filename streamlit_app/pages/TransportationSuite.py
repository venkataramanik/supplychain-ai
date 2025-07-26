import streamlit as st

st.set_page_config(page_title="Transportation Optimization Suite", layout="wide")

st.title("🚛 Transportation Optimization Suite")
import os
st.write("Files in images folder:", os.listdir("images"))

st.image("../images/Supply-chain-shipping-modes.jpg",
         caption="Example of a transportation optimization network.",
         use_container_width=True)




st.markdown("""
This suite demonstrates **end-to-end transportation optimization**:
- **Rating & Mode Selection**
- **Multi-stop VRP**
- **Time Windows & SLA (VRPTW)**
- **Cross-Dock & Multi-Echelon**
- **Multi-Modal & Green Logistics**
- **Dynamic Re-Routing**

Explore each pilot for **business context, math foundations, AI/ML enhancements, and interactive demos.**
""")

st.divider()

st.header("Explore Pilots")
st.page_link("pages/ShipmentPlanner_Basic.py", label="1. Rating & Mode Selection (Cost Engine)")
st.page_link("pages/ShipmentPlanner_VRP.py", label="2. Multi-Stop Routing (VRP)")
st.page_link("pages/ShipmentPlanner_VRPTW.py", label="3. Time Windows & SLA (VRPTW)")
st.page_link("pages/ShipmentPlanner_CrossDock.py", label="4. Cross-Dock & Multi-Echelon")
st.page_link("pages/ShipmentPlanner_MultiModal.py", label="5. Multi-Modal & Green Logistics")
st.page_link("pages/ShipmentPlanner_Dynamic.py", label="6. Dynamic Re-Routing")

st.divider()

st.info("Each pilot includes **business context, math explanation, tools, and AI/ML notes.**")
