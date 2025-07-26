import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Transportation Optimization Suite", layout="wide")

st.title("🚛 Transportation Optimization Suite")

# ---------- Robust image loader with centered layout ----------
def show_image_safe(relative_path_from_images: str, caption: str):
    """
    Displays an image from the ../images folder, centered using columns.
    """
    try:
        streamlit_app_root = Path(__file__).resolve().parents[1]
        img_path = streamlit_app_root / "images" / relative_path_from_images

        if img_path.exists():
            col1, col2, col3 = st.columns([1, 4, 1])  # Center image
            with col2:
                st.image(str(img_path), caption=caption, use_container_width=True)
        else:
            st.warning(f"Image not found at: {img_path}")
    except Exception as e:
        st.warning(f"Could not load image due to: {e}")

# Display the image
show_image_safe(
    "Supply-chain-shipping-modes.jpg",
    caption="Example of a transportation optimization network."
)

# ----------- Page Content -----------
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
