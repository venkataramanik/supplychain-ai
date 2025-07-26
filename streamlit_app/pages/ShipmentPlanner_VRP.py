import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="VRP + Multi-Stop Routing", layout="wide")
# Navigation link to go back
st.page_link("../TransportationSuite.py", label="⬅ Back to Transportation Suite")


# --- TITLE ---
st.title("🗺 VRP + Multi-Stop Routing (Classic Vehicle Routing Problem)")

# --- BUSINESS CONTEXT ---
st.header("Business Problem")
st.markdown("""
For a single vehicle serving multiple customers, the **order of stops** dramatically 
impacts total distance and cost. Without optimization, companies often travel **10–20% extra miles**, 
resulting in higher fuel costs, missed delivery windows, and unnecessary emissions.

**KPIs Impacted:**
- **Total transportation cost.**
- **Distance traveled (fuel consumption).**
- **On-time delivery %.**
""")

# --- WHY THIS APPROACH ---
st.header("Why This Approach?")
st.markdown("""
The **Vehicle Routing Problem (VRP)**, first formulated by **Dantzig and Ramser in 1959**, 
remains central to logistics optimization. Solving VRP allows companies to:
- Minimize total travel distance.
- Reduce fuel and driver costs.
- Improve fleet utilization.
""")

# --- TOOLS USED ---
st.header("Tools Used")
st.markdown("""
- **Google OR-Tools:** Open-source solver for VRP variants (used by major tech companies).
- **pandas:** To manage stop and distance data.
- **Folium/Plotly:** For route visualization on maps.
""")

# --- MATH BEHIND IT ---
st.header("Math Behind It")
st.markdown(r"""
VRP can be modeled as a graph optimization problem:

\[
Minimize \ \sum_{(i,j)} C_{ij} x_{ij}
\]

Where:
- **\(C_{ij}\)** = cost (distance/time) between locations i and j.
- **\(x_{ij}\)** = 1 if the route goes from i to j, 0 otherwise.
""")

# --- AI & ML ANGLE ---
st.header("How AI & ML Enhance VRP")
st.markdown("""
- **ML Models:** Predict travel times (ETA) using real-time traffic and weather data.
- **LLMs:** Generate constraint logic or optimize OR-Tools parameters on the fly.
- **Reinforcement Learning:** Adaptive route planning as conditions change dynamically.
""")

# --- FUN FACT ---
st.header("Fun Fact")
st.markdown("""
> The VRP problem has inspired algorithms like **Google Maps routing engine** and 
delivery route optimization for **FedEx and UPS**.
""")

# --- DEMO PLACEHOLDER ---
st.header("Demo")
st.info("Interactive multi-stop route optimization demo **coming soon**.")

