import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit.components.v1 import html
import random
import math
import time

st.set_page_config(page_title="SupplyChain.ai — Network Design", layout="wide")

# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------
# Removed: st.page_link("pages/TransportationSuite.py", label="⬅ Back to Transportation Suite")
st.page_link("Home.py", label="🏠 Back to Home")

# ---------------------------------------------------------
# Header Image
# ---------------------------------------------------------
st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Network_icon.png", # Placeholder image for network
          caption="Supply Chain Network Design & Optimization", use_container_width=True)

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("🌐 Network Design & Optimization")

# ---------------------------------------------------------
# Business Context (Practitioner's View)
# ---------------------------------------------------------
st.header("Business Problem: Building the Right Supply Chain Backbone")
st.markdown("""
One of the most strategic decisions in supply chain is **where to locate facilities** (plants, distribution centers, warehouses, cross-docks) and **how to connect them** to suppliers and customers. This is the essence of **Network Design & Optimization**. A poorly designed network can lead to excessive transportation costs, long lead times, poor customer service, and vulnerability to disruptions. The challenge is to find the optimal balance between fixed facility costs, variable transportation costs, and desired service levels.

**Key KPIs Impacted by Network Design:**
-   **Total Logistics Cost:** Sum of facility, transportation, and inventory costs.
-   **Customer Lead Time:** Time from order placement to delivery.
-   **Network Resilience:** Ability to withstand disruptions (e.g., natural disasters, geopolitical events).
-   **Capital Expenditure (CapEx):** Investment in new facilities.
-   **CO2 Emissions:** Environmental impact of the network.
""")

# ---------------------------------------------------------
# Why This Solution Matters (Practitioner's View)
# ---------------------------------------------------------
st.header("Why This Optimization Approach Matters?")
st.markdown("""
Effective network design is a foundational element of supply chain strategy. It allows businesses to:
-   **Significantly Reduce Costs:** By optimizing facility locations and transportation flows.
-   **Improve Service Levels:** Locating facilities closer to customers for faster delivery.
-   **Enhance Supply Chain Resilience:** Designing redundancy and flexibility into the network.
-   **Support Growth & Expansion:** Proactively planning for future demand and market entry.
-   **Minimize Environmental Footprint:** Optimizing routes and facility locations to reduce emissions.
""")

# ---------------------------------------------------------
# Tools Used
# ---------------------------------------------------------
st.header("Tools Used")
st.markdown("""
-   **Python (Pandas, NumPy):** For data handling and basic calculations.
-   **OR-Tools / PuLP / Gurobi:** For solving complex facility location and network flow optimization models.
-   **NetworkX:** For modeling and visualizing network graphs.
-   **Folium / Plotly:** For geographical visualization of proposed networks.
-   **Streamlit:** For building interactive scenario planning dashboards.
""")

# ---------------------------------------------------------
# Math Behind It (Practitioner's View)
# ---------------------------------------------------------
st.header("The Logic Behind the Optimization (Simplified)")
st.markdown(r"""
Network design problems are typically modeled as **Mixed-Integer Linear Programs (MILPs)**. A common formulation is the **Capacitated Facility Location Problem (CFLP)**, which aims to select a subset of potential facility locations to open, assign customers to opened facilities, and determine the flow of goods, all while respecting capacities and minimizing total cost.

**Objective Function (Minimize Total Cost):**
\[
\text{Minimize } \sum_{j \in F} f_j y_j + \sum_{i \in C} \sum_{j \in F} c_{ij} x_{ij}
\]
Where:
-   \(F\) is the set of potential facility locations.
-   \(C\) is the set of customers.
-   \(f_j\) is the fixed cost of opening facility \(j\).
-   \(c_{ij}\) is the cost of serving customer \(i\) from facility \(j\) (e.g., transportation cost).
-   \(y_j\) is a binary decision variable: 1 if facility \(j\) is opened, 0 otherwise.
-   \(x_{ij}\) is the amount of demand from customer \(i\) served by facility \(j\).

**Key Constraints:**
* **Each customer's demand is met:**
    \[
    \sum_{j \in F} x_{ij} = D_i \quad \forall i \in C \quad (\text{where } D_i \text{ is demand of customer } i)
    \]
* **Capacity constraint for opened facilities:**
    \[
    \sum_{i \in C} x_{ij} \le K_j y_j \quad \forall j \in F \quad (\text{where } K_j \text{ is capacity of facility } j)
    \]
* **Non-negativity and binary constraints:**
    \[
    x_{ij} \ge 0, \quad y_j \in \{0, 1\}
    \]
This demo would illustrate the impact of changing parameters (e.g., number of facilities, fixed costs) on the optimal network structure and total cost.
""")

# ---------------------------------------------------------
# AI & ML Angle
# ---------------------------------------------------------
st.header("How AI & ML Enhance Network Design")
st.markdown("""
-   **Predictive Analytics:** ML models can forecast future demand patterns, demographic shifts, or supplier reliability, which are critical inputs for network design.
-   **Geospatial AI:** Advanced AI can analyze vast amounts of geospatial data (e.g., traffic patterns, infrastructure, competitor locations) to identify optimal facility sites.
-   **Reinforcement Learning:** Could explore complex network configurations and learn optimal design policies over time, especially for dynamic or uncertain environments.
-   **Generative AI/LLMs:** Can assist in rapidly generating and evaluating alternative network scenarios, providing insights into trade-offs and potential risks, accelerating the strategic planning process.
""")

# ---------------------------------------------------------
# Fun Fact
# ---------------------------------------------------------
st.header("Fun Fact")
st.markdown("""
> During World War II, **operations research (the field behind many of these optimization problems)** saw a massive surge in development. Its techniques were used to optimize convoy routing, submarine search patterns, and logistics networks, directly impacting the war effort and laying the groundwork for modern supply chain optimization.
""")

# ---------------------------------------------------------
# DEMO PLACEHOLDER
# ---------------------------------------------------------
st.header("Demo")
st.info("Interactive network design optimization demo **coming soon**.")
st.markdown("""
This demo will allow you to:
- Define potential facility locations and customer demand points.
- Adjust parameters like fixed facility costs and transportation costs.
- Visualize the optimal network structure (which facilities to open, which customers to serve from which facility).
- See the impact on total logistics costs and service coverage.
""")
